import logging
import threading
import time
from collections import deque
from typing import Optional, Callable

import numpy as np

from .config import Config
from .audio_capture import VADProcessor

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._vad = VADProcessor(
            sample_rate=config.sample_rate,
            silence_padding_ms=config.silence_padding_ms,
            min_speech_duration=config.min_speech_duration,
            max_inactivity=config.max_inactivity,
        )
        self._audio_chunks: deque = deque()
        self._lock = threading.Lock()
        self._callback: Optional[Callable[[str, str], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_transcript = ""
        self._transcript_time = 0.0
        self._model_ready = threading.Event()

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model '%s' (compute=%s, device=%s)...",
                self.config.whisper_model,
                self.config.whisper_compute_type,
                self.config.whisper_device,
            )
            self._model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
                num_workers=self.config.whisper_num_workers,
                download_root=self.config.cache_dir,
            )
            logger.info("Whisper model loaded successfully")
            self._model_ready.set()
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise

    def start(self, callback: Callable[[str, str], None]):
        self._callback = callback
        self._running = True

        load_thread = threading.Thread(target=self._load_model, daemon=True)
        load_thread.start()

        self._thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def feed_audio(self, audio_data: np.ndarray):
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        with self._lock:
            self._audio_chunks.append(audio_bytes)

    def _transcription_loop(self):
        self._model_ready.wait(timeout=60.0)

        ring_buffer = bytearray()
        last_process = time.monotonic()

        while self._running:
            with self._lock:
                while self._audio_chunks:
                    ring_buffer.extend(self._audio_chunks.popleft())

            now = time.monotonic()
            elapsed = now - last_process

            if elapsed < 0.5 or len(ring_buffer) < self.config.sample_rate * 2:
                time.sleep(0.05)
                continue

            max_bytes = int(self.config.sample_rate * self.config.max_buffer_duration * 2)
            if len(ring_buffer) > max_bytes:
                del ring_buffer[: len(ring_buffer) - max_bytes]

            last_process = now

            try:
                result = self._vad.process(bytes(ring_buffer))
                if result is None:
                    continue

                audio_np = np.frombuffer(result, dtype=np.int16).astype(np.float32) / 32768.0

                if len(audio_np) < self.config.sample_rate * self.config.min_speech_duration:
                    continue

                segments, info = self._model.transcribe(
                    audio_np,
                    beam_size=self.config.whisper_beam_size,
                    language=self.config.whisper_language,
                    vad_filter=self.config.whisper_vad_filter,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                )

                detected_lang = info.language if info else "unknown"
                text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

                if not text:
                    continue

                if len(text) > self.config.max_subtitle_length:
                    text = text[:self.config.max_subtitle_length] + "..."

                if text and text != self._last_transcript:
                    self._last_transcript = text
                    self._transcript_time = time.time()
                    if self._callback:
                        try:
                            self._callback(text, detected_lang)
                        except Exception as e:
                            logger.error("Transcript callback error: %s", e)

            except Exception as e:
                logger.error("Transcription error: %s", e)
