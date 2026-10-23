import asyncio
import logging
import threading
import time
from collections import deque
from typing import Optional, Callable

import numpy as np

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

from .config import Config

logger = logging.getLogger(__name__)


class VADProcessor:
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 1,
                 frame_ms: int = 30, silence_padding_ms: int = 400,
                 min_speech_duration: float = 0.3, max_inactivity: float = 3.0,
                 energy_threshold: float = 0.01):
        self.sample_rate = sample_rate
        self.silence_padding_ms = silence_padding_ms
        self.min_speech_duration = min_speech_duration
        self.max_inactivity = max_inactivity
        self.energy_threshold = energy_threshold
        self.frame_size = int(sample_rate * frame_ms / 1000)

        self._ring = bytearray()
        self._speaking = False
        self._speech_start_idx = 0
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._total_frames = 0

    @property
    def enabled(self) -> bool:
        return True

    def process(self, audio_bytes: bytes) -> Optional[bytes]:
        self._ring.extend(audio_bytes)

        frame_n = self.frame_size
        pos = 0
        while pos + frame_n <= len(self._ring):
            frame = self._ring[pos:pos + frame_n]
            frame_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            energy = float(np.sqrt(np.mean(frame_np ** 2)))
            is_speech = energy > (self.energy_threshold * 32768)

            if is_speech:
                if not self._speaking:
                    self._speaking = True
                    self._speech_start_idx = pos
                self._speech_frame_count += 1
                self._silence_frame_count = 0
            elif self._speaking:
                self._silence_frame_count += 1

            if self._speaking:
                self._total_frames += 1
                silent_ms = self._silence_frame_count * 30
                if silent_ms >= self.silence_padding_ms:
                    if self._speech_frame_count * 30 / 1000.0 >= self.min_speech_duration:
                        end_idx = pos + frame_n - self._silence_frame_count * frame_n
                        end_idx = max(self._speech_start_idx + frame_n, end_idx)
                        segment = bytes(self._ring[self._speech_start_idx:end_idx])
                        del self._ring[:end_idx]
                        self._reset()
                        return segment
                    else:
                        del self._ring[:pos + frame_n]
                        self._reset()

            pos += frame_n

        trim_threshold = int(self.max_inactivity * self.sample_rate * 2)
        if not self._speaking and len(self._ring) > trim_threshold:
            excess = len(self._ring) - max(frame_n * 5, trim_threshold // 2)
            if excess > 0:
                del self._ring[:excess]

        if self._speaking and len(self._ring) > trim_threshold:
            duration = self._speech_frame_count * 30 / 1000.0
            if duration >= self.min_speech_duration:
                segment = bytes(self._ring[self._speech_start_idx:])
                self._ring.clear()
                self._reset()
                return segment if len(segment) >= frame_n else None
            else:
                self._ring.clear()
                self._reset()

        return None

    def _reset(self):
        self._speaking = False
        self._speech_start_idx = 0
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._total_frames = 0


class AudioCapture:
    def __init__(self, config: Config):
        self.config = config
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        self._audio_buffer: deque = deque()
        self._lock = threading.Lock()

    def _select_device(self) -> Optional[int]:
        if self.config.audio_device_index is not None:
            return self.config.audio_device_index

        if not HAS_SOUNDDEVICE:
            return None

        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if not dev["max_input_channels"]:
                continue
            name = dev["name"].lower()
            if self.config.audio_source:
                if self.config.audio_source.lower() in name:
                    return idx
            if "translate_sink" in name and "monitor" in name:
                return idx
            if "monitor" in name and "alsa_output" in name:
                return idx

        for idx, dev in enumerate(devices):
            if not dev["max_input_channels"]:
                continue
            name = dev["name"].lower()
            if "monitor" in name:
                return idx
            if "pipewire" in name or "pulse" in name:
                return idx
            if "default" in name:
                return idx

        return 0

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio status: %s", status)
        with self._lock:
            self._audio_buffer.append(indata.copy())

    def _capture_loop(self):
        logger.info("Audio capture starting...")
        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                device=self._select_device(),
                callback=self._audio_callback,
                blocksize=int(self.config.sample_rate * self.config.chunk_duration),
                dtype="float32",
            ):
                logger.info("Audio stream opened on device %s", self._select_device())
                while self._running:
                    sd.sleep(100)
        except Exception as e:
            logger.error("Audio capture error: %s", e)
            self._running = False

    def start(self, callback: Callable[[np.ndarray], None]):
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        def _process_loop():
            while self._running:
                with self._lock:
                    if self._audio_buffer:
                        chunk = self._audio_buffer.popleft()
                    else:
                        chunk = None
                if chunk is not None and self._callback:
                    try:
                        self._callback(chunk)
                    except Exception as e:
                        logger.error("Callback error: %s", e)
                else:
                    time.sleep(0.01)

        processing_thread = threading.Thread(target=_process_loop, daemon=True)
        processing_thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def list_devices(self) -> list:
        if not HAS_SOUNDDEVICE:
            return []
        return list(sd.query_devices())
