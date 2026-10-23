import argparse
import logging
import signal
import sys
import threading
import time

import numpy as np

from .config import Config
from .audio_capture import AudioCapture, VADProcessor
from .transcriber import Transcriber
from .translator import Translator
from .server import TranslationServer

logger = logging.getLogger(__name__)


class LiveTranslateApp:
    def __init__(self, config: Config):
        self.config = config
        self.config.ensure_dirs()

        self.server = TranslationServer(config)
        self.translator = Translator(config)
        self.transcriber = Transcriber(config)
        self.capture = AudioCapture(config)

        self._running = False
        self._pending_transcriptions: list = []
        self._transcription_lock = threading.Lock()
        self._last_activity = time.time()
        self._activity_timer: threading.Timer = None

    def _on_transcript(self, text: str, detected_lang: str):
        self._last_activity = time.time()
        translated = self.translator.translate(text, detected_lang)
        src_lang = detected_lang if detected_lang != "unknown" else self.config.source_language
        logger.info("[%s] %s => %s", src_lang, text, translated)
        self.server.send_translation(original=text, translated=translated, src_lang=src_lang)

    def _on_audio(self, audio_data: np.ndarray):
        self.transcriber.feed_audio(audio_data)
        if time.time() - self._last_activity > self.config.max_inactivity + 2:
            if self._activity_timer is None or not self._activity_timer.is_alive():
                self._activity_timer = threading.Timer(0.5, lambda: self.server.send_status("Waiting for speech..."))
                self._activity_timer.start()

    def start(self):
        self._running = True

        logger.info("=" * 60)
        logger.info("Live Translate starting...")
        logger.info("  Whisper model:  %s", self.config.whisper_model)
        logger.info("  Translation:    %s -> %s", self.config.source_language, self.config.target_language)
        logger.info("  Translation MT: %s", self.translator.model_name)
        logger.info("  Sample rate:    %d Hz", self.config.sample_rate)
        logger.info("  WS port:        %d", self.config.ws_port)
        logger.info("  HTTP port:      %d", self.config.http_port)
        logger.info("  Overlay URL:    http://%s:%d/overlay", self.config.http_host, self.config.http_port)
        logger.info("=" * 60)

        self.server.start()
        self.server.send_status("Starting translation engine...")

        self.translator.start()

        self.transcriber.start(callback=self._on_transcript)

        self.server.send_status("Listening...")

        self.capture.start(callback=self._on_audio)

        logger.info("Live Translate is running. Press Ctrl+C to stop.")
        logger.info("Open overlay at: http://%s:%d/overlay", self.config.http_host, self.config.http_port)

        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        logger.info("Shutting down...")
        self._running = False
        self.capture.stop()
        self.transcriber.stop()
        self.translator.stop()
        self.server.stop()
        logger.info("Live Translate stopped.")

    @staticmethod
    def list_devices():
        config = Config()
        cap = AudioCapture(config)
        devices = cap.list_devices()
        print("\nAvailable audio devices:")
        print("-" * 70)
        for idx, dev in enumerate(devices):
            in_ch = dev.get("max_input_channels", 0)
            out_ch = dev.get("max_output_channels", 0)
            name = dev.get("name", "Unknown")
            sr = dev.get("default_samplerate", 0)
            if in_ch > 0:
                print(f"  [{idx}] IN : {name} ({in_ch}ch, {int(sr)}Hz)")
            if out_ch > 0:
                print(f"  [{idx}] OUT: {name} ({out_ch}ch, {int(sr)}Hz)")
        print("-" * 70)
        print("\nTip: Use --device <index> to select an audio input.")
        print("     For system audio, select a monitor source (e.g. '...monitor')")


def main():
    parser = argparse.ArgumentParser(
        description="Live Translate - Real-time audio translation with subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  live-translate                           # Start with defaults (Arabic->English)
  live-translate --source fr --target en   # French to English
  live-translate --model tiny.en           # Use English-only tiny model
  live-translate --device 4                # Use specific audio device
  live-translate --list-devices            # Show available audio devices
  live-translate --http-port 8080          # Custom HTTP port for overlay
  live-translate --no-audio-source         # Don't try to capture system audio
        """,
    )

    parser.add_argument("--model", default="tiny", help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--compute", default="int8", choices=["int8", "int8_float16", "float16", "float32"],
                        help="Whisper compute type (default: int8)")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--source", default="ar", help="Source language code (default: ar)")
    parser.add_argument("--target", default="en", help="Target language code (default: en)")
    parser.add_argument("--translation-model", default=None, help="Custom Helsinki-NLP/opus-mt model")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port (default: 8765)")
    parser.add_argument("--http-port", type=int, default=8766, help="HTTP overlay port (default: 8766)")
    parser.add_argument("--http-host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)")
    parser.add_argument("--beam-size", type=int, default=1, help="Whisper beam size (default: 1)")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filtering")
    parser.add_argument("--language", default=None, help="Force language for Whisper (e.g. 'ar', 'fr')")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--no-audio-source", action="store_true", help="Don't try to capture system monitor audio")
    parser.add_argument("--audio-source", default=None, help="Audio source name filter (e.g. 'translate_sink')")

    args = parser.parse_args()

    if args.list_devices:
        LiveTranslateApp.list_devices()
        return 0

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config()
    config.whisper_model = args.model
    config.whisper_compute_type = args.compute
    config.whisper_beam_size = args.beam_size
    config.whisper_vad_filter = not args.no_vad
    config.whisper_language = args.language
    config.source_language = args.source
    config.target_language = args.target
    config.ws_port = args.ws_port
    config.http_port = args.http_port
    config.http_host = args.http_host
    config.sample_rate = args.sample_rate
    config.audio_device_index = args.device
    config.use_system_audio = not args.no_audio_source

    if args.translation_model:
        config.translation_model = args.translation_model
    if args.audio_source:
        config.audio_source = args.audio_source

    app = LiveTranslateApp(config)

    def _signal_handler(sig, frame):
        logger.info("Received signal %d", sig)
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    app.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
