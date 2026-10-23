import pytest
import numpy as np
import threading
import time

from live_translate.config import Config
from live_translate.audio_capture import VADProcessor
from live_translate.translator import Translator


class TestConfig:
    def test_default_values(self):
        cfg = Config()
        assert cfg.whisper_model == "tiny"
        assert cfg.source_language == "ar"
        assert cfg.target_language == "en"
        assert cfg.sample_rate == 16000
        assert cfg.ws_port == 8765
        assert cfg.http_port == 8766

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LT_SOURCE_LANGUAGE", "fr")
        monkeypatch.setenv("LT_TARGET_LANGUAGE", "es")
        monkeypatch.setenv("LT_WS_PORT", "9999")
        cfg = Config.from_env()
        assert cfg.source_language == "fr"
        assert cfg.target_language == "es"
        assert cfg.ws_port == 9999

    def test_ensure_dirs(self, tmp_path):
        cfg = Config()
        cfg.data_dir = str(tmp_path / "data")
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.ensure_dirs()
        import os
        assert os.path.isdir(cfg.data_dir)
        assert os.path.isdir(cfg.cache_dir)


class TestVADProcessor:
    def test_creation(self):
        vad = VADProcessor(sample_rate=16000)
        assert vad.sample_rate == 16000
        assert vad.frame_size == 480

    def test_creation_any_rate(self):
        vad = VADProcessor(sample_rate=22050)
        assert vad.enabled

    def test_process_silence_returns_none(self):
        vad = VADProcessor(sample_rate=16000, silence_padding_ms=100, energy_threshold=0.5)
        silence = b"\x00" * 480
        result = vad.process(silence)
        assert result is None

    def test_process_loud_audio_detects_speech(self):
        vad = VADProcessor(sample_rate=16000, silence_padding_ms=100,
                           min_speech_duration=0.1, energy_threshold=0.01)
        import numpy as np
        loud = (np.sin(np.linspace(0, 100, 4000)) * 20000).astype(np.int16).tobytes()
        silence = b"\x00" * 8000
        result = None
        for _ in range(5):
            result = vad.process(loud) or result
        # Feed silence to end the speech segment
        result = vad.process(silence) or result
        assert result is not None
        assert len(result) > 0

    def test_feed_small_chunks(self):
        vad = VADProcessor(sample_rate=16000, frame_ms=30)
        chunk = b"\x00" * 100
        for _ in range(5):
            result = vad.process(chunk)
        assert True


class TestTranslator:
    def test_creation(self):
        cfg = Config()
        t = Translator(cfg)
        assert t._source_lang == "ar"
        assert t._target_lang == "en"

    def test_resolve_model_fr_en(self):
        cfg = Config(source_language="fr", target_language="en")
        t = Translator(cfg)
        assert t._model_name == "Helsinki-NLP/opus-mt-fr-en"

    def test_translate_empty(self, monkeypatch):
        cfg = Config()
        t = Translator(cfg)
        t._ready.set()
        assert t.translate("") == ""
        assert t.translate("   ") == "   "

    def test_supported_pairs(self):
        pairs = Translator.supported_pairs()
        assert isinstance(pairs, list)
        assert ("ar", "en") in pairs
        assert ("fr", "en") in pairs

    def test_model_name_property(self):
        cfg = Config(source_language="zh", target_language="en")
        t = Translator(cfg)
        assert t.model_name == "Helsinki-NLP/opus-mt-zh-en"


class TestLiveTranslateAppIntegration:
    def test_imports(self):
        from live_translate import (
            Config, AudioCapture, Transcriber, Translator, TranslationServer
        )
        assert Config is not None
        assert AudioCapture is not None
        assert Transcriber is not None
        assert Translator is not None
        assert TranslationServer is not None

    def test_version(self):
        from live_translate import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0
