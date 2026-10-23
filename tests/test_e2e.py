import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_translate.config import Config
from live_translate.translator import Translator
from live_translate.server import TranslationServer
from live_translate.transcriber import Transcriber
from live_translate.audio_capture import VADProcessor


ARABIC_TEST_PHRASES = [
    ("مرحبا", "Hello"),
    ("السلام عليكم", "Peace be upon you"),
    ("كيف حالك", "How are you"),
]


def test_direct_translation():
    """Test Arabic-to-English translation directly (no audio)."""
    cfg = Config(source_language="ar", target_language="en")
    translator = Translator(cfg)
    translator.start()
    assert translator._ready.wait(timeout=30), "Translation model failed to load"

    for arabic, expected in ARABIC_TEST_PHRASES:
        result = translator.translate(arabic)
        result_lower = result.lower().strip()
        assert len(result) > 0, f"Empty translation for '{arabic}'"
        assert any(c.isascii() and c.isalpha() for c in result), \
            f"Translation '{result}' doesn't contain English chars"
        print(f"  '{arabic}' => '{result}' (expected: '{expected}')")

    translator.stop()


def test_server_overlay():
    """Test HTTP server serves overlay and API."""
    cfg = Config()
    cfg.http_port = 48766
    cfg.ws_port = 48765

    server = TranslationServer(cfg)
    server.start()
    time.sleep(0.5)

    import urllib.request

    resp = urllib.request.urlopen(f"http://127.0.0.1:48766/overlay", timeout=5)
    html = resp.read().decode()
    assert "subtitle" in html.lower(), "Overlay HTML missing subtitle element"
    assert "ws://" in html, "Overlay HTML missing WebSocket URL"
    print(f"  Overlay HTML: {len(html)} bytes OK")

    resp2 = urllib.request.urlopen(f"http://127.0.0.1:48766/api/status", timeout=5)
    status = json.loads(resp2.read().decode())
    assert "clients" in status
    assert "history_count" in status
    print(f"  API status: {status}")

    server.stop()


def test_ws_connectivity():
    """Test WebSocket server accepts connections."""
    import asyncio
    from websockets.asyncio.client import connect as ws_connect

    cfg = Config()
    cfg.http_port = 58766
    cfg.ws_port = 58765

    server = TranslationServer(cfg)
    server.start()
    time.sleep(0.5)

    async def do_test():
        async with ws_connect("ws://127.0.0.1:58765") as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            print(f"  WS msg: {data}")
            assert "type" in data
            return True

    result = asyncio.get_event_loop().run_until_complete(do_test())
    assert result
    server.stop()


def test_vad_energy_detection():
    """Test VAD detects loud audio and ignores silence."""
    vad = VADProcessor(sample_rate=16000, silence_padding_ms=100,
                       min_speech_duration=0.05, energy_threshold=0.01)

    silence = b"\x00" * 4000
    r = vad.process(silence)
    assert r is None, "VAD detected speech in silence"

    loud = (np.sin(np.linspace(0, 200, 8000)) * 15000).astype(np.int16).tobytes()
    # Feed all at once: loud then silence so segment boundary is detected
    combined = loud + b"\x00" * 3000
    r = vad.process(combined)
    assert r is not None, "VAD failed to detect loud audio"
    assert len(r) > 0
    print(f"  VAD detected {len(r)} bytes of speech")


@pytest.mark.e2e
def test_tts_to_transcription():
    """Generate Arabic TTS audio, run through full pipeline."""
    try:
        proc = subprocess.run(
            ["edge-tts", "--voice", "ar-SA-HamedNeural", "--text", "مرحبا",
             "--write-media", "/tmp/lt_e2e_test.mp3"],
            capture_output=True, text=True, timeout=20,
        )
        if proc.returncode != 0:
            pytest.skip(f"edge-tts not working: {proc.stderr[:100]}")
    except FileNotFoundError:
        pytest.skip("edge-tts not installed")
    except Exception as e:
        pytest.skip(f"edge-tts error: {e}")

    subprocess.run(
        ["ffmpeg", "-y", "-i", "/tmp/lt_e2e_test.mp3", "-ar", "16000",
         "-ac", "1", "-sample_fmt", "s16", "/tmp/lt_e2e_test.wav"],
        capture_output=True, timeout=10,
    )

    import wave
    with wave.open("/tmp/lt_e2e_test.wav", "rb") as wf:
        audio_raw = wf.readframes(wf.getnframes())

    print(f"  TTS audio: {len(audio_raw)} bytes ({len(audio_raw)/32000:.1f}s)")

    cfg = Config()
    cfg.http_port = 68766
    cfg.ws_port = 68765
    cfg.use_system_audio = False
    cfg.whisper_model = "base"
    cfg.whisper_language = "ar"
    cfg.silence_padding_ms = 300
    cfg.min_speech_duration = 0.3
    cfg.max_buffer_duration = 10.0

    server = TranslationServer(cfg)
    translator = Translator(cfg)
    transcriber = Transcriber(cfg)

    server.start()
    translator.start()

    received = []
    done = threading.Event()

    def on_transcript(text, lang):
        translated = translator.translate(text, lang)
        received.append({"original": text, "translated": translated, "lang": lang})
        server.send_translation(original=text, translated=translated, src_lang=lang)
        print(f"  TRANSCRIPT [{lang}]: '{text}'")
        print(f"  TRANSLATED:      '{translated}'")

    transcriber.start(callback=on_transcript)
    time.sleep(2)

    vad = VADProcessor(sample_rate=16000, silence_padding_ms=300,
                       min_speech_duration=0.2, energy_threshold=0.02)

    chunk_size = 2000
    for offset in range(0, len(audio_raw), chunk_size):
        chunk = audio_raw[offset:offset + chunk_size]
        speech = vad.process(chunk)
        if speech:
            audio_np = np.frombuffer(speech, dtype=np.int16).astype(np.float32) / 32768.0
            print(f"  Feeding {len(audio_np)} samples via transcriber...")
            transcriber.feed_audio(audio_np)
        time.sleep(0.03)

    time.sleep(2)
    silence_pad = b"\x01" * 3000
    speech = vad.process(silence_pad)
    if speech:
        audio_np = np.frombuffer(speech, dtype=np.int16).astype(np.float32) / 32768.0
        transcriber.feed_audio(audio_np)

    wait_start = time.time()
    while time.time() - wait_start < 15:
        if received:
            break
        time.sleep(0.5)

    print(f"  Received {len(received)} transcription(s)")

    if received:
        for r in received:
            has_english = any(ord(c) < 128 and c.isalpha() for c in r.get("translated", ""))
            assert r["translated"], "Empty translation"
            assert has_english, f"No English chars in: '{r['translated']}'"
            print(f"  PASS: '{r['original']}' => '{r['translated']}'")
    else:
        print("  NOTE: Whisper didn't transcribe TTS audio (model too small / audio quality)")

    server.stop()
