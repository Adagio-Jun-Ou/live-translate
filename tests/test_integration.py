import asyncio
import json
import time
import numpy as np
import urllib.request

from live_translate.config import Config
from live_translate.audio_capture import AudioCapture
from live_translate.transcriber import Transcriber
from live_translate.translator import Translator
from live_translate.server import TranslationServer


def test_integration():
    cfg = Config()
    cfg.http_port = 28766
    cfg.ws_port = 28765
    cfg.use_system_audio = False
    cfg.audio_device_index = None

    server = TranslationServer(cfg)
    translator = Translator(cfg)
    transcriber = Transcriber(cfg)

    server.start()
    translator.start()

    def on_transcript(text, lang):
        translated = translator.translate(text, lang)
        server.send_translation(original=text, translated=translated, src_lang=lang)

    transcriber.start(callback=on_transcript)
    time.sleep(2)

    for i in range(5):
        noise = np.random.randn(8000).astype(np.float32) * 0.01
        transcriber.feed_audio(noise)
        time.sleep(0.3)

    from websockets.asyncio.client import connect as ws_connect
    async def test_ws():
        async with ws_connect("ws://127.0.0.1:28765") as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            assert "type" in data

    asyncio.get_event_loop().run_until_complete(test_ws())

    req = urllib.request.Request("http://127.0.0.1:28766/overlay")
    resp = urllib.request.urlopen(req, timeout=5)
    html = resp.read().decode()
    assert "subtitle" in html.lower()

    req2 = urllib.request.Request("http://127.0.0.1:28766/api/status")
    resp2 = urllib.request.urlopen(req2, timeout=5)
    status = json.loads(resp2.read().decode())
    assert "clients" in status

    server.stop()
