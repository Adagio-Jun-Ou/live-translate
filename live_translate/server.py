import asyncio
import json
import logging
import threading
import time
from http import HTTPStatus
from typing import Optional, Set

import websockets
from websockets.asyncio.server import serve

logger = logging.getLogger(__name__)

OVERLAY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Translate Overlay</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: rgba(0, 0, 0, 0.75);
  --text: #ffffff;
  --text-secondary: rgba(255, 255, 255, 0.7);
  --accent: #4fc3f7;
  --font-size: 20px;
  --panel-bg: rgba(10, 10, 20, 0.85);
  --border: rgba(255, 255, 255, 0.12);
  --shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
}

body {
  background: transparent;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  overflow: hidden;
  user-select: none;
  height: 100vh;
  width: 100vw;
  color: var(--text);
}

#subtitle-container {
  position: fixed;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%);
  max-width: 85vw;
  min-width: 300px;
  z-index: 999999;
  cursor: move;
  transition: opacity 0.15s;
}

#subtitle-panel {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 22px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: var(--shadow);
  line-height: 1.5;
}

#subtitle-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--text-secondary);
  letter-spacing: 0.5px;
}

#subtitle-header .lang-badge {
  background: var(--accent);
  color: #000;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
  font-size: 10px;
  text-transform: uppercase;
}

#subtitle-header .status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #4caf50;
  display: inline-block;
  margin-right: 4px;
}

#subtitle-header .status-dot.inactive { background: #ff9800; }
#subtitle-header .status-dot.error { background: #f44336; }

#translated-text {
  font-size: var(--font-size);
  font-weight: 500;
  color: var(--text);
  letter-spacing: 0.2px;
  margin-bottom: 6px;
}

#original-text {
  font-size: calc(var(--font-size) * 0.75);
  color: var(--text-secondary);
  font-style: italic;
  display: none;
}

#subtitle-container.show-original #original-text { display: block; }

.settings-bar {
  position: fixed;
  top: 12px;
  right: 12px;
  display: flex;
  gap: 6px;
  z-index: 999999;
}

.settings-bar button {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 12px;
  cursor: pointer;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: background 0.15s;
}

.settings-bar button:hover { background: rgba(255, 255, 255, 0.12); }
.settings-bar button.active { background: var(--accent); color: #000; }

.history-modal {
  position: fixed;
  bottom: 200px;
  left: 50%;
  transform: translateX(-50%);
  max-width: 70vw;
  min-width: 400px;
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  backdrop-filter: blur(20px);
  box-shadow: var(--shadow);
  display: none;
  z-index: 1000000;
  max-height: 300px;
  overflow-y: auto;
  font-size: 14px;
}

.history-modal.visible { display: block; }
.history-modal .entry { padding: 6px 0; border-bottom: 1px solid var(--border); }
.history-modal .entry .time { font-size: 10px; color: var(--text-secondary); }
.history-modal .entry .tr { color: var(--text); }
.history-modal .entry .orig { color: var(--text-secondary); font-size: 12px; }
</style>
</head>
<body>

<div class="settings-bar">
  <button id="btn-pause" title="Pause/Resume">[Pause]</button>
  <button id="btn-original" title="Toggle original text">[Original]</button>
  <button id="btn-history" title="Show history">[History]</button>
  <button id="btn-font-up" title="Increase font">[A+]</button>
  <button id="btn-font-down" title="Decrease font">[A-]</button>
</div>

<div id="subtitle-container" data-draggable="true">
  <div id="subtitle-panel">
    <div id="subtitle-header">
      <span><span class="status-dot" id="status-dot"></span> <span id="status-label">Connecting...</span></span>
      <span class="lang-badge" id="lang-badge">EN</span>
    </div>
    <div id="translated-text">Waiting for audio...</div>
    <div id="original-text"></div>
  </div>
</div>

<div class="history-modal" id="history-modal"></div>

<script>
(function() {
  const WS_URL = 'ws://127.0.0.1:8765';
  let ws = null;
  let reconnectTimer = null;
  let paused = false;
  let showOriginal = false;
  let fontSize = 20;
  let history = [];
  let lastActivity = Date.now();
  let statusCheck = null;

  const el = {
    translated: document.getElementById('translated-text'),
    original: document.getElementById('original-text'),
    container: document.getElementById('subtitle-container'),
    statusDot: document.getElementById('status-dot'),
    statusLabel: document.getElementById('status-label'),
    langBadge: document.getElementById('lang-badge'),
    historyModal: document.getElementById('history-modal'),
    btnPause: document.getElementById('btn-pause'),
    btnOriginal: document.getElementById('btn-original'),
    btnHistory: document.getElementById('btn-history'),
    btnFontUp: document.getElementById('btn-font-up'),
    btnFontDown: document.getElementById('btn-font-down'),
  };

  function connect() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    if (ws) { try { ws.close(); } catch(e) {} }

    try {
      ws = new WebSocket(WS_URL);
    } catch(e) {
      scheduleReconnect();
      return;
    }

    ws.onopen = function() {
      el.statusDot.className = 'status-dot';
      el.statusLabel.textContent = 'Connected';
      el.langBadge.textContent = 'EN';
    };

    ws.onmessage = function(evt) {
      try {
        const data = JSON.parse(evt.data);
        handleMessage(data);
      } catch(e) {
        console.error('Parse error:', e);
      }
    };

    ws.onclose = function() {
      el.statusDot.className = 'status-dot inactive';
      el.statusLabel.textContent = 'Disconnected';
      scheduleReconnect();
    };

    ws.onerror = function() {
      el.statusDot.className = 'status-dot error';
      el.statusLabel.textContent = 'Error';
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setInterval(function() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        clearInterval(reconnectTimer);
        reconnectTimer = null;
        return;
      }
      connect();
    }, 3000);
  }

  function handleMessage(data) {
    lastActivity = Date.now();
    el.statusDot.className = 'status-dot';
    el.statusLabel.textContent = 'Live';

    if (data.type === 'translation') {
      if (!paused) {
        el.translated.textContent = data.translated || data.text || '';
        el.original.textContent = data.original || '';
      }
      if (data.src_lang) {
        el.langBadge.textContent = data.src_lang.toUpperCase();
      }
      history.unshift({
        time: new Date().toLocaleTimeString(),
        tr: data.translated || data.text,
        orig: data.original || '',
      });
      if (history.length > 50) history.length = 50;
    } else if (data.type === 'status') {
      el.statusLabel.textContent = data.status || data.text || '';
      if (data.lang) el.langBadge.textContent = data.lang.toUpperCase();
    }
  }

  function updateFontSize(delta) {
    fontSize = Math.max(12, Math.min(48, fontSize + delta));
    document.documentElement.style.setProperty('--font-size', fontSize + 'px');
  }

  function togglePause() {
    paused = !paused;
    el.btnPause.textContent = paused ? '[Resume]' : '[Pause]';
    el.btnPause.classList.toggle('active', paused);
    if (paused) {
      el.statusLabel.textContent = 'Paused';
    }
  }

  function toggleOriginal() {
    showOriginal = !showOriginal;
    el.btnOriginal.classList.toggle('active', showOriginal);
    if (showOriginal) {
      el.container.classList.add('show-original');
    } else {
      el.container.classList.remove('show-original');
    }
  }

  function toggleHistory() {
    const modal = el.historyModal;
    if (modal.classList.contains('visible')) {
      modal.classList.remove('visible');
      return;
    }
    let html = '<div style="font-weight:600;margin-bottom:8px;">Translation History</div>';
    if (history.length === 0) {
      html += '<div class="entry"><div class="tr">No translations yet</div></div>';
    } else {
      for (const h of history) {
        html += '<div class="entry">';
        html += '<div class="time">' + h.time + '</div>';
        html += '<div class="tr">' + h.tr + '</div>';
        if (h.orig) {
          html += '<div class="orig">' + h.orig + '</div>';
        }
        html += '</div>';
      }
    }
    modal.innerHTML = html;
    modal.classList.add('visible');
  }

  el.btnPause.addEventListener('click', togglePause);
  el.btnOriginal.addEventListener('click', toggleOriginal);
  el.btnHistory.addEventListener('click', toggleHistory);
  el.btnFontUp.addEventListener('click', function() { updateFontSize(2); });
  el.btnFontDown.addEventListener('click', function() { updateFontSize(-2); });

  document.addEventListener('click', function(e) {
    if (!el.historyModal.contains(e.target) && e.target !== el.btnHistory) {
      el.historyModal.classList.remove('visible');
    }
  });

  document.addEventListener('keydown', function(e) {
    if (e.key === 'p' || e.key === 'P') togglePause();
    if (e.key === 'o' || e.key === 'O') toggleOriginal();
    if (e.key === 'h' || e.key === 'H') toggleHistory();
    if (e.key === '=' || e.key === '+') updateFontSize(2);
    if (e.key === '-') updateFontSize(-2);
  });

  var dragData = { dragging: false, startX: 0, startY: 0, left: 0, bottom: 0 };
  el.container.addEventListener('mousedown', function(e) {
    if (e.target.tagName === 'BUTTON') return;
    var rect = el.container.getBoundingClientRect();
    dragData.dragging = true;
    dragData.startX = e.clientX;
    dragData.startY = e.clientY;
    dragData.left = rect.left;
    dragData.bottom = window.innerHeight - rect.bottom;
    el.container.style.transition = 'none';
  });

  document.addEventListener('mousemove', function(e) {
    if (!dragData.dragging) return;
    var dx = e.clientX - dragData.startX;
    var dy = e.clientY - dragData.startY;
    var newLeft = dragData.left + dx;
    var newBottom = dragData.bottom - dy;
    newLeft = Math.max(0, Math.min(window.innerWidth - 300, newLeft));
    newBottom = Math.max(0, Math.min(window.innerHeight - 60, newBottom));
    el.container.style.left = newLeft + 'px';
    el.container.style.bottom = newBottom + 'px';
    el.container.style.transform = 'none';
  });

  document.addEventListener('mouseup', function() {
    if (dragData.dragging) {
      dragData.dragging = false;
      el.container.style.transition = 'opacity 0.15s';
    }
  });

  statusCheck = setInterval(function() {
    if (Date.now() - lastActivity > 10000) {
      el.statusDot.className = 'status-dot inactive';
      if (!paused) el.statusLabel.textContent = 'Waiting...';
    }
  }, 5000);

  connect();
})();
</script>
</body>
</html>
"""


class TranslationServer:
    def __init__(self, config):
        self.config = config
        self._clients: Set = set()
        self._subtitle_history: list = []
        self._lock = threading.Lock()
        self._http_server = None
        self._ws_server = None
        self._running = False

    async def _ws_handler(self, websocket):
        self._clients.add(websocket)
        client_ip = websocket.remote_address
        logger.info("Client connected: %s (total: %d)", client_ip, len(self._clients))

        try:
            history_snapshot = list(self._subtitle_history)
            for entry in history_snapshot[-5:]:
                await websocket.send(json.dumps(entry))

            await websocket.send(json.dumps({
                "type": "status", "status": "Connected", "text": "Ready"
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.error("WS client error: %s", e)
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected: %s (total: %d)", client_ip, len(self._clients))

    def broadcast(self, data: dict):
        message = json.dumps(data, ensure_ascii=False)
        with self._lock:
            self._subtitle_history.append(data)
            if len(self._subtitle_history) > self.config.max_history * 2:
                self._subtitle_history = self._subtitle_history[-self.config.max_history:]

        dead_clients = set()

        async def _send():
            for client in list(self._clients):
                try:
                    await client.send(message)
                except Exception:
                    dead_clients.add(client)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_send(), loop)
            else:
                asyncio.run(_send())
        except Exception as e:
            logger.error("Broadcast error: %s", e)

        self._clients -= dead_clients

    async def _http_handler(self, scope, receive, send):
        path = scope.get("path", "/")

        if path == "/" or path == "/overlay":
            await send({
                "type": "http.response.start",
                "status": HTTPStatus.OK,
                "headers": [
                    (b"content-type", b"text/html; charset=utf-8"),
                    (b"access-control-allow-origin", b"*"),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": OVERLAY_HTML.encode("utf-8"),
            })
            return

        if path == "/api/status":
            status = {
                "clients": len(self._clients),
                "history_count": len(self._subtitle_history),
                "running": self._running,
            }
            body = json.dumps(status).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": HTTPStatus.OK,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"access-control-allow-origin", b"*"),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        if path == "/api/history":
            entries = self._subtitle_history[-50:]
            body = json.dumps(entries, ensure_ascii=False).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": HTTPStatus.OK,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"access-control-allow-origin", b"*"),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        await send({
            "type": "http.response.start",
            "status": HTTPStatus.NOT_FOUND,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": b'{"error": "Not found"}'})

    def _make_asgi_app(self):
        handler = self._http_handler
        async def asgi_app(scope, receive, send):
            await handler(scope, receive, send)
        return asgi_app

    async def _run(self):
        import uvicorn

        config = uvicorn.Config(
            app=self._make_asgi_app(),
            host=self.config.http_host,
            port=self.config.http_port,
            log_level="warning",
        )
        self._http_server = uvicorn.Server(config)

        ws_handler = self._ws_handler
        self._ws_server = await serve(
            ws_handler,
            host=self.config.ws_host,
            port=self.config.ws_port,
        )

        async with self._ws_server:
            await self._http_server.serve()

    def _server_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        except Exception as e:
            logger.error("Server error: %s", e)
        finally:
            loop.close()

    def start(self):
        self._running = True
        thread = threading.Thread(target=self._server_thread, daemon=True)
        thread.start()
        time.sleep(0.5)
        logger.info(
            "Server started - WS: ws://%s:%d, HTTP: http://%s:%d",
            self.config.ws_host, self.config.ws_port,
            self.config.http_host, self.config.http_port,
        )

    def stop(self):
        self._running = False
        if self._http_server:
            self._http_server.should_exit = True
        if self._ws_server:
            try:
                self._ws_server.close()
            except Exception:
                pass

    def send_translation(self, original: str, translated: str, src_lang: str = "ar"):
        self.broadcast({
            "type": "translation",
            "original": original,
            "translated": translated,
            "src_lang": src_lang,
            "timestamp": time.time(),
        })

    def send_status(self, status: str):
        self.broadcast({
            "type": "status",
            "status": status,
            "text": status,
            "timestamp": time.time(),
        })
