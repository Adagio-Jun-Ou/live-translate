#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/live-translate-venv"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -e "$SCRIPT_DIR"
else
    source "$VENV/bin/activate"
fi

echo "=== Live Translate Setup ==="
echo ""

echo "1. Setting up audio routing..."
SINK_NAME="translate_sink"
if ! pactl list short sinks 2>/dev/null | grep -q "$SINK_NAME"; then
    pactl load-module module-null-sink \
        sink_name="$SINK_NAME" \
        sink_properties=device.description=TranslateSink 2>/dev/null || true
    echo "   Created null sink: $SINK_NAME"
else
    echo "   Null sink already exists: $SINK_NAME"
fi

echo "   Recording from: $(pactl list short sources 2>/dev/null | grep -i monitor | head -1 | awk '{print $2}')"
echo ""

echo "2. Opening browser overlay..."
OVERLAY_URL="http://127.0.0.1:8766/overlay"
xdg-open "$OVERLAY_URL" 2>/dev/null || echo "   Please open: $OVERLAY_URL"
echo ""

echo "3. Starting Live Translate engine..."
echo "   Source language: ${LT_SOURCE_LANGUAGE:-ar} → ${LT_TARGET_LANGUAGE:-en}"
echo "   Whisper model:   ${LT_WHISPER_MODEL:-tiny}"
echo "   Press Ctrl+C to stop"
echo ""

exec python3 -m live_translate.app "${@}"
