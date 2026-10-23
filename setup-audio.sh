#!/bin/bash
set -euo pipefail

SINK_NAME="translate_sink"
AUDIO_SRC="${1:-}"

echo "=== Audio Routing Setup ==="

if ! pactl list short sinks 2>/dev/null | grep -q "$SINK_NAME"; then
    pactl load-module module-null-sink \
        sink_name="$SINK_NAME" \
        sink_properties=device.description=TranslateSink 2>/dev/null
    echo "Created null sink: $SINK_NAME"
else
    echo "Null sink exists: $SINK_NAME"
fi

echo ""
echo "Available monitor sources:"
pactl list short sources 2>/dev/null | grep monitor | while read line; do
    echo "  $line"
done

echo ""
echo "To route audio to the translation sink:"
echo "  1. Open pavucontrol (GUI) or use pactl"
echo "  2. In the 'Playback' tab, redirect the application to 'TranslateSink'"
echo "  3. In the 'Recording' tab, connect Live Translate to the monitor of your output"
echo ""
echo "The Live Translate app will auto-detect the Translate Sink monitor."
