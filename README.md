# live-translate

Real-time speech transcription and translation with a transparent browser subtitle overlay.

## Pipeline

1. **Audio capture** — sounddevice grabs microphone input, and on Linux it can also attach to the PulseAudio system-audio monitor (so you can translate audio playing on your machine)
2. **VAD** — energy threshold plus webrtcvad filter out silence
3. **Transcription** — faster-whisper runs on CPU with `int8` compute; model sizes `tiny`, `base`, `small`, `medium`, `large-v3`
4. **Translation** — MarianMT via Helsinki-NLP/opus-mt models (50+ source languages; default `Helsinki-NLP/opus-mt-ar-en`)
5. **Subtitles** — original + translated text are pushed over WebSocket to a draggable, transparent overlay page in your browser

## Defaults

- WebSocket server: `127.0.0.1:8765`
- Overlay UI: `http://127.0.0.1:8766/overlay`
- Every setting is overridable via `LT_*` environment variables (e.g. `LT_SOURCE_LANGUAGE=fr`, `LT_WS_PORT=9999`) — see `live_translate/config.py`

## Usage

```bash
live-translate                              # defaults: Arabic -> English, tiny model
live-translate --source ar --target en --model base
live-translate --list-devices               # show available audio devices
live-translate --device 4                   # use a specific input device
live-translate --http-port 8080             # custom overlay port
live-translate --no-audio-source            # don't capture system monitor audio
live-translate --translation-model Helsinki-NLP/opus-mt-fr-en
```

Other flags: `--compute` (Whisper compute type, default `int8`), `--beam-size`, `--language` (force Whisper language), `--no-vad`, `--sample-rate`, `--ws-port`, `--http-host`, `--log-level`, `--audio-source` (source name filter, e.g. `translate_sink`).

## Translating system audio on Linux

`run.sh` + `setup-audio.sh` create a `translate_sink` PulseAudio null sink: route the application you want translated to `TranslateSink` in pavucontrol, and the app auto-detects the sink's monitor source.

```bash
./setup-audio.sh   # creates the null sink
./run.sh           # starts live-translate with system audio
```

## Install

```bash
pip install -e .
```

Models (Whisper + MarianMT) download automatically on first run and are cached under `~/.cache/live-translate`.

## License

MIT — see [LICENSE](LICENSE).