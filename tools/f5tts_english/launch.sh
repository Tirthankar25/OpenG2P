#!/usr/bin/env bash
# tools/f5tts_english/launch.sh — starts the genuine English F5-TTS
# service in ITS OWN venv — see server.py's module docstring for why this
# has to be a separate venv from the main app's (ai4bharat/IndicF5's own
# installer overwrites the real "f5-tts" package, same import name).
#
# Set F5TTS_ENG_VENV to point at that venv (create it ONCE per
# server.py's "One-time setup" section) — defaults to
# ~/myenv/.virtualenvs/f5tts_eng, matching this app's existing
# ~/myenv/.virtualenvs/torchenv311 naming convention.

F5TTS_ENG_VENV="${F5TTS_ENG_VENV:-$HOME/myenv/.virtualenvs/f5tts_eng}"

if [ ! -x "$F5TTS_ENG_VENV/bin/python3" ]; then
    echo "[f5tts_english] ERROR: no venv found at $F5TTS_ENG_VENV" >&2
    echo "[f5tts_english] see tools/f5tts_english/server.py's module docstring for one-time setup, or set F5TTS_ENG_VENV to point at an existing one." >&2
    exit 1
fi

echo "[f5tts_english] starting genuine F5-TTS (English) service on port ${F5TTS_ENG_PORT:-8862} (venv: $F5TTS_ENG_VENV)…"
"$F5TTS_ENG_VENV/bin/python3" "$(dirname "$0")/server.py"
