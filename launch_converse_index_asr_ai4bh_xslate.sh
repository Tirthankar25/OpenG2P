#!/usr/bin/env bash
# launch_converse_index_asr_ai4bh_xslate.sh — starts the app.
#
# Only ONE process now: ASR + LLM + translation + TTS (Meta MMS-TTS) +
# the web server, all in THIS shell's own venv (torchenv311).
#
# Earlier versions of this script also started a SEPARATE Sooktam2 TTS
# service (tts_service.py) under its own venv (sooktamenv, Python 3.10) —
# that was required because bharatgenai/sooktam2 needed a Python version
# and pinned torch/transformers stack that couldn't share a venv with
# ASR/IndicTrans2 (see tts_service.py's own docstring, still on disk for
# reference, for the full story). MMS-TTS (tts_mms.py) has no such
# requirement — it's native to transformers, no extra pinned package at
# all — so it now loads IN-PROCESS inside the main app, same as ASR/the
# LLM client/the translator. One less process to start, keep alive, and
# debug.
#
# If you ever need the separate-process pattern again (e.g. a future TTS
# model with its own conflicting pinned stack), tts_service.py shows the
# shape to copy: a thin FastAPI wrapper run under its own venv, reached
# over local HTTP from the main app instead of imported directly.

echo "[launch] starting app (torchenv311) on port 8860…"
uvicorn js_asr_demo_server_ai4bh_xslate:app --host 0.0.0.0 --port 8860 \
  2> >(grep -vE "NNPACK|UNEXPECTED" >&2)
