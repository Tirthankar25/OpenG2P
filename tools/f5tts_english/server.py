# tools/f5tts_english/server.py — genuine, unmodified SWivid/F5-TTS
# (English only), running as its OWN process in its OWN venv — NOT part
# of the main app's venv, and deliberately NOT using the tools/<name>/
# MCP convention (see tools/__init__.py) either, because this isn't an
# LLM-callable tool — it's an internal TTS synthesis backend, called
# directly by tts_f5.py the same way the main app calls Ollama: plain
# HTTP, no MCP. Reached from tts_f5.py over localhost (see
# F5TTS_ENGLISH_URL there).
#
# --- WHY this has to be a separate venv, not just installed alongside --------
# --- IndicF5 in the main app's venv -----------------------------------------
# Confirmed directly from ai4bharat/IndicF5's own setup.py (on GitHub, not
# guessed): `pip install git+https://github.com/ai4bharat/IndicF5.git`
# does NOT install IndicF5 as a package depending on the real "f5-tts"
# PyPI project — IndicF5's installable package IS ITSELF published under
# the name `f5_tts`, version "0.1.0", author "svp19", pointing back to
# github.com/svp19/inf5 ("INF5: A Speech Processing Model"). It's a FORK
# of F5-TTS's codebase that happens to keep the identical import name.
#
# Since pip can only have ONE installed distribution per import name,
# installing IndicF5 into a venv that already has the real "f5-tts"
# SILENTLY OVERWRITES it — they can never coexist in the same venv. And
# this fork's own copy of f5_tts/api.py's generic F5TTS class (which
# IndicF5 itself never uses — it only needs f5_tts.infer.utils_infer's
# shared helpers for its own trust_remote_code model) turned out to be
# broken for that case: its own load_model() has checkpoint-loading
# commented out, and its bundled infer/examples/vocab.txt is missing
# from the install entirely. Patching those holes one at a time
# (tried first, see tts_f5.py's git history/comments) works short-term but
# is an unverified bet that this fork's model architecture exactly matches
# the genuine English checkpoint's weights — not guaranteed, since the
# fork may have been adapted for IndicF5's own checkpoint.
#
# So: IndicF5 (Indic languages) stays in the main app's venv, in-process,
# using its own fork exactly as ai4bharat intended. Genuine English
# F5-TTS runs HERE instead, in a completely separate venv with the real
# PyPI "f5-tts" package, as its own small HTTP service. No shared Python
# process, no shared site-packages, no name collision possible.
#
# --- One-time setup (run ON THE SERVER, in a NEW venv — NOT the main ------
# --- app's venv, and NOT activated at the same time as it) ------------------
#   python3 -m venv ~/myenv/.virtualenvs/f5tts_eng
#   source ~/myenv/.virtualenvs/f5tts_eng/bin/activate
#   pip install f5-tts "fastapi" "uvicorn[standard]"
#   # If the default pip install pulls a torch build newer than your
#   # actual NVIDIA driver supports (this app's own main venv's console
#   # log shows exactly that warning — "CUDA initialization: the NVIDIA
#   # driver on your system is too old" — so check this new venv too),
#   # install a matching torch/torchaudio build instead, e.g.:
#   #   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#   # (pick the cuXXX matching your driver — see pytorch.org/get-started/locally)
#   deactivate
#
# This venv shares the same GPU as the main app (GPUs aren't venv-scoped)
# — just another process using the same card, same as running `nvidia-smi`
# alongside anything else. Worth checking VRAM headroom once both are
# loaded (IndicF5 + ASR + IndicTrans2 in the main process, F5-TTS English
# here) if you see CUDA out-of-memory errors.
#
# --- Running -----------------------------------------------------------------
#   F5TTS_ENG_VENV=~/myenv/.virtualenvs/f5tts_eng ./launch.sh
#   # or directly:
#   ~/myenv/.virtualenvs/f5tts_eng/bin/python3 tools/f5tts_english/server.py
#   # Serves plain HTTP (not MCP) at http://<host>:8862 by default — see
#   # F5TTS_ENG_HOST/F5TTS_ENG_PORT below, and F5TTS_ENGLISH_URL in
#   # tts_f5.py, which must point at wherever this actually runs.
#
# --- Sanity check (once running) --------------------------------------------
#   curl http://localhost:8862/health
#   curl -X POST http://localhost:8862/synthesize \
#        -H "Content-Type: application/json" \
#        -d '{"ref_audio_path": "/absolute/path/to/ref.wav", "ref_text": "...", "gen_text": "Hello, this is a test."}' \
#        --output /tmp/test.wav
# -----------------------------------------------------------------------------

import io
import os
import traceback

# --- Compatibility shim for vocos — almost certainly needed here too --------
# The genuine f5-tts package (installed in THIS venv) also depends on
# `vocos`, unpinned — which resolves to vocos's LAST EVER PyPI release,
# 0.1.0 (Oct 2023; confirmed no release since, and the same bug is still
# unfixed on vocos's current GitHub main branch). That release does, in
# vocos/heads.py:
#   from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz
# reaching into a PRIVATE internal submodule from torchaudio's OLD layout,
# which any torchaudio from its "maintenance phase" restructuring onward
# (github.com/pytorch/audio/issues/3902) no longer has — this is exactly
# the bug this app's main venv hit and fixed in tts_f5.py; proactively
# applied here too since it's near-certain to recur with any reasonably
# current torchaudio. Implemented from torchaudio's own last public
# source for these two functions (plain HTK/Slaney mel-scale math,
# unchanged for years) — see tts_f5.py's own copy of this same shim for
# the fuller writeup. No-op (see the try/except below) if your installed
# torchaudio still has this nested path, or brings it back later.
try:
    import importlib as _importlib
    _importlib.import_module("torchaudio.functional.functional")
except ImportError:
    import sys as _sys
    import math as _math
    import types as _types
    import torch as _torch

    def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
        if mel_scale not in ("slaney", "htk"):
            raise ValueError('mel_scale should be one of "htk" or "slaney".')
        if mel_scale == "htk":
            return 2595.0 * _math.log10(1.0 + (freq / 700.0))
        f_min = 0.0
        f_sp = 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = _math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + _math.log(freq / min_log_hz) / logstep
        return mels

    def _mel_to_hz(mels, mel_scale: str = "htk"):
        if mel_scale not in ("slaney", "htk"):
            raise ValueError('mel_scale should be one of "htk" or "slaney".')
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = _math.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * _torch.exp(logstep * (mels[log_t] - min_log_mel))
        return freqs

    _shim = _types.ModuleType("torchaudio.functional.functional")
    _shim._hz_to_mel = _hz_to_mel
    _shim._mel_to_hz = _mel_to_hz
    _sys.modules["torchaudio.functional.functional"] = _shim
    print("[F5-TTS-ENG] patched in a torchaudio.functional.functional "
          "compatibility shim for vocos (see this file's comment above "
          "this block for why this is needed).")
# -----------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

_model = None
_load_error = None


def _load():
    """Loads the genuine F5-TTS English model ONCE at process start —
    same "load once, reuse for every request" shape as every other model
    in this app, just in its own process this time. A load failure is
    recorded, not fatal: /health and /synthesize both report it clearly
    rather than letting the process crash or hang.
    """
    global _model, _load_error
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[F5-TTS-ENG] loading genuine SWivid/F5-TTS (English) onto {device}…")
        from f5_tts.api import F5TTS
        _model = F5TTS(device=device)
        print("[F5-TTS-ENG] ready.")
    except Exception as e:
        _load_error = e
        print(f"[F5-TTS-ENG] warning: could not load F5-TTS ({e}). This "
              f"service will report 'unavailable' to callers until this "
              f"is fixed — see this file's module docstring (pip install "
              f"f5-tts in THIS venv).")
        traceback.print_exc()


_load()


class SynthesizeRequest(BaseModel):
    ref_audio_path: str   # a path on THIS SAME MACHINE's filesystem — the
                           # main app and this service run on the same
                           # host, just different venvs, so passing a
                           # path (not uploading bytes) is fine and avoids
                           # a pointless extra copy.
    ref_text: str
    gen_text: str


@app.get("/health")
def health():
    return {"available": _model is not None,
            "error": str(_load_error) if _load_error else None}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if _model is None:
        return Response(
            content=f"F5-TTS (English) unavailable: {_load_error}".encode("utf-8"),
            status_code=503,
            media_type="text/plain",
        )
    try:
        import soundfile as sf

        print(f"[F5-TTS-ENG] synthesizing ({len(req.gen_text)} chars)…")
        # Exact param names per F5-TTS's own api.py: F5TTS.infer(ref_file,
        # ref_text, gen_text, ...) — everything else left at its
        # documented defaults (nfe_step=32, etc.) rather than guessed at.
        wav, sr, _spec = _model.infer(
            ref_file=req.ref_audio_path,
            ref_text=req.ref_text,
            gen_text=req.gen_text,
            show_info=lambda *a, **k: None,   # suppress its own console spam
        )
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")
    except Exception as e:
        print(f"[F5-TTS-ENG] synthesis failed: {e!r}")
        traceback.print_exc()
        return Response(
            content=f"Inference failure: {e}".encode("utf-8"),
            status_code=500,
            media_type="text/plain",
        )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("F5TTS_ENG_HOST", "0.0.0.0")
    port = int(os.getenv("F5TTS_ENG_PORT", "8862"))
    print(f"[F5-TTS-ENG] starting on {host}:{port}…")
    uvicorn.run(app, host=host, port=port)
