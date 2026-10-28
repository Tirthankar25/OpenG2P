# tts_f5.py — ALTERNATIVE TTS backend (opt-in via TTS_BACKEND, see
# js_asr_demo_server_ai4bh_xslate.py): ai4bharat/IndicF5 for this app's
# Indic languages, SWivid/F5-TTS for English. NOT the default — tts_mms.py
# (Meta MMS-TTS) stays the default, because of the tradeoffs documented
# below; this exists as a drop-in alternative for when F5's voice quality
# is worth those tradeoffs for a particular deployment.
#
# Researched against the actual published packages/model cards before
# writing this (not reconstructed from general TTS knowledge, same
# discipline as every other model integration in this app) — see the
# specific facts cited below. What was NOT possible to verify from this
# sandbox: actually running either model (both are real, GPU-sized
# checkpoints — hundreds of MB to low GB — not something to download and
# exercise here). This file's CONTRACT (the {"success", "sampling_rate",
# "mime_type", "audio_bytes"} shape, matching tts_mms.MmsTTS.synthesize()
# exactly) and the backend-selection wiring are what's been verified by
# inspection/compile; the actual inference calls are built directly from
# each project's own documented example code, but have not been run
# end-to-end against real weights. Test with the sanity check below
# before relying on this in the running app.
#
# --- Tradeoff 1: slower per-utterance than MMS-TTS, confirmed not assumed --
# Both IndicF5 and F5-TTS are flow-matching / diffusion-based — NON-
# autoregressive (not the old token-by-token Sooktam2/Indic-Parler-TTS
# category this app moved away from), but they still run ~32 iterative
# denoising steps (NFE) per utterance to produce the waveform, versus
# MMS-TTS's single VITS forward pass. Confirmed from F5-TTS's own
# published default (nfe_step=32), not guessed. Expect noticeably higher
# per-utterance latency than tts_mms.py — and since each call now holds a
# GPU_CONCURRENCY slot (see js_asr_demo_server_ai4bh_xslate.py's "GPU-call
# concurrency cap") for longer, it's worth re-checking that cap's sizing
# if you switch to this backend under real concurrent load.
#
# --- Tradeoff 2: reference audio is required — but ONE clip covers EVERY --
# --- language, thanks to cross-lingual cloning ------------------------------
# Unlike MMS-TTS's fixed, built-in per-language voices, BOTH IndicF5 and
# F5-TTS are voice-CLONING models: every call needs a short REFERENCE
# AUDIO CLIP plus its EXACT matching transcript (the voice the output
# imitates). This is the SAME paradigm as the old Sooktam2/admin-
# enrollment setup this app removed — but NOT the same burden: IndicF5
# does CROSS-LINGUAL cloning, confirmed directly from AI4Bharat's own
# published model-card example, which generates HINDI speech from a
# PUNJABI reference clip (ref_text is Punjabi, the generated text is
# Hindi) — the reference clip's language is independent of whatever
# language you ask it to generate. So ONE reference clip carries a
# speaker's voice characteristics across all 11 of IndicF5's supported
# languages; you do NOT need a separate clip per language. An ENGLISH
# reference clip is the natural choice, since the SAME clip then also
# works NATIVELY (not just cross-lingually) for the separate F5-TTS
# English path below — both model APIs just take a plain ref_audio_path
# + ref_text, with no language tag of their own, so nothing stops
# passing the identical files to both.
#
# This file does NOT bring enrollment UI back — per this being a
# code-level/operator setting, not a user-facing one, the reference clip
# is simply a static file pair placed on disk ONCE, used for every
# language unless you deliberately override it for one:
#
#   tts_f5_voices/default/ref.wav   — a short (a few seconds), clean
#                                      recording of the voice to clone —
#                                      used for EVERY language unless
#                                      overridden below. English is the
#                                      natural choice (see above).
#   tts_f5_voices/default/ref.txt   — the EXACT transcript of that clip's
#                                      speech, plain UTF-8 text, nothing else
#
#   tts_f5_voices/<lang>/ref.wav    — OPTIONAL per-language override, if
#   tts_f5_voices/<lang>/ref.txt      you ever want a different voice (or
#                                      a same-language reference, which
#                                      may sound marginally more natural
#                                      than cross-lingual for that one
#                                      language even though it isn't
#                                      required) for a SPECIFIC language —
#                                      <lang> is one of this app's own
#                                      codes (hin, eng, ben, kan, tam,
#                                      ory, guj, mar — see
#                                      SUPPORTED_LANGUAGES below). Checked
#                                      FIRST, before falling back to
#                                      default/ above.
#
# Missing BOTH the override and the default reports a clear, actionable
# error at SYNTHESIZE time (not at startup) — mirrors tts_mms.py's
# per-language load-error handling. TTS_F5_VOICES_DIR overrides the
# default location (this file's own directory / tts_f5_voices/).
#
# --- Prereqs (on the backend host, in THIS app's own venv) -----------------
#   pip install "git+https://github.com/ai4bharat/IndicF5.git"
#
#   # IMPORTANT, confirmed from IndicF5's own requirements.txt — it pins
#   # transformers<4.50. This app's venv ALSO runs transformers for ASR
#   # (omnilingual_asr_ai4bh_xslate.py), IndicTrans2 (indic_translator.py —
#   # which already carries its OWN transformers-version compat shim, for
#   # a DIFFERENT symptom — a re-export REMOVED in a newer transformers
#   # release), and tts_mms.py's own VitsModel. Installing IndicF5 may
#   # downgrade transformers venv-wide. TEST IN AN ISOLATED VENV FIRST
#   # rather than installing straight into the running app's venv.
#
#   # Do NOT also `pip install f5-tts` in THIS venv (see "Architecture"
#   # below) — IndicF5's own installer IS published under the identical
#   # import name `f5_tts`, so whichever you install second silently
#   # overwrites the other. English F5-TTS runs as its own separate
#   # service instead — see tools/f5tts_english/server.py.
#
# --- Architecture: why English F5-TTS is a SEPARATE service, not in-process -
# Confirmed directly from ai4bharat/IndicF5's own setup.py (GitHub, not
# guessed): `pip install git+https://github.com/ai4bharat/IndicF5.git`
# installs ITSELF as a package literally named `f5_tts`, version "0.1.0",
# author "svp19" (github.com/svp19/inf5 — "INF5: A Speech Processing
# Model") — a FORK of F5-TTS's codebase, not a separate dependency of it.
# Since pip can only have one installed distribution per import name,
# that silently overwrites whatever real "f5-tts" (PyPI, SWivid/F5-TTS)
# was installed before it in the SAME venv. Confirmed from the actual
# installed files: this fork's own api.py/load_model() (which IndicF5
# itself never calls — it only needs f5_tts.infer.utils_infer's shared
# helpers for its own trust_remote_code model) has checkpoint-loading
# commented out and is missing its bundled vocab.txt — broken for the
# generic English use case, and not safe to assume its model classes are
# architecturally identical to the genuine English checkpoint's weights
# even once patched.
#
# So: IndicF5 (Indic languages) stays here, in-process, using its own
# fork exactly as ai4bharat intended (that part works correctly — see
# below). Genuine English F5-TTS runs in tools/f5tts_english/server.py
# instead, in a completely separate venv/process with the real PyPI
# f5-tts package, reached over plain localhost HTTP (F5TTS_ENGLISH_URL
# below) — same pattern this app already uses for Ollama (llama_llm.py),
# just without the MCP layer tools/GAS and tools/policies use, since this
# isn't an LLM-callable tool. See tools/f5tts_english/server.py's module
# docstring for one-time setup (its own venv, `pip install f5-tts`) and
# tools/f5tts_english/launch.sh for running it alongside this app.
#
# --- Sanity check (standalone, no main server needed; needs real weights) --
#   # 1. Start the English service first (see tools/f5tts_english/launch.sh)
#   # 2. Then, in THIS app's own venv:
#   python3 -c "
#   from tts_f5 import F5VoiceTTS
#   tts = F5VoiceTTS()
#   r = tts.synthesize('Hello, this is a test.', 'eng')
#   print(r['success'], r.get('error'))
#   open('/tmp/test.wav','wb').write(r['audio_bytes']) if r['success'] else None
#   "
# -----------------------------------------------------------------------------

import io
import os
import traceback

import requests

# --- Root cause, confirmed: evict the ASR-startup torchaudio STUB before ----
# --- this backend touches torchaudio at all ---------------------------------
# js_asr_demo_server_ai4bh_xslate.py installs a DELIBERATE placeholder
# torchaudio module into sys.modules right at the top of that file ("Stub
# torchaudio BEFORE anything else imports it"), before OmnilingualASR /
# transformers even get imported. It's an empty types.ModuleType carrying
# ModuleSpec(loader=None, origin="stub") and empty sub-stub modules with no
# real attributes and no __path__ — confirmed exactly from a diagnostic
# placed here earlier: dir(torchaudio.transforms) == [], no import warnings
# at all (because it's never a real import), __file__ missing.
#
# ASR doesn't actually need torchaudio itself — omnilingual_asr_ai4bh_xslate.py
# loads audio via librosa — so that stub exists purely to stop some OTHER
# library transformers/ASR pulls in from eagerly importing the REAL
# torchaudio during that startup phase. It stays in sys.modules for the rest
# of the process, so by the time this file runs, `import torchaudio` was
# just returning that same empty shell — which ALSO explains the very first
# error this file hit (vocos's "torchaudio.functional.functional ... is not
# a package"): the stub's one-level-deep "functional" sub-stub has no
# __path__ at all, so it looks like "not a package" regardless of what the
# REAL installed torchaudio's layout actually is. That was never really
# about torchaudio 2.11's layout — it was this stub the whole time.
#
# This backend genuinely needs the real torchaudio (specifically
# torchaudio.transforms.MelSpectrogram, via vocos) — and your own standalone
# check confirmed the real installed package is completely healthy in
# isolation. Nothing holds a live reference to this exact stub module object
# across a later re-import (ASR's own imports already ran and resolved
# against it), so it's safe to evict it here and force a genuine import
# before IndicF5/F5-TTS/vocos load. Watch the console after this change —
# if ASR's own startup (earlier in the log, before "Loading TTS backend:
# f5…") still shows success, this swap didn't disturb it.
import sys as _sysmod
_ta_mod = _sysmod.modules.get("torchaudio")
if _ta_mod is not None and getattr(getattr(_ta_mod, "__spec__", None), "origin", None) == "stub":
    print("[F5-TTS] detected the ASR-startup torchaudio stub in sys.modules "
          "— evicting it and re-importing the real torchaudio for "
          "IndicF5/F5-TTS/vocos to use.")
    for _name in list(_sysmod.modules):
        if _name == "torchaudio" or _name.startswith("torchaudio."):
            del _sysmod.modules[_name]
    import torchaudio as _real_torchaudio
    print(f"[F5-TTS] real torchaudio imported: {_real_torchaudio.__version__} "
          f"({getattr(_real_torchaudio, '__file__', '?')}); "
          f"MelSpectrogram present = "
          f"{hasattr(_real_torchaudio.transforms, 'MelSpectrogram')}")
# -----------------------------------------------------------------------------

# --- Compatibility shim for vocos (a dependency of BOTH f5_tts AND -----------
# --- IndicF5's own trust_remote_code model.py) -------------------------------
# vocos 0.1.0 — its LAST release, October 2023 (checked on PyPI: no release
# since) — does, in vocos/heads.py:
#   from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz
# reaching into a PRIVATE internal submodule from torchaudio's OLD layout
# (functional/ as a package with its real implementation in an internal
# functional/functional.py file). Confirmed from this app's OWN traceback,
# hit identically while loading BOTH IndicF5 and F5-TTS (IndicF5's remote
# model.py itself imports f5_tts.infer.utils_infer -> vocos, same path):
#   ModuleNotFoundError: No module named 'torchaudio.functional.functional';
#   'torchaudio.functional' is not a package
# Current torchaudio (this app runs 2.11.0 against torch 2.14.0) flattened
# that nested layout away as part of its own "maintenance phase"
# restructuring (github.com/pytorch/audio/issues/3902 — torchaudio no
# longer bumps its version to track torch's, so a torch/torchaudio
# mismatch is NOT the cause here, that was checked and ruled out first).
# Checked vocos's current GitHub main branch too, not just the stale PyPI
# release — the broken import is identical there, so this is an unfixed
# upstream bug, not a "you have an old copy" problem.
#
# Rather than depend on exactly where (or whether) torchaudio's own
# internals happen to keep equivalent functions across ITS future releases
# too — the same fragility that caused this break — this installs a small,
# self-contained replacement module at
# sys.modules["torchaudio.functional.functional"], implemented directly
# from torchaudio's own last public source for these two functions
# (plain HTK/Slaney mel-scale conversion math, unchanged for years — see
# github.com/pytorch/audio/blob/v2.1.0/torchaudio/functional/functional.py),
# so vocos's import succeeds no matter what torchaudio's internal layout
# is, now or later. This never touches torchaudio itself — it only ever
# supplies a module that otherwise wouldn't exist, so it's a no-op (see
# the try/except below) if a future torchaudio release brings this nested
# path back on its own. Mirrors indic_translator.py's own
# PreTrainedTokenizerBase re-export shim — same technique, same reason.
#
# MUST run before f5_tts/vocos get imported anywhere below, which is why
# it's up here, right after the stdlib imports.
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
    print("[F5-TTS] patched in a torchaudio.functional.functional "
          "compatibility shim for vocos (see tts_f5.py's comment above "
          "this block for why this is needed).")
# -----------------------------------------------------------------------------

from tts_text_normalize import normalize_for_tts

SUPPORTED_LANGUAGES = [
    "hin", "eng", "ben", "kan", "tam", "ory", "guj", "mar",
]

# IndicF5 handles ALL of these Indic languages through ONE shared model
# (unlike MMS-TTS's one-checkpoint-per-language design) — routed by the
# input text's own script/content, not a per-language model swap.
# English is NOT one of IndicF5's supported languages (it's an
# Indic-languages-specific model, 11 languages per its model card: Assamese,
# Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi,
# Tamil, Telugu — this app only uses the 7 of those it already has codes
# for) — English goes through plain F5-TTS instead, a separate model/package.
_INDIC_LANGUAGES = {"hin", "ben", "kan", "tam", "ory", "guj", "mar"}

_VOICES_DIR = os.getenv(
    "TTS_F5_VOICES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_f5_voices"),
)

# Confirmed from both projects' own docs/config (target_sample_rate /
# samplerate in their example code) — not assumed.
SAMPLE_RATE = 24000

# Where the SEPARATE genuine-English-F5-TTS service (tools/f5tts_english/
# server.py) is reachable — see this module's "Architecture" note above
# for why English runs there instead of in-process. Same
# localhost-HTTP-service pattern as OLLAMA_URL in llama_llm.py.
F5TTS_ENGLISH_URL = os.getenv("F5TTS_ENGLISH_URL", "http://localhost:8862")
F5TTS_ENGLISH_TIMEOUT_S = float(os.getenv("F5TTS_ENGLISH_TIMEOUT_S", "120"))


def _load_reference(lang: str):
    """Returns (ref_audio_path, ref_text) to use for `lang`, or raises
    with a clear, actionable message if nothing usable is on disk.

    Checks tts_f5_voices/<lang>/ first (an OPTIONAL per-language
    override), then falls back to the single SHARED
    tts_f5_voices/default/ clip — which, thanks to IndicF5's cross-
    lingual cloning (see module docstring), is the normal setup: ONE
    reference clip covers every language, not one per language.

    Called fresh on every synthesize() (cheap: a couple of path checks +
    a small text-file read) rather than cached at load time, so a
    reference clip can be added or swapped without restarting the app.
    """
    for lang_dir in (os.path.join(_VOICES_DIR, lang), os.path.join(_VOICES_DIR, "default")):
        wav_path = os.path.join(lang_dir, "ref.wav")
        txt_path = os.path.join(lang_dir, "ref.txt")
        if os.path.isfile(wav_path) and os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
            if not ref_text:
                raise ValueError(f"{txt_path} is empty — it must contain the exact transcript of {wav_path}.")
            return wav_path, ref_text

    raise FileNotFoundError(
        f"No reference voice available for '{lang}' — need BOTH ref.wav "
        f"and ref.txt, either under {os.path.join(_VOICES_DIR, 'default')} "
        f"(the single shared clip used for every language — see "
        f"tts_f5.py's module docstring on cross-lingual cloning; an "
        f"English clip is the natural choice) or under "
        f"{os.path.join(_VOICES_DIR, lang)} (an optional override for "
        f"THIS language specifically). Neither pair exists yet."
    )


class F5VoiceTTS:
    """Drop-in alternative to tts_mms.MmsTTS — SAME synthesize() contract
    ({"success", "sampling_rate", "mime_type", "audio_bytes"} on success,
    {"success": False, "error"} on failure), so switching backends in
    js_asr_demo_server_ai4bh_xslate.py is the one-line change documented
    there (see TTS_BACKEND). Internally routes Indic languages to a
    single shared IndicF5 model instance, English to a separate F5-TTS
    instance — see module docstring for why these are two different
    models/packages, and for the reference-audio requirement both share.
    """

    def __init__(self, device: str = None):
        self.device = device
        self._indic_model = None
        self._indic_load_error = None
        # English F5-TTS is NOT loaded in-process (see module docstring's
        # "Architecture" note) — it's a separate HTTP service. There's
        # nothing to load here; availability/errors are discovered at
        # synthesize() time, same "not a strict dependency, fails clean
        # and recovers next call" spirit as tools/policies/client.py.
        self._f5_english_last_error = None
        self._load()

    def _load(self):
        try:
            import torch
        except Exception as e:
            self._indic_load_error = e
            print(f"[F5-TTS] warning: could not import torch ({e}). Indic "
                  f"'speak reply' (IndicF5) will be unavailable until this is "
                  f"fixed. (English F5-TTS doesn't need torch HERE — it runs "
                  f"in its own separate process; see F5TTS_ENGLISH_URL.)")
            traceback.print_exc()
            return

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # IndicF5 — loaded as a `trust_remote_code` HF model, per its own
        # model card's example (AutoModel.from_pretrained(repo_id,
        # trust_remote_code=True); no device kwarg shown there, so this
        # relies on the standard nn.Module .to(device) interface, which
        # custom trust_remote_code models normally still support — NOT
        # independently confirmed against real weights from this sandbox;
        # check nvidia-smi during your first real synthesis to confirm it
        # actually lands on GPU rather than silently running on CPU.
        try:
            print("[F5-TTS] loading ai4bharat/IndicF5 (Indic languages)…")
            from transformers import AutoModel
            self._indic_model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
            self._indic_model = self._indic_model.to(self.device)
            print("[F5-TTS] IndicF5 ready.")
        except Exception as e:
            self._indic_load_error = e
            print(f"[F5-TTS] warning: could not load IndicF5 ({e}). Indic-language "
                  f"'speak reply' will be unavailable until this is fixed — see "
                  f"tts_f5.py's module docstring (pip install, transformers<4.50).")
            # Just {e} only ever gives the message, with no file/line — not
            # enough to tell "my own code called this wrong" apart from "a
            # dependency's internal import is broken against my installed
            # torch/torchaudio/transformers versions" (exactly the
            # ambiguity that came up with the F5-TTS load error below).
            traceback.print_exc()

        # English F5-TTS — a separate HTTP service, not loaded in-process
        # (see module docstring's "Architecture" note for why). This is a
        # one-time, NON-fatal readiness probe at startup, purely for a
        # clearer console banner — it does NOT gate anything: the service
        # may still be starting up (loading its own GPU weights takes a
        # while) when this app starts, and may come up AFTER this point,
        # or restart independently later. Every real call goes through
        # synthesize() below, which re-checks and reports failures fresh
        # each time regardless of what this probe found.
        try:
            r = requests.get(f"{F5TTS_ENGLISH_URL}/health", timeout=3)
            r.raise_for_status()
            info = r.json()
            if info.get("available"):
                print(f"[F5-TTS] English F5-TTS service is up at {F5TTS_ENGLISH_URL}.")
            else:
                print(f"[F5-TTS] English F5-TTS service is reachable at "
                      f"{F5TTS_ENGLISH_URL} but reports it failed to load "
                      f"({info.get('error')}) — check that service's own "
                      f"console/logs (tools/f5tts_english/server.py).")
        except requests.exceptions.RequestException as e:
            print(f"[F5-TTS] note: English F5-TTS service not reachable yet "
                  f"at {F5TTS_ENGLISH_URL} ({e}) — English 'speak reply' will "
                  f"report a clear error until it's started (see "
                  f"tools/f5tts_english/launch.sh). This is NOT fatal and "
                  f"isn't re-checked until the next actual synthesize call.")

    def available(self) -> bool:
        """True if IndicF5 loaded OR the English service is configured —
        mirrors tts_mms.MmsTTS.available()'s convention, but "available"
        for English is necessarily optimistic here (it's a separate
        process whose real state can change at any time) — see
        synthesize() for the call that actually confirms it works."""
        return self._indic_model is not None or bool(F5TTS_ENGLISH_URL)

    def synthesize(self, text: str, language: str) -> dict:
        """Same contract as tts_mms.MmsTTS.synthesize() — see that
        method's docstring. Routes to IndicF5 or F5-TTS based on
        `language`; falls back to 'eng' for an unrecognized code, same
        fallback spirit as MmsTTS."""
        text = (text or "").strip()
        if not text:
            return {"success": False, "error": "No text to speak."}

        lang = (language or "").strip().lower()
        if lang not in SUPPORTED_LANGUAGES:
            lang = "eng"

        try:
            ref_path, ref_text = _load_reference(lang)
        except Exception as e:
            return {"success": False, "error": str(e)}

        # Same pre-synthesis cleanup MMS-TTS gets (°, ranges, %, ₹, …) —
        # see tts_text_normalize.py. Applies regardless of backend.
        text = normalize_for_tts(text, lang)

        if lang in _INDIC_LANGUAGES:
            try:
                import numpy as np

                if self._indic_model is None:
                    return {"success": False,
                            "error": f"IndicF5 unavailable ({self._indic_load_error})"}
                print(f"[F5-TTS] synthesizing via IndicF5 ({lang}, {len(text)} chars)…")
                # Exact call shape per ai4bharat/IndicF5's own model-card
                # example: model(text, ref_audio_path=..., ref_text=...).
                audio = self._indic_model(text, ref_audio_path=ref_path, ref_text=ref_text)
                audio = np.array(audio)
                if audio.dtype == np.int16:
                    # Matches IndicF5's own example code's int16->float32
                    # normalization step verbatim.
                    audio = audio.astype(np.float32) / 32768.0
                waveform = audio.astype(np.float32)
                sr = SAMPLE_RATE

                import soundfile as sf
                out_buffer = io.BytesIO()
                sf.write(out_buffer, waveform, sr, format="WAV")
                return {
                    "success": True,
                    "sampling_rate": sr,
                    "mime_type": "audio/wav",
                    "audio_bytes": out_buffer.getvalue(),
                }
            except Exception as e:
                print(f"[F5-TTS] synthesis failed: {e!r}")
                traceback.print_exc()
                return {"success": False, "error": f"Inference failure: {e}"}

        # English — the separate genuine-F5-TTS service (see module
        # docstring's "Architecture" note). A plain HTTP call, same shape
        # as llama_llm.py's Ollama calls: never assume it's up (it's an
        # independent process that can be down, still loading, or
        # restarting), always report a clear, actionable error instead of
        # raising.
        print(f"[F5-TTS] synthesizing via English F5-TTS service "
              f"({F5TTS_ENGLISH_URL}, {len(text)} chars)…")
        try:
            resp = requests.post(
                f"{F5TTS_ENGLISH_URL}/synthesize",
                json={"ref_audio_path": ref_path, "ref_text": ref_text, "gen_text": text},
                timeout=F5TTS_ENGLISH_TIMEOUT_S,
            )
        except requests.exceptions.RequestException as e:
            self._f5_english_last_error = e
            print(f"[F5-TTS] could not reach English F5-TTS service at "
                  f"{F5TTS_ENGLISH_URL}: {e!r}")
            return {"success": False,
                    "error": f"English F5-TTS service unavailable at "
                             f"{F5TTS_ENGLISH_URL} ({e}) — is it running? "
                             f"See tools/f5tts_english/launch.sh."}

        if resp.status_code != 200:
            self._f5_english_last_error = resp.text
            print(f"[F5-TTS] English F5-TTS service returned "
                  f"{resp.status_code}: {resp.text[:500]!r}")
            return {"success": False,
                    "error": f"English F5-TTS service error "
                             f"({resp.status_code}): {resp.text[:300]}"}

        try:
            import soundfile as sf
            sr = sf.info(io.BytesIO(resp.content)).samplerate
        except Exception:
            # The response IS valid audio either way (the service only
            # ever returns 200 with a real WAV body or a non-200 error, so
            # this only fails if soundfile itself can't be imported) —
            # fall back to the documented default rather than failing the
            # whole call over a cosmetic metadata read.
            sr = SAMPLE_RATE

        self._f5_english_last_error = None
        return {
            "success": True,
            "sampling_rate": sr,
            "mime_type": "audio/wav",
            "audio_bytes": resp.content,
        }
