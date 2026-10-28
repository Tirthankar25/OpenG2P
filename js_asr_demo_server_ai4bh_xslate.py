# js_asr_demo_server.py — FastAPI backend
#
# Drop-in replacement: same VAD + OmnilingualASR logic, REST + WebSocket API
# + on-demand Llama 3/3.1 response generation via a local Ollama server.
# Run: uvicorn js_asr_demo_server_ai4bh_xslate:app --host 0.0.0.0 --port 8860

# using seamless finetuned on Indic - ai4bharat - https://huggingface.co/ai4bharat/indic-seamless

import sys
import types
import importlib.machinery

# ---------------------------------------------------------------------------
# Stub torchaudio BEFORE anything else imports it.
# ---------------------------------------------------------------------------
def _stub_torchaudio():
    ta = types.ModuleType("torchaudio")
    ta.__version__ = "2.9.1"
    ta.__path__    = []
    ta.__package__ = "torchaudio"
    ta.__loader__  = None
    ta.__spec__    = importlib.machinery.ModuleSpec(
        name="torchaudio", loader=None, origin="stub"
    )
    ta.__spec__.submodule_search_locations = []
    for sub in ["transforms", "_extension", "functional", "backend",
                "sox_effects", "compliance", "compliance.kaldi",
                "io", "utils", "pipelines", "models"]:
        mod = types.ModuleType(f"torchaudio.{sub}")
        mod.__spec__ = importlib.machinery.ModuleSpec(
            f"torchaudio.{sub}", None, origin="stub"
        )
        ta.__dict__[sub.split(".")[0]] = mod
        sys.modules[f"torchaudio.{sub}"] = mod
    ta._extension._IS_TORCHAUDIO_EXT_AVAILABLE = False
    sys.modules["torchaudio"] = ta

_stub_torchaudio()
# ---------------------------------------------------------------------------

import uuid
import time
import threading
import queue
import tempfile
import os
import io
import re
import traceback
import base64
import warnings
import logging
from typing import Optional

import numpy as np
import torch
import requests
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from math import gcd

warnings.filterwarnings("ignore")
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.responses import FileResponse

from omnilingual_asr_ai4bh_xslate import OmnilingualASR
from llama_llm import LlamaResponder, status_text_for
from indic_translator import IndicTranslator

# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------
print("Loading ASR model…")
asr = OmnilingualASR()
print("ASR ready.")

print("Connecting to Ollama (Llama)…")
llm = LlamaResponder()
print("LLM ready.")

print("Loading IndicTrans2…")
translator = IndicTranslator()
print("Translator ready.")

# num2words is pure Python (no GPU, no torch dependency) — used only to turn
# digits into spoken-word form before handing text to TTS (see
# _normalize_tts_text() below). Optional: if it's missing, normalization is
# skipped entirely (numbers go to TTS exactly as before this change) rather
# than crashing app startup over a small text-quality feature.
try:
    from num2words import num2words
    _NUM2WORDS_AVAILABLE = True
except ImportError:
    _NUM2WORDS_AVAILABLE = False
    print("[TTS] warning: num2words not installed — numbers/percentages will "
          "NOT be normalized to words before TTS (run `pip install num2words` "
          "in this app's own venv, then restart, to enable it).")

# ---------------------------------------------------------------------------
# TTS — backend selected HERE, in code (TTS_BACKEND below) — deliberately
# NOT a user-facing setting: no frontend control, no per-request API
# param, nothing a browser client can influence. An operator changes
# which backend runs by editing this default (or setting the
# TTS_BACKEND env var) and restarting — same pattern as every other
# server-side-only knob in this file (GPU_CONCURRENCY, OLLAMA_MODEL, …).
#
#   "mms" (default) — tts_mms.py, Meta MMS-TTS. One small, fast,
#     non-autoregressive VITS model per language, fixed built-in voices,
#     no reference clip, no admin enrollment. Replaced tts_parler.py
#     (AI4Bharat Indic Parler-TTS — worked, but autoregressive generation
#     was reported slow in practice, the Play button hanging rather than
#     erroring); MMS-TTS's non-autoregressive architecture is a different
#     speed class entirely.
#
#   "f5" — tts_f5.py, ai4bharat/IndicF5 (Indic languages) + SWivid/F5-TTS
#     (English). Different voice-quality/latency/setup tradeoff — see
#     tts_f5.py's module docstring before switching: it's a flow-matching
#     model (slower per utterance than MMS-TTS, though still far faster
#     than the old autoregressive Sooktam2/Parler path) AND, unlike
#     MMS-TTS, it's a voice-CLONING model — it needs a reference audio
#     clip + transcript per language on disk before it can speak at all
#     (tts_f5_voices/<lang>/ref.wav + ref.txt). Not a drop-in you can
#     just flip without that setup done first.
#
# tts_parler.py and tts_sooktam.py are both left on disk, unreferenced,
# for the same "might be worth revisiting" reason as always — neither is
# wired into TTS_BACKEND below; add a branch here if one of them ever is.
TTS_BACKEND = os.getenv("TTS_BACKEND", "mms")   # "mms" or "f5"

print(f"Loading TTS backend: {TTS_BACKEND}…")
if TTS_BACKEND == "f5":
    from tts_f5 import F5VoiceTTS
    tts = F5VoiceTTS()
else:
    from tts_mms import MmsTTS
    tts = MmsTTS()
print("TTS ready." if tts.available() else
      "TTS unavailable — check the error printed above (load failed).")

# ---------------------------------------------------------------------------
# GPU-call concurrency cap
# ---------------------------------------------------------------------------
# asr/translator/tts above are each a SINGLE model instance, loaded once at
# startup and shared by every session (make_session() gives each browser
# tab its own conversation STATE, not its own model) — see the design
# discussion that led here. Each session already serializes its OWN calls
# (one utterance at a time via _utterance_worker()'s queue, one chat turn
# at a time via _chat_turn_worker()'s queue), but nothing previously
# stopped DIFFERENT sessions' worker threads from calling into the shared
# asr/translator/tts objects at the exact same moment — harmless for
# correctness (none of the three keep mutable state on `self` mid-call,
# confirmed by reading transcribe()/translate()/synthesize() themselves),
# but a real risk for a single shared GPU's VRAM if enough sessions'
# GPU-bound calls land together: every concurrent forward pass holds its
# own CUDA memory until it returns, and that can add up to an OOM under
# real simultaneous load, especially stacked with the Ollama process also
# sharing the same card.
#
# This semaphore is a CAP, not a queue manager or a scheduler — it just
# refuses to let more than GPU_CONCURRENCY of these calls be in flight at
# once, system-wide, blocking anything beyond that until a slot frees.
# Sized to the assumed worst case (5 concurrent users) rather than some
# larger number precisely so that AT that worst case, acquiring a slot is
# a no-op (a semaphore with a free slot returns immediately — there is no
# polling, no sleep, nothing that could add latency) and the cap only
# ever actually blocks anything if load somehow exceeds the assumption
# this was sized for. In other words: this cannot make a normal request
# slower than it already was, only prevent an ABNORMAL pile-up from
# taking the GPU down with it.
#
# One shared semaphore across ASR + translate + TTS (not three separate
# ones) — it bounds the TOTAL number of concurrent GPU-bound inference
# calls across the whole app to GPU_CONCURRENCY, which is what "5
# concurrent users" actually means for shared-GPU capacity planning, not
# "5 concurrent ASR calls AND SEPARATELY ALSO 5 concurrent TTS calls"
# (that would let as many as 2x GPU_CONCURRENCY calls stack up at once,
# since a session's ASR worker and chat-turn worker are two independent
# threads that CAN legitimately overlap — this session's own next
# utterance being transcribed while ITS OWN previous reply is still being
# spoken — so bounding the two stages separately would under-protect
# against exactly the pile-up this exists to prevent).
#
# Known, accepted edge case: because of that same ASR/TTS overlap, it IS
# possible — briefly, not sustained — for all GPU_CONCURRENCY slots to be
# genuinely busy even with exactly GPU_CONCURRENCY *users* connected (one
# session's ASR overlapping another session's TTS, etc.), in which case a
# call can wait a short moment for a slot rather than start instantly.
# That's intentional: it's a safety margin, not a bug — raise
# GPU_CONCURRENCY (env var, no code change) if this is observed often
# enough in practice to matter, once you have real usage data to size it
# from.
GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", "5"))
_GPU_SEMAPHORE  = threading.BoundedSemaphore(GPU_CONCURRENCY)
print(f"[GPU] capping concurrent ASR/translate/TTS calls at {GPU_CONCURRENCY} "
      f"(set GPU_CONCURRENCY to change).")


def _call_tts(text: str, language: str) -> dict:
    """Synthesize `text` in `language`'s fixed built-in voice. Thin wrapper
    around tts.synthesize() with two jobs: (1) apply the same text
    normalization (digits/%/currency -> spoken words, see
    _normalize_tts_text() below) every caller needs — a single choke point
    so the Chat Mode prefetch AND /speak's own fallback call both get it
    for free without duplicating the call; (2) base64-encode the raw
    "audio_bytes" MmsTTS.synthesize() returns into "audio_b64" — the
    JSON-over-the-wire shape every call site (speak_reply(),
    _push_speak_audio()) already expects, matching the old
    _call_tts_service()'s contract exactly so none of those call sites
    needed to change. Never raises — synthesize() itself already never
    does."""
    text = _normalize_tts_text(text, language)
    # See the "GPU-call concurrency cap" block above _call_tts() for why
    # this is bounded — note _normalize_tts_text() just ran ABOVE this
    # line and may itself have already acquired/released this same
    # semaphore once (for number-phrase translation, see
    # _localize_number_phrase_cached()) — sequential, non-nested
    # acquisitions on the same thread, never held at the same time, so
    # there's no deadlock risk here.
    with _GPU_SEMAPHORE:
        result = tts.synthesize(text=text, language=language)
    if result.get("success"):
        result = dict(result)
        result["audio_b64"] = base64.b64encode(result.pop("audio_bytes")).decode("ascii")
    return result
# ---------------------------------------------------------------------------

print("Loading Silero VAD…")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
(_, _, _, VADIterator, _) = vad_utils
print("VAD ready.")

TARGET_SR             = 16000
MIN_UTTERANCE_SAMPLES = int(TARGET_SR * 0.5)

LANG_LABELS = {
    "hin": "Hindi",
    "eng": "English",
    "ben": "Bengali",
    "kan": "Kannada",
    "tam": "Tamil",
    "ory": "Odia",
    "guj": "Gujarati",
    "mar": "Marathi",
}

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
_SESSIONS: dict = {}


def _new_vad():
    return VADIterator(
        vad_model,
        threshold=0.5,
        sampling_rate=TARGET_SR,
        min_silence_duration_ms=400,
        speech_pad_ms=200,
    )


def make_session(language="eng") -> dict:
    sid = str(uuid.uuid4())
    session = {
        "sid":          sid,
        "vad_iterator": _new_vad(),
        "audio_buffer": [],
        "remainder":    np.array([], dtype=np.float32),
        "speaking":     False,
        "transcript":    [],   # native script
        "translations":  [],   # English
        "status":        "",
        "stats":         "",
        "llm_response":     "",   # last Llama reply, translated to `language`
        "llm_response_en":  "",   # same reply, original English
        "llm_status":       "",   # "Thinking…" / "Answered in 1.2s" / error text
        "llm_busy":         False,   # True from the moment a turn is queued until its reply is shown —
                                     # see _fire_chat_turn()/_generate_llm_response(); drives the frontend's
                                     # "processing" indicator on the LLM Response panels.

        # Set when a tool (GAS's register_user/authenticate_user, e.g.)
        # needs one more piece of information and is mid-flow — see
        # tools/__init__.py's "needs_input" convention and
        # llama_llm.py's respond()/resume_followup(). While this is
        # non-None, the user's NEXT utterance is handed straight back to
        # THIS SAME tool (_generate_llm_response() below) instead of
        # going through the model's own tool-selection — exactly like
        # submitting the next field of a multi-step form. None means no
        # flow is in progress; a normal turn.
        "pending_tool_followup": None,
        "chat_mode":    False,   # when True, utterances accumulate into a debounced turn (see _schedule_chat_turn())
        "chat_history": [],   # [{"role": "user"/"assistant", "content": "..."}] — English only, oldest first
        "_pending_turn_texts": [],   # English translations accumulated since the last LLM answer
        "_pending_turn_timer": None, # the active trailing-silence threading.Timer, if any
        "_chat_turn_queue": queue.Queue(),   # combined-turn texts awaiting _generate_llm_response() —
                                             # see _chat_turn_worker() for why this exists
        "lock":         threading.Lock(),
        "language":     language,
        "ws":           None,   # active WebSocket for push updates

        # NOTE: no per-session voice-reference state at all — MMS-TTS's
        # voices are fixed, one dedicated model per language, selected
        # purely from session["language"] at TTS-call time (see
        # tts_mms.py). Nothing to record, nothing that can be
        # session-specific.

        # ── TTS prefetch ──
        # Set by _prefetch_tts_reply(), consumed by /speak — see both for
        # the full story. The idea: start synthesizing the LLM's reply as
        # soon as it's translated, in the background, so by the time the
        # user presses Play the audio is already there (or nearly so)
        # instead of /speak starting cold. _tts_prefetch_event is a fresh
        # threading.Event() per prefetch job (not reused) so a slow OLDER
        # job finishing late can never signal "ready" for a NEWER one.
        "_tts_prefetch_text":   None,   # the llm_response text the current/last prefetch job targeted
        "_tts_prefetch_result": None,   # that job's _call_tts() return dict, once it finishes
        "_tts_prefetch_event":  None,   # threading.Event for the in-flight/most recent job

        # ── live (unprompted) playback ──
        # See _prefetch_tts_reply() / the VAD start/end handling in
        # websocket_endpoint() for the full story. Lets the reply audio
        # play automatically — no Play click needed — while still never
        # talking over the user.
        #
        # "speaking" (above) is the USER's live mic VAD state. This is the
        # separate, server-tracked belief that the ASSISTANT's reply is
        # currently sounding out of the browser's speakers — set the
        # instant audio is pushed to the client (_push_speak_audio()),
        # cleared when playback ends normally (client calls
        # POST /session/{sid}/playback_ended) OR is cut short by a
        # barge-in (see below).
        "tts_speaking": False,

        # A reply that finished being synthesized WHILE the user was still
        # talking (session["speaking"] was True) isn't pushed right away —
        # it's held here instead, and delivered the moment the user's
        # current utterance ends (VAD "end" event), so the assistant never
        # starts talking over someone mid-sentence. Only ever holds at most
        # one entry: _chat_turn_worker() serializes turns (see its own
        # docstring), and _prefetch_tts_reply() runs synthesis to
        # completion before that worker picks up the next turn, so a
        # second reply can never finish while one is still pending here.
        "_pending_speak_result": None,

        # ── utterance processing queue ──
        # See _utterance_worker() below for the full story: every VAD-
        # detected utterance is put() here instead of getting its own
        # throwaway thread, and a SINGLE dedicated worker thread per
        # session pulls them off one at a time, in order — this is what
        # keeps transcript/translation lines in speech order and stops
        # overlapping ASR calls from fighting each other for the GPU.
        "_utterance_queue": queue.Queue(),
    }
    _SESSIONS[sid] = session
    threading.Thread(target=_utterance_worker, args=(session,), daemon=True).start()
    threading.Thread(target=_chat_turn_worker, args=(session,), daemon=True).start()
    return session


def _utterance_worker(session: dict):
    """The ONLY thing that calls _transcribe_utterance() for this session —
    pulls utterances off session["_utterance_queue"] one at a time, strictly
    in the order they were enqueued (VAD end-of-utterance / flush order).

    Previously each utterance got its own throwaway threading.Thread, fired
    off concurrently. That let results finish out of order — a longer or
    more complex utterance spoken FIRST could still finish transcribing
    AFTER a short one spoken right after it, so the transcript/translation
    boxes could show text out of speech order. It also meant multiple ASR
    calls (and, in Chat Mode, LLM/translation/TTS calls too) could all be
    mid-flight on the GPU at once, competing for the same resources —
    almost certainly why transcription had gotten noticeably slower.
    Serializing here fixes both: at most one utterance is ever being
    transcribed at a time for a given session, in the right order.

    Runs for the lifetime of the process (daemon thread, one per session,
    started once in make_session()) — there's no per-session shutdown
    needed for a demo like this, so the loop just runs forever.
    """
    q = session["_utterance_queue"]
    while True:
        utterance = q.get()
        try:
            _transcribe_utterance(utterance, session)
        except Exception as e:
            print(f"[ASR] worker error: {e}")
        finally:
            q.task_done()


def _chat_turn_worker(session: dict):
    """The ONLY thing that calls _generate_llm_response() for this session —
    pulls combined-turn texts off session["_chat_turn_queue"] one at a
    time, strictly in the order _fire_chat_turn() enqueued them.

    Without this, two debounced turns close together (e.g. a question
    about PM-KISAN, then — before that reply has actually finished coming
    back, since a tool-call round trip can take several seconds — a
    second question about the weather once its own silence gap elapses)
    would run _generate_llm_response() concurrently on two different
    threading.Timer threads. Both read/write the SAME session fields
    (llm_response, llm_response_en, chat_history) with no ordering
    guarantee between them, so whichever call happened to finish LAST
    would silently overwrite the other's reply — this is exactly the "I
    asked about PM-KISAN, then the weather, but only ever saw one answer"
    bug this fixes. Serializing here guarantees one turn is fully
    answered (and shown) before the next one is even started, same
    pattern as _utterance_worker() for ASR.

    Runs for the lifetime of the process (daemon thread, one per session,
    started once in make_session()).
    """
    q = session["_chat_turn_queue"]
    while True:
        combined_text = q.get()
        try:
            with session["lock"]:
                still_on = session.get("chat_mode", False)
            # Re-check chat_mode here too (not just when this was
            # enqueued) — it may have been switched off while this turn
            # was waiting behind an earlier one still in flight.
            if still_on and combined_text:
                _generate_llm_response(combined_text, session)
            else:
                with session["lock"]:
                    session["llm_busy"] = False
        except Exception as e:
            print(f"[LLM] chat-turn worker error: {e}")
            with session["lock"]:
                session["llm_busy"] = False
        finally:
            q.task_done()


def _reset_session(session: dict, status_msg: str = ""):
    """Wipe transcript/translation/LLM state back to a fresh session.
    Caller must already hold session["lock"]. Shared by /clear and the
    Chat Mode toggle — both are "start over" actions from the user's POV.
    """
    session["transcript"]      = []
    session["translations"]    = []
    session["audio_buffer"]    = []
    session["speaking"]        = False
    session["remainder"]       = np.array([], dtype=np.float32)
    session["status"]          = status_msg
    session["stats"]           = ""
    session["llm_response"]    = ""
    session["llm_response_en"] = ""
    session["llm_status"]      = ""
    session["llm_busy"]        = False
    session["chat_history"]    = []
    # A mid-flow tool (GAS registration/authentication — see
    # make_session()) shouldn't survive a reset into a conversation
    # that's just been wiped; the next utterance should start fresh.
    session["pending_tool_followup"] = None
    session["vad_iterator"]    = _new_vad()
    # Live-playback state (see make_session()) — the frontend already
    # pauses its own <audio> element on every action that calls this
    # function (Clear, language switch, Chat Mode toggle, Disconnect), so
    # there's nothing to push over the socket here; just stop believing
    # a reply is pending or playing so a stale one can't surface later.
    session["tts_speaking"]           = False
    session["_pending_speak_result"]  = None

    # Cancel any pending debounced Chat Mode turn — /clear and toggling
    # Chat Mode either direction should both wipe this out too, not just
    # leave a timer running that fires into a conversation that's just
    # been reset (harmless either way since _fire_chat_turn() re-checks
    # chat_mode itself, but tidier to stop it outright here).
    old_timer = session.get("_pending_turn_timer")
    if old_timer is not None:
        old_timer.cancel()
    session["_pending_turn_timer"] = None
    session["_pending_turn_texts"] = []

    # Also drain any turn(s) already queued for _chat_turn_worker() — a
    # question that was asked right before /clear (or a Chat Mode toggle)
    # shouldn't surface an answer afterward, into a conversation that's
    # just been wiped. The worker re-checks chat_mode itself too (belt and
    # suspenders), but draining here means it never even gets picked up.
    turn_queue = session.get("_chat_turn_queue")
    if turn_queue is not None:
        while True:
            try:
                turn_queue.get_nowait()
                turn_queue.task_done()
            except queue.Empty:
                break


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _to_float32(audio_np: np.ndarray) -> np.ndarray:
    if audio_np.dtype == np.float32:
        return audio_np
    if audio_np.dtype == np.int16:
        return audio_np.astype(np.float32) / 32768.0
    if audio_np.dtype == np.int32:
        return audio_np.astype(np.float32) / 2147483648.0
    return audio_np.astype(np.float32)


def _resample(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio_np
    g    = gcd(TARGET_SR, src_sr)
    up   = TARGET_SR // g
    down = src_sr    // g
    return resample_poly(audio_np, up, down).astype(np.float32)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
# Global event loop reference — set when the first WebSocket connects
_EVENT_LOOP = None

def _push_update(session: dict):
    """Push transcript/translation/LLM update to browser via WebSocket.
    Called from background threads — uses run_coroutine_threadsafe
    with the stored event loop reference.
    """
    ws = session.get("ws")
    if ws is None:
        return
    with session["lock"]:
        payload = {
            "type":            "update",
            "transcript":      " ".join(session["transcript"]),
            "translation":     " ".join(session.get("translations", [])),
            "status":          session["status"],
            "stats":           session["stats"],
            "llm_response":    session.get("llm_response", ""),
            "llm_response_en": session.get("llm_response_en", ""),
            "llm_status":      session.get("llm_status", ""),
            "llm_busy":        session.get("llm_busy", False),
        }
    import asyncio, json
    try:
        loop = _EVENT_LOOP
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps(payload)), loop
            )
            future.result(timeout=2)   # wait up to 2s for send to complete
        else:
            print("[PUSH] no event loop available")
    except Exception as e:
        print(f"[PUSH] error: {e}")


def _ws_send_json(session: dict, payload: dict):
    """Low-level: push one JSON message to this session's browser over its
    WebSocket, the same run_coroutine_threadsafe mechanism _push_update()
    uses. Shared by _push_speak_audio()/_push_stop_playback() below so
    there's exactly one place that talks to the socket directly."""
    ws = session.get("ws")
    if ws is None:
        return
    import asyncio, json
    try:
        loop = _EVENT_LOOP
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps(payload)), loop
            )
            future.result(timeout=2)
        else:
            print("[PUSH] no event loop available")
    except Exception as e:
        print(f"[PUSH] error: {e}")


def _push_speak_audio(session: dict, result: dict):
    """Push a synthesized reply to the browser for IMMEDIATE, unprompted
    playback — the "live" counterpart to the manual POST /speak endpoint.
    Called from _prefetch_tts_reply() the moment synthesis finishes, but
    ONLY once it's actually safe to start talking (see that function and
    the VAD "end" handling in websocket_endpoint() for the "hold while the
    user is still talking" logic — this function itself doesn't check
    that, it just sends).

    Marks session["tts_speaking"] = True so the VAD loop knows the
    assistant is now the one making sound — that's what lets a genuine
    barge-in (the user starting to talk mid-reply) be recognized and
    interrupt this playback, see the VAD "start" handling below.
    """
    with session["lock"]:
        session["tts_speaking"] = True
    _ws_send_json(session, {
        "type":          "speak",
        "mime_type":     result.get("mime_type"),
        "sampling_rate": result.get("sampling_rate"),
        "audio_b64":     result.get("audio_b64"),
    })


def _push_stop_playback(session: dict):
    """Tell the browser to immediately stop whatever reply audio is
    currently playing — a barge-in interrupt. Called the instant the VAD
    detects the user starting to speak while session["tts_speaking"] is
    True (see websocket_endpoint()). The client-side handler just pauses
    its <audio> element; the mic itself was never actually stopped (see
    the frontend's barge-in grace-period comments) — speech that just
    triggered this was already being heard and is already mid-utterance
    by the time this message goes out.
    """
    with session["lock"]:
        session["tts_speaking"] = False
    _ws_send_json(session, {"type": "stop_playback"})


def _transcribe_utterance(utterance_np: np.ndarray, session: dict):
    duration_s = len(utterance_np) / TARGET_SR
    # qsize() here is how many MORE utterances are already waiting behind
    # this one — a number that keeps growing across a session is a queue
    # backlog (speech arriving faster than one utterance can be
    # transcribed+translated), not a single slow model call. Helps tell
    # the two apart when transcription "feels slow" — see also the
    # native_pass/translation_pass split timing OmnilingualASR.transcribe()
    # prints for this same utterance right below.
    queue_depth = session["_utterance_queue"].qsize()
    print(f"[ASR] utterance: {len(utterance_np)} samples / {duration_s:.2f}s "
          f"(queue depth: {queue_depth})")

    if len(utterance_np) < MIN_UTTERANCE_SAMPLES:
        with session["lock"]:
            session["status"] = f"Too short ({duration_s:.2f}s) — speak longer"
        _push_update(session)
        return

    with session["lock"]:
        language  = session["language"]

    lang_name = LANG_LABELS.get(language, language.upper())

    # Two-step push: as soon as OmnilingualASR.transcribe()'s FIRST pass
    # (native script) is done, this callback fires and puts the transcript
    # on screen right away — instead of only after the second (English
    # translation) pass finishes too, which can take roughly as long again.
    # Total compute for this utterance is unchanged; only how soon the
    # native transcript becomes visible changes. This is also why the
    # transcript is appended HERE and not again below: this callback is the
    # single place that does it, whether or not translation follows.
    native_appended = False

    def _on_native_done(native_text: str):
        nonlocal native_appended
        text = native_text.strip()
        if not text:
            return
        with session["lock"]:
            session["transcript"].append(text)
            session["status"] = f"{lang_name} | translating…"
        native_appended = True
        _push_update(session)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        audio_int16 = (utterance_np * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(tmp_path, TARGET_SR, audio_int16)

        t0 = time.time()
        # See the "GPU-call concurrency cap" block above _call_tts().
        with _GPU_SEMAPHORE:
            result = asr.transcribe(tmp_path, src_lang=language,
                                     on_native_done=_on_native_done)
    except Exception as e:
        print(f"[ASR] error: {e}")
        with session["lock"]:
            session["status"] = f"Transcription error: {e}"
        _push_update(session)
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    elapsed       = time.time() - t0
    transcription = result["transcription"].strip()
    translation   = result.get("translation", "").strip()
    used_lang     = result.get("used_language", language)
    lang_name     = LANG_LABELS.get(used_lang, used_lang.upper())

    print(f"[ASR] native='{transcription}' eng='{translation}' lang={used_lang} ({elapsed:.2f}s)")

    if not transcription:
        with session["lock"]:
            session["status"] = "Empty result — try speaking more clearly"
        _push_update(session)
        return

    with session["lock"]:
        # _on_native_done() above already appended this exact string to
        # session["transcript"] the moment it was ready (it's the same
        # `transcription`, decoded once inside transcribe()) — appending it
        # again here would duplicate every utterance in the transcript. The
        # `not native_appended` fallback only covers the (currently
        # unreachable in production, but cheap to guard) case of an ASR
        # backend that doesn't call the callback at all.
        if not native_appended:
            session["transcript"].append(transcription)
        if translation:
            session["translations"].append(translation)
        session["status"] = f"{lang_name} | {elapsed:.2f}s"
        total_words = sum(len(t.split()) for t in session["transcript"])
        total_chars = sum(len(t)         for t in session["transcript"])
        total_eng_words = sum(len(t.split()) for t in session.get("translations", []))
        session["stats"] = (
            f"Utterances: {len(session['transcript'])} | "
            f"Words: {total_words} | Chars: {total_chars}"
        )
        chat_mode = session.get("chat_mode", False)
    _push_update(session)

    # Chat Mode: accumulate this utterance's translation into the pending
    # turn and (re)start the trailing-silence timer — the LLM only actually
    # gets asked once the user pauses for CHAT_MODE_DEBOUNCE_S (see
    # _schedule_chat_turn() / _fire_chat_turn()), with everything said
    # since the last answer, not just this one utterance in isolation.
    if chat_mode and translation:
        with session["lock"]:
            session["_pending_turn_texts"].append(translation)
            pending_count = len(session["_pending_turn_texts"])
        # A provisional status once there's more than one utterance
        # pending — not on the very first, which might just be an opening
        # remark with more to follow. Replaced by the real "Thinking…"
        # status once the debounce window elapses and the LLM is actually
        # asked (see _generate_llm_response()).
        if pending_count == 2:
            with session["lock"]:
                session["llm_status"] = "I am looking into the details…"
            _push_update(session)
        _schedule_chat_turn(session)


def _flush_buffer(session: dict):
    with session["lock"]:
        if not session.get("speaking") or not session["audio_buffer"]:
            return
        session["speaking"] = False
        utterance = np.concatenate(session["audio_buffer"])
        session["audio_buffer"] = []
    print(f"[VAD] flush — {len(utterance)/TARGET_SR:.2f}s")
    # Enqueue rather than transcribe directly on THIS thread — keeps flushed
    # utterances in the same strict order as everything else (see
    # _utterance_worker()); this call itself already runs on its own
    # short-lived thread (see call sites), so putting onto the queue here
    # is fine, it's just a quick, thread-safe handoff.
    session["_utterance_queue"].put(utterance)


# ---------------------------------------------------------------------------
# LLM response (on-demand, single-turn — no chat history is kept)
#
# Input  : the ENGLISH translation of what the user said (not the native
#          transcript) — this is what session["translations"] holds.
# Output : Llama replies in English, then IndicTrans2 (not the ASR model —
#          see indic_translator.py for why) translates that reply into
#          whatever language the session is currently set to, so the
#          frontend can render it in native script.
# ---------------------------------------------------------------------------
MAX_CHAT_HISTORY_TURNS = 8   # keep the last N user+assistant exchanges per session

# ── Chat Mode: debounced turn detection ──
# Originally every single utterance immediately fired its own
# _generate_llm_response() call. That meant a person thinking out loud
# across a few short utterances got asked about EACH one separately,
# out of context, and (worse) piled up concurrent LLM+translation+TTS
# calls competing with ASR for the same GPU/CPU — a real contributor to
# transcription itself feeling slower. Now each utterance's English
# translation is appended to a pending-turn buffer, and a short trailing-
# silence timer is (re)started: if the user starts talking again before it
# elapses, the timer is cancelled and restarted, so mid-thought pauses
# don't cut anything off early. Only once the user has been quiet for
# CHAT_MODE_DEBOUNCE_S does the LLM actually get asked, with everything
# accumulated since the last answer as one combined turn.
CHAT_MODE_DEBOUNCE_S = 1.75


def _fire_chat_turn(session: dict):
    """Timer callback — runs on its own thread once CHAT_MODE_DEBOUNCE_S
    has elapsed with no new utterance. Guards against a race where a new
    utterance arrived (and rescheduled the timer) in the brief window
    between this timer firing and it acquiring the lock: only proceeds
    with the text that was pending WHEN THIS RAN, and only if Chat Mode is
    still on (it may have been switched off while this was waiting)."""
    with session["lock"]:
        texts = session.get("_pending_turn_texts") or []
        session["_pending_turn_texts"] = []
        session["_pending_turn_timer"] = None
        chat_mode = session.get("chat_mode", False)

    if not chat_mode:
        return
    combined_text = " ".join(t.strip() for t in texts if t and t.strip())
    if not combined_text:
        return

    # Hand off to the dedicated chat-turn worker rather than calling
    # _generate_llm_response() directly here — see _chat_turn_worker() for
    # why (keeps two debounced turns from racing on the same session
    # fields if the previous one is still in flight). llm_busy flips on
    # right now (not just once the worker actually picks this up), so the
    # frontend's "processing" indicator appears the instant a turn is
    # queued, not only once it starts generating.
    turn_queue = session["_chat_turn_queue"]
    with session["lock"]:
        session["llm_busy"] = True
        if turn_queue.qsize() > 0:
            session["llm_status"] = "Queued — answering the previous question first…"
    _push_update(session)
    turn_queue.put(combined_text)


def _schedule_chat_turn(session: dict):
    """Called after each utterance while Chat Mode is on, once its text
    has already been appended to session["_pending_turn_texts"] —
    (re)starts the trailing-silence timer rather than asking the LLM
    immediately. Cancels whichever timer was already running (if the user
    is still talking, that one hasn't fired yet) and starts a fresh one,
    so only the LAST utterance's silence gap actually has to elapse
    before the LLM is asked, however many utterances came before it."""
    with session["lock"]:
        old_timer = session.get("_pending_turn_timer")
        if old_timer is not None:
            old_timer.cancel()
        timer = threading.Timer(CHAT_MODE_DEBOUNCE_S, _fire_chat_turn, args=(session,))
        timer.daemon = True
        session["_pending_turn_timer"] = timer
        timer.start()

# NOTE: this used to also warm a cache of pre-translated "Let me check
# the weather/scheme details for you..." filler lines, shown the instant
# llm.respond()'s on_tool_call callback fired (tool-augmented turns took
# an extra Ollama round trip, so this covered that latency). That whole
# pre-translated-audio-filler mechanism is STILL removed even though
# get_weather/file_grievance tools are back (see llama_llm.py's module
# docstring) — _generate_llm_response()'s _on_tool_call now only updates
# the TEXT status line, not spoken audio. Revisit this cache if a spoken
# filler during tool calls turns out to be worth it in practice.


# ---------------------------------------------------------------------------
# TTS text normalization — numbers/percentages/currency spoken as WORDS
# ---------------------------------------------------------------------------
# Originally added for a reported bug with Sooktam2 (didn't read "12%", or
# bare digits generally, out loud) — kept as-is through both Indic
# Parler-TTS and now MMS-TTS: most TTS models have no built-in front-end
# that expands digits and symbols into words before tokenizing, and
# there's no reason to assume this one is different. That's a gap in what
# reaches the TTS call, not something specific to English, so this is
# applied to every language this app supports, not only English.
#
# Approach: every number found in the text (percentages, ₹/Rs currency, and
# plain digit runs) is converted to English words with num2words, then —
# for any language other than English — that short phrase (e.g. "twelve
# percent") is run through the SAME already-loaded IndicTrans2 model this
# app already trusts for everything else (translator.translate()), instead
# of a hand-authored word list per language for "percent"/"rupees" that
# would be hard to get right (and hard to verify) without a native speaker
# listening to the result. Results are cached — see _NUMBER_PHRASE_CACHE —
# since the same amounts (a scheme's fixed benefit amount, a percentage
# quoted repeatedly) recur often.
#
# IMPORTANT / not yet confirmed live: this has been reasoned through and
# unit-tested offline (see test_tts_integration.py), but not confirmed
# against a real MMS-TTS call on real hardware. The [TTS] normalize
# print below shows exactly what text is being sent so it's easy to spot,
# per language, whether any digits are still leaking through unspoken —
# please check a few number-heavy replies (in English AND in whichever
# Indic language you actually use) and report back.
_NUMBER_PATTERN = re.compile(
    r'(?P<currency>(?:₹|Rs\.?)\s?(?P<currency_num>\d[\d,]*(?:\.\d+)?))'
    r'|(?P<percent>(?P<percent_num>\d[\d,]*(?:\.\d+)?)\s?%)'
    r'|(?P<bare>\b\d[\d,]*(?:\.\d+)?\b)'
)

_number_phrase_lock = threading.Lock()
_NUMBER_PHRASE_CACHE: dict = {}   # (english_phrase, target_lang) -> localized text
_NUMBER_PHRASE_CACHE_MAX = 2000   # numbers aren't a small fixed set like fillers —
                                  # cap so a long-running process can't grow this
                                  # unboundedly; simply stops caching past the cap
                                  # (still normalizes correctly, just re-translates)
                                  # rather than doing anything more disruptive.


def _number_to_english_words(num_str: str) -> str:
    """'1,00,000' / '12.5' / '12' -> 'one hundred thousand' / 'twelve point
    five' / 'twelve'. Indian-style comma grouping (1,00,000) and Western
    (100,000) both just have their commas stripped — the digits are what
    matter, not how they were grouped."""
    cleaned = num_str.replace(",", "")
    if "." in cleaned:
        value = float(cleaned)
    else:
        value = int(cleaned)
    return num2words(value)


def _localize_number_phrase_cached(english_phrase: str, target_lang: str) -> str:
    if target_lang == "eng":
        return english_phrase
    key = (english_phrase, target_lang)
    with _number_phrase_lock:
        cached = _NUMBER_PHRASE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        # See the "GPU-call concurrency cap" block above _call_tts().
        with _GPU_SEMAPHORE:
            localized = translator.translate(english_phrase, target_lang)
    except Exception as e:
        print(f"[TTS] number-phrase translation failed for {key}: {e}")
        localized = english_phrase
    with _number_phrase_lock:
        if len(_NUMBER_PHRASE_CACHE) < _NUMBER_PHRASE_CACHE_MAX:
            _NUMBER_PHRASE_CACHE[key] = localized
    return localized


def _normalize_tts_text(text: str, target_lang: str) -> str:
    """Expand digit/percent/currency expressions in `text` into spoken
    words, in `target_lang`, before it's handed to the TTS service. A no-op
    (returns `text` unchanged) if num2words isn't installed, or if `text`
    has no digits at all — the common case, so this stays cheap."""
    if not _NUM2WORDS_AVAILABLE or not text or not any(c.isdigit() for c in text):
        return text

    def _replace(m: "re.Match") -> str:
        try:
            if m.group("currency"):
                words = _number_to_english_words(m.group("currency_num"))
                phrase_en = f"{words} rupees"
            elif m.group("percent"):
                words = _number_to_english_words(m.group("percent_num"))
                phrase_en = f"{words} percent"
            else:
                phrase_en = _number_to_english_words(m.group("bare"))
            return _localize_number_phrase_cached(phrase_en, target_lang)
        except Exception as e:
            # Never let one malformed/unexpected number break the whole
            # reply — worst case that ONE number is mispronounced (spoken
            # as digits, same as before this change), not the rest of it.
            print(f"[TTS] number normalization failed for {m.group(0)!r}: {e}")
            return m.group(0)

    normalized = _NUMBER_PATTERN.sub(_replace, text)
    if normalized != text:
        print(f"[TTS] normalized for speech ({target_lang}): {text!r} -> {normalized!r}")
    return normalized


def _prefetch_tts_reply(session: dict):
    """Kick off TTS synthesis for the CURRENT llm_response right away, and
    — this is the "live"/unprompted playback path — deliver it to the
    browser the moment it's ready, with NO Play click needed. Called from
    _generate_llm_response() right after a reply is translated, on that
    same background thread (already detached from the request that
    triggered it, so this doesn't block or slow anything down; the text
    reply has already been pushed to the browser by the time this runs).

    "Ready" alone isn't enough to start talking, though — see the
    docstring on session["_pending_speak_result"] (in make_session()) and
    on _push_speak_audio()/_push_stop_playback() for the full story:

      - If the user is quiet right now (session["speaking"] is False),
        push the audio immediately — _push_speak_audio() plays it as soon
        as it reaches the browser.
      - If the user is still mid-utterance (session["speaking"] is True),
        hold it in session["_pending_speak_result"] instead of talking
        over them — the VAD "end" handler in websocket_endpoint() is what
        actually delivers it, the instant the user goes quiet.

    The manual POST /speak endpoint is unaffected by any of this — a
    deliberate click always plays immediately, same as before this
    feature existed. This function only governs the automatic path.

    A no-op if there's no reply text yet to speak — MMS-TTS's voices are
    fixed/built-in (see tts_mms.py), so unlike the old voice-cloning
    design there's no separate "voice not ready" case to check for here
    at all; the only way this has nothing to do is an empty reply.

    Uses a FRESH threading.Event() per call rather than one shared,
    long-lived event: if the user asks a second question before this
    job finishes, a new call starts (its own text/event), and this
    older job's result — checked against session["_tts_prefetch_text"]
    by identity of the event object — is discarded rather than
    overwriting the newer one's slot or spuriously signaling it "ready".
    """
    with session["lock"]:
        text        = (session.get("llm_response") or "").strip()
        target_lang = session["language"]

    if not text:
        return   # nothing to speak yet

    my_event = threading.Event()
    with session["lock"]:
        session["_tts_prefetch_text"]   = text
        session["_tts_prefetch_result"] = None
        session["_tts_prefetch_event"]  = my_event

    result = _call_tts(text=text, language=target_lang)

    deliver_now = False
    with session["lock"]:
        if session.get("_tts_prefetch_event") is my_event:
            session["_tts_prefetch_result"] = result
            if result.get("success"):
                if session.get("speaking"):
                    # Still talking — hold it. The VAD "end" handler
                    # delivers it once they stop (see websocket_endpoint()).
                    session["_pending_speak_result"] = result
                else:
                    deliver_now = True
    my_event.set()

    if deliver_now:
        _push_speak_audio(session, result)


# If the user says one of these INSTEAD of answering a tool's question
# (session["pending_tool_followup"] set — see make_session()), the
# mid-flow GAS registration/authentication is abandoned and the
# utterance is treated as a normal, fresh turn instead. English only —
# by the time `text` reaches _generate_llm_response() below it's already
# the ASR pipeline's English translation (see make_session()'s
# "_pending_turn_texts" comment), never the native-script transcript.
_CANCEL_PHRASES = {
    "cancel", "cancel that", "nevermind", "never mind", "never mind that",
    "forget it", "stop", "stop that", "quit",
}


def _generate_llm_response(text: str, session: dict):
    with session["lock"]:
        session["llm_status"] = "Thinking…"
        pending = session.get("pending_tool_followup")
    _push_update(session)

    # Check for an outright cancel BEFORE deciding whether to resume a
    # pending flow below — falls through to a normal turn afterward with
    # pending already cleared, so whatever the model says next (a plain
    # "okay, never mind" is exactly right) needs no special-case here.
    if pending and text.strip().lower().strip(" .!?") in _CANCEL_PHRASES:
        with session["lock"]:
            session["pending_tool_followup"] = None
        pending = None

    try:
        with session["lock"]:
            target_lang = session["language"]
            history     = list(session.get("chat_history", []))   # copy — respond()/resume_followup() only read it

        t0            = time.time()

        # Tools are back (weather, GAS grievances, policy/scheme lookups —
        # see llama_llm.py's module docstring and the tools/ package
        # registry), so a turn CAN now take an extra Ollama round trip,
        # and some tools (GAS, Policies — both reached over MCP) add a
        # real network hop on top of that. _on_tool_call fires the
        # instant llm.respond() decides to call a tool — before that
        # tool actually runs — so the status line changes from
        # "Thinking…" to something specific instead of just sitting
        # there for however long the round trip + tool takes.
        #
        # The status text itself comes from status_text_for(), which asks
        # whichever tools/<name>/ package owns `name` for its own
        # human-readable line (falling back to a generic "Calling
        # <name>…" otherwise) — this function deliberately has NO
        # per-tool knowledge, so a brand-new tool package shows a sensible
        # status line with zero changes here. This only updates
        # session["llm_status"] (text, pushed over the websocket like
        # every other status change) — it does NOT (yet) synthesize/speak
        # a filler line the way the old pre-tools-removal design
        # apparently did (see the removed NOTE this replaced, above
        # _prefetch_tts_reply()); add that back here too if the silent
        # gap during a tool call still feels too long in practice.
        def _on_tool_call(name: str, args: dict):
            status = status_text_for(name, args)
            with session["lock"]:
                session["llm_status"] = status
            _push_update(session)

        # session["pending_tool_followup"] (see make_session() /
        # tools/__init__.py's "needs_input" convention) means some tool
        # — GAS's register_user/authenticate_user, say — asked the user
        # something LAST turn and is waiting on their answer. THIS
        # utterance is that raw answer, not a fresh question for the
        # model to route: resume_followup() re-enters that SAME tool
        # directly, with no Ollama call at all, so it can never
        # accidentally drift onto a different tool mid-flow. Both
        # branches return the exact same {"text", "followup"} shape (see
        # llama_llm.py), so everything below is unified regardless of
        # which path was taken.
        if pending:
            result = llm.resume_followup(pending, text, on_tool_call=_on_tool_call)
        else:
            result = llm.respond(text, history=history, on_tool_call=_on_tool_call)

        # Both respond() and resume_followup() are contractually required
        # to return {"text": str, "followup": dict|None} — see their own
        # docstrings in llama_llm.py. The ONLY realistic way `result` ends
        # up NOT being that shape (e.g. a bare string) is this file and
        # llama_llm.py being mismatched versions on disk — e.g. an older
        # llama_llm.py from before the tools/dict-return redesign, still
        # returning a plain string reply, paired with this newer server
        # file that already expects the dict shape. Checked explicitly
        # here, with a message that says exactly that, instead of letting
        # it fail as a bare "string indices must be integers, not 'str'"
        # two lines down with no indication of WHY.
        if not isinstance(result, dict) or "text" not in result:
            raise TypeError(
                f"llm.{'resume_followup' if pending else 'respond'}() "
                f"returned {result!r} — expected a "
                '{"text": str, "followup": dict|None} dict. This almost '
                "always means js_asr_demo_server_ai4bh_xslate.py and "
                "llama_llm.py on THIS server are mismatched versions "
                "(e.g. an old llama_llm.py predating the tools/dict-return "
                "redesign). Redeploy the current llama_llm.py alongside "
                "this file."
            )

        english_reply = result["text"]
        new_pending   = result["followup"]
        # See the "GPU-call concurrency cap" block above _call_tts().
        with _GPU_SEMAPHORE:
            native_reply = translator.translate(english_reply, target_lang)
        elapsed       = time.time() - t0

        print(f"[LLM] in='{text}' en_out='{english_reply}' "
              f"{target_lang}_out='{native_reply}' ({elapsed:.2f}s)"
              f"{' [awaiting more input]' if new_pending else ''}")

        with session["lock"]:
            session["llm_response_en"]       = english_reply
            session["llm_response"]          = native_reply
            session["llm_status"]            = f"Answered in {elapsed:.2f}s"
            session["llm_busy"]              = False   # the reply is about to be shown — see below
            session["pending_tool_followup"] = new_pending

            # Only a successful exchange joins the conversation — an error
            # below never gets appended, so a failed turn can't corrupt
            # the history the model sees next time. A mid-flow exchange
            # (new_pending set, or `pending` just consumed above) still
            # joins it same as any other turn — "what's your phone
            # number" / "9876543210" is as real a part of the
            # conversation as anything else, so a later unrelated
            # question still has that context available.
            hist = session.setdefault("chat_history", [])
            hist.append({"role": "user", "content": text})
            hist.append({"role": "assistant", "content": english_reply})
            max_msgs = MAX_CHAT_HISTORY_TURNS * 2
            if len(hist) > max_msgs:
                del hist[: len(hist) - max_msgs]
        _push_update(session)   # show the text reply right away

        # Now spend a bit more time on this same background thread starting
        # TTS synthesis — "in the background" relative to the user, since
        # the text reply above is already on their screen and nothing else
        # is waiting on this thread. See _prefetch_tts_reply() / /speak.
        # Applies identically whether this was a normal reply or a
        # tool's own "what's your phone number?" prompt — by this point
        # it's just text in session["llm_response"], same pipeline, no
        # special-casing needed to have it spoken aloud too.
        _prefetch_tts_reply(session)
        return
    except Exception as e:
        # Just f"{e}" only ever gives the message ("string indices must be
        # integers, not 'str'", say) with NO file/line — impossible to
        # localize among respond()/resume_followup()/translate(), all of
        # which run in this same try block. traceback.print_exc() prints
        # the full frame-by-frame traceback to the console (same stream
        # the other [LLM]/[ASR]/[GAS] prefixed lines already go to), so
        # the NEXT time this fires, the exact line is right there instead
        # of having to reason about it blind after the fact.
        print(f"[LLM] error: {e}")
        traceback.print_exc()
        with session["lock"]:
            session["llm_status"] = f"LLM error: {e}"
            session["llm_busy"]   = False
            # Don't leave a stale mid-flow marker behind after an error —
            # the user has no way to know the flow silently died, and a
            # later, unrelated utterance would otherwise get wrongly
            # routed back into resume_followup() for a flow that's gone.
            session["pending_tool_followup"] = None
        _push_update(session)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="OmnilingualASR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REST: create session ---
class SessionRequest(BaseModel):
    language: str = "eng"

@app.get("/")
def serve_index():
    return FileResponse("/home/tirthankar/converse/index_asr_demo_converse_ai4bh.html")


# NOTE: the separate /admin voice-enrollment page (admin_voice_enroll.html)
# is GONE — it only ever existed to record per-language reference clips for
# Sooktam2's voice cloning (see the deleted "per-language voice-clone
# reference profiles" block this replaced, in git history). Every TTS
# backend since (Indic Parler-TTS, now MMS-TTS) ships fixed, built-in
# voices (see tts_mms.py), so there's nothing left to enroll and nowhere
# for an admin page to write to. admin_voice_enroll.html is no longer
# referenced by this app at all — safe to delete from disk.



@app.post("/session")
def create_session(req: SessionRequest):
    if req.language not in LANG_LABELS:
        raise HTTPException(400, f"Unknown language '{req.language}'")
    session = make_session(req.language)

    return {"sid": session["sid"], "language": req.language}


# --- REST: update language ---
class LanguageRequest(BaseModel):
    language: str


@app.post("/session/{sid}/language")
def set_language(sid: str, req: LanguageRequest):
    session = _SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")
    if req.language not in LANG_LABELS:
        raise HTTPException(400, f"Unknown language '{req.language}'")
    with session["lock"]:
        session["language"]     = req.language
        session["transcript"]   = []
        session["translations"] = []
        session["audio_buffer"] = []
        session["speaking"]     = False
        session["status"]          = f"Language set to {LANG_LABELS[req.language]}"
        session["stats"]           = ""
        session["llm_response"]    = ""
        session["llm_response_en"] = ""
        session["llm_status"]      = ""
        session["chat_history"]    = []
        # See _reset_session()'s matching comment — the frontend already
        # pauses its own <audio> element on a language switch, so this
        # just keeps the server's belief in sync.
        session["tts_speaking"]           = False
        session["_pending_speak_result"]  = None
    return {"ok": True}


# --- REST: clear session ---
@app.post("/session/{sid}/clear")
def clear_session(sid: str):
    session = _SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")
    with session["lock"]:
        _reset_session(session, "Cleared — ready to listen.")
    return {"ok": True}


# --- REST: poll state (fallback if WebSocket unavailable) ---
@app.get("/session/{sid}/state")
def get_state(sid: str):
    session = _SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")
    with session["lock"]:
        state = {
            "transcript":      " ".join(session["transcript"]),
            "status":          session["status"],
            "stats":           session["stats"],
            "llm_response":    session.get("llm_response", ""),
            "llm_response_en": session.get("llm_response_en", ""),
            "llm_status":      session.get("llm_status", ""),
            "llm_busy":        session.get("llm_busy", False),
            "chat_mode":       session.get("chat_mode", False),
        }
    return state


# --- REST: ask the LLM (explicit trigger only — never fires automatically) ---
class AskRequest(BaseModel):
    text: Optional[str] = None   # defaults to the most recent transcribed utterance


@app.post("/session/{sid}/ask")
def ask_llm(sid: str, req: AskRequest):
    session = _SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")

    text = (req.text or "").strip()
    if not text:
        # Default to the most recent ENGLISH translation, not the native
        # transcript — the LLM is fed English input.
        with session["lock"]:
            text = session["translations"][-1] if session.get("translations") else ""

    if not text:
        raise HTTPException(
            400,
            "No English translation available yet — speak first, or pass "
            "`text` explicitly.",
        )

    threading.Thread(
        target=_generate_llm_response, args=(text, session), daemon=True
    ).start()
    return {"ok": True, "queried": text}


# --- REST: speak the current LLM reply aloud, in that language's fixed voice ---
# Explicit-trigger only (the "🔊 Play" button) — mirrors /ask's "never fires
# automatically" design. Always speaks the TRANSLATED reply (llm_response),
# in the session's current language, matching what's shown in the "LLM
# Response — Translated" box: a Hindi speaker asked in Hindi hears Hindi
# back, not English.
class SpeakRequest(BaseModel):
    text: Optional[str] = None   # defaults to the session's current translated LLM reply


@app.post("/session/{sid}/speak")
def speak_reply(sid: str, req: SpeakRequest):
    session = _SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")

    with session["lock"]:
        text            = (req.text or session.get("llm_response") or "").strip()
        target_lang     = session["language"]
        prefetch_text   = session.get("_tts_prefetch_text")
        prefetch_event  = session.get("_tts_prefetch_event")

    if not text:
        raise HTTPException(400, "No reply to speak yet — ask a question first.")

    # If _generate_llm_response() already kicked off a background
    # synthesis job for exactly this text (the common case: playing the
    # reply as-is, not a custom override), wait on THAT instead of
    # starting a second, redundant synthesis from scratch — by the time
    # Play is pressed it's usually done or close to it, turning what used
    # to be a multi-second "generating…" wait into little or none. A
    # custom `text` override, or a prefetch that hasn't finished after a
    # generous wait, falls straight through to a fresh synchronous call
    # below, exactly as before this feature existed.
    result = None
    if req.text is None and prefetch_event is not None and prefetch_text == text:
        if prefetch_event.wait(timeout=60.0):
            with session["lock"]:
                if session.get("_tts_prefetch_text") == text:
                    result = session.get("_tts_prefetch_result")

    if result is None:
        # target_lang is the session's own 3-letter code (hin/eng/ben/...) —
        # tts_mms.py maps it to that language's own MMS-TTS model itself,
        # so this process doesn't need any mapping of its own.
        result = _call_tts(text=text, language=target_lang)

    if not result.get("success"):
        raise HTTPException(500, f"TTS error: {result.get('error', 'unknown error')}")

    return {
        "ok":            True,
        "mime_type":     result["mime_type"],
        "sampling_rate": result["sampling_rate"],
        "audio_b64":     result["audio_b64"],   # already base64 — service encodes it
    }


# --- REST: playback state (manual Play button only) -----------------------
# The LIVE/automatic path (see _push_speak_audio()) sets/clears
# session["tts_speaking"] itself, since the SERVER is the one deciding to
# start that playback. A manually-triggered /speak call is different: the
# BROWSER decided to play it (the user clicked), so the browser is what
# tells the server when that starts and ends — these two endpoints exist
# purely so a manual Play can ALSO be interrupted by a barge-in, the same
# as a live/automatic reply can. Called by the frontend right before
# ttsAudio.play() and on its "ended"/"error" events respectively.
@app.post("/session/{sid}/playback_started")
def playback_started(sid: str):
    session = _SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")
    with session["lock"]:
        session["tts_speaking"] = True
    return {"ok": True}


@app.post("/session/{sid}/playback_ended")
def playback_ended(sid: str):
    session = _SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")
    with session["lock"]:
        session["tts_speaking"] = False
    return {"ok": True}


# --- REST: toggle Chat Mode ---
# When enabled, _transcribe_utterance() automatically asks the LLM about
# every utterance as soon as it's transcribed+translated (see there) — no
# per-utterance request from the frontend needed while it's on.
class ChatModeRequest(BaseModel):
    enabled: bool


@app.post("/session/{sid}/chat_mode")
def set_chat_mode(sid: str, req: ChatModeRequest):
    session = _SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")
    with session["lock"]:
        session["chat_mode"] = req.enabled
        # Reset transcript/translation/LLM state on every toggle (both
        # directions) — otherwise old utterances from before the flip stay
        # in session["transcript"]/["translations"] and _push_update() keeps
        # re-sending them (and any stale LLM reply) on the next update.
        _reset_session(
            session,
            f"💬 Chat Mode {'on' if req.enabled else 'off'} — ready to listen.",
        )
    _push_update(session)
    return {"ok": True, "chat_mode": req.enabled}


# --- WebSocket: audio streaming ---
@app.websocket("/ws/{sid}")
async def websocket_endpoint(websocket: WebSocket, sid: str):
    session = _SESSIONS.get(sid)
    if not session:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    session["ws"] = websocket
    print(f"[WS] connected: {sid}")

    # Store the running event loop so background threads can push updates
    import asyncio
    global _EVENT_LOOP
    _EVENT_LOOP = asyncio.get_event_loop()

    try:
        while True:
            # Browser sends raw PCM float32 LE as binary frames
            # along with a 4-byte header: [sample_rate: uint32 LE]
            data = await websocket.receive_bytes()

            if len(data) < 4:
                continue

            # First 4 bytes: sample rate as uint32
            src_sr   = int.from_bytes(data[:4], "little")
            pcm_data = data[4:]

            if len(pcm_data) == 0:
                # Empty frame = mic stopped → flush
                threading.Thread(
                    target=_flush_buffer, args=(session,), daemon=True
                ).start()
                continue

            audio_np = np.frombuffer(pcm_data, dtype=np.float32).copy()
            audio_np = _resample(audio_np, src_sr)
            audio_np = np.concatenate([session["remainder"], audio_np])
            n_windows = len(audio_np) // 512

            for i in range(n_windows):
                chunk_512    = audio_np[i * 512 : (i + 1) * 512]
                audio_tensor = torch.from_numpy(chunk_512).float()

                speech_event = session["vad_iterator"](
                    audio_tensor, return_seconds=False
                )

                if speech_event and "start" in speech_event:
                    barge_in = False
                    with session["lock"]:
                        session["audio_buffer"] = [chunk_512]
                        session["speaking"]     = True
                        # Barge-in: the user just started talking WHILE the
                        # assistant's own reply is sounding out of the
                        # speakers. See _push_stop_playback()'s docstring —
                        # this is the one place that gets triggered from.
                        # The mic was never stopped during playback (that's
                        # what makes catching this possible at all — see
                        # the frontend's barge-in grace-period comments for
                        # the tradeoff that involves), so this utterance is
                        # already being captured normally; all that's
                        # needed here is telling the browser to cut the
                        # reply off right now.
                        if session.get("tts_speaking"):
                            barge_in = True
                    if barge_in:
                        print("[VAD] barge-in — user started talking over the reply")
                        _push_stop_playback(session)

                elif session.get("speaking"):
                    with session["lock"]:
                        session["audio_buffer"].append(chunk_512)

                if speech_event and "end" in speech_event:
                    pending_result = None
                    with session["lock"]:
                        session["speaking"] = False
                        utterance = (
                            np.concatenate(session["audio_buffer"])
                            if session["audio_buffer"] else None
                        )
                        session["audio_buffer"] = []
                        # A reply that finished synthesizing while this
                        # utterance was still being spoken (see
                        # _prefetch_tts_reply()) was held back instead of
                        # talking over it — now that the user has gone
                        # quiet again, it's safe to deliver.
                        if session.get("_pending_speak_result") is not None:
                            pending_result = session["_pending_speak_result"]
                            session["_pending_speak_result"] = None
                    if pending_result is not None:
                        _push_speak_audio(session, pending_result)
                    if utterance is not None:
                        print(f"[VAD] end — {len(utterance)/TARGET_SR:.2f}s")
                        # Enqueue for this session's single ordered worker
                        # (see _utterance_worker()) instead of a throwaway
                        # thread per utterance — keeps results in speech
                        # order and stops overlapping ASR calls from
                        # competing for the GPU.
                        session["_utterance_queue"].put(utterance)

            session["remainder"] = audio_np[n_windows * 512:].copy()

    except WebSocketDisconnect:
        print(f"[WS] disconnected: {sid}")
        session["ws"] = None
        threading.Thread(
            target=_flush_buffer, args=(session,), daemon=True
        ).start()
