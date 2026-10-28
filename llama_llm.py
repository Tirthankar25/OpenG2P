# llama_llm.py — response generation via a local Ollama server running
# Llama 3.1, now agentic again. Tools live under tools/<name>/ (see
# tools/__init__.py for the convention) — this file doesn't know the
# details of any one tool, only the small registry contract every
# tools/<name>/client.py exposes (TOOLS, call_tool(), optionally
# status_text()/USAGE_NOTES). Adding a new tool later is ONE line in
# _TOOL_PACKAGES below plus the matching tools/<name>/ package — nothing
# else here changes.
#
# NOTE on history here (tools removed, then reintroduced): this file used
# to hand the model get_weather + search_scheme_docs and let it decide per
# turn whether to call them. Both were dropped at one point because every
# tool-augmented turn costs at least one extra Ollama round trip (model
# decides to call a tool -> tool runs -> model is called AGAIN with the
# tool result to produce the real answer) — pure added latency on exactly
# the turns that use a tool. That tradeoff is now accepted: live weather,
# grievance-filing, and scheme lookups are real capabilities this demo
# needs, and the round-trip cost only applies to turns that actually call
# a tool — a plain "how are you" turn is still exactly one /api/chat
# call, since the "tools" field in the request doesn't force a tool call,
# it just makes one available (verified against Ollama's own documented
# tool-calling format: https://github.com/ollama/ollama/blob/main/docs/api.md
# and https://registry.ollama.ai/blog/tool-support — see respond() below,
# built to match that exact request/response shape rather than guessed).
#
# This also quietly resolves the earlier Qwen3-vs-Llama3.1 tradeoff noted in
# this file's prior revisions: that whole discussion was about which model
# picks the RIGHT TOOL more reliably. Llama 3.1 is used here, and does its
# own tool selection via Ollama's native tool-calling — no separate
# intent-classification step routes the query first (see the longer
# reasoning for that choice below, at respond()).
#
#   respond(text, history=None, on_tool_call=None) — text is the ENGLISH
#     translation of what the user said (produced by the ASR pipeline).
#     `history` is an optional list of prior {"role": "user"/"assistant",
#     "content": "..."} turns for THIS session. `on_tool_call(name, args)`,
#     if given, fires synchronously the moment the model decides to call a
#     tool (before that tool actually runs) — the intended use is a
#     filler/status update to the user ("Checking the weather…", "Filing
#     your grievance…") during what would otherwise be a silent extra
#     round trip. Returns the ENGLISH reply.
#
# History itself is NOT stored here — this class is still just a thin,
# stateless client. The caller (js_asr_demo_server_ai4bh_xslate.py) owns
# each session's history list: it reads it before calling respond(), and
# appends the new user/assistant turn afterward. (Tool-call exchanges
# themselves — the assistant's tool_calls message, the tool result message
# — live only inside ONE respond() call's local `messages` list; they are
# NOT folded into the session's saved chat_history, so a later turn's
# history doesn't replay old tool plumbing back at the model.)
#
# Translating the English reply into the user's selected language is a
# SEPARATE step, handled by indic_translator.py (IndicTrans2) — see that
# file for why this isn't done here or by the ASR model.
#
# NOTE on keep_alive (added when GPU memory freed up by the tools removal
# prompted a look at LLM speed): the model itself generating tokens faster
# needs a smaller/different model (see OLLAMA_KEEP_ALIVE's own comment
# below for that tradeoff) — but a separate, easy latency win is just
# making sure Ollama doesn't UNLOAD Llama 3.1 between turns and pay full
# reload time on the next request. See OLLAMA_KEEP_ALIVE below; this is
# the lever actually being used here, not a model swap.
#
# --- Prereqs (on the backend host) -----------------------------------------
#   curl -fsSL https://ollama.com/install.sh | sh
#   ollama pull llama3.1:8b
#   ollama serve                    # usually already running as a systemd service
#
# --- Sanity check ------------------------------------------------------------
#   curl http://localhost:11434/api/chat -d \
#     '{"model":"llama3.1:8b","messages":[{"role":"user","content":"hello"}],"stream":false}'
#
#   # Confirm it's actually staying resident (run right after a request,
#   # then again a few minutes later with no requests in between — both
#   # should show llama3.1:8b with keep_alive now that this is wired in):
#   ollama ps
# -----------------------------------------------------------------------------

import json
import os
import requests
from datetime import datetime

from tools.weather import client as weather_tools
from tools.GAS import client as gas_tools
from tools.policies import client as policy_tools

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:
    _IST = None   # falls back to server-local time if tzdata isn't available

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_TIMEOUT_S  = float(os.getenv("LLM_TIMEOUT_S", "60"))

# Hard cap on how many tool-call <-> tool-result rounds ONE respond() call
# will go through before forcing a final answer. A single turn needing
# two tools in one go (e.g. "what's the weather in Patna, also file a
# complaint about...") is 2 rounds; this leaves headroom without risking
# an unbounded loop if the model gets stuck re-requesting the same tool.
MAX_TOOL_ROUNDS = 3

# --- Tool registry — see tools/__init__.py for the convention every -------
# --- tools/<name>/ package follows. ADD A NEW TOOL BY EDITING ONLY THIS ---
# --- LIST (plus writing the tools/<name>/ package itself). ----------------
_TOOL_PACKAGES = [weather_tools, gas_tools, policy_tools]

# Flattened Ollama tool-schema list (the exact {"type": "function",
# "function": {...}} shape Ollama's /api/chat "tools" field expects —
# verified against Ollama's own docs rather than assumed, see module
# docstring) and a name -> owning-package lookup, both built ONCE here
# rather than recomputed per call.
_TOOLS = [t for _pkg in _TOOL_PACKAGES for t in _pkg.TOOLS]

_TOOL_OWNER = {}
for _pkg in _TOOL_PACKAGES:
    for _t in _pkg.TOOLS:
        _TOOL_OWNER[_t["function"]["name"]] = _pkg

# Ollama's own default behavior is to UNLOAD a model from GPU memory after
# 5 minutes with no requests, then reload it (paying the full load time
# again) the next time it's needed — a real, user-visible latency spike on
# whichever turn happens to land right after an idle gap, completely
# separate from how fast the model itself generates tokens. Passed as
# "keep_alive" on every /api/chat call below (Ollama reads it per-request,
# not just at load time, so this takes effect immediately with no server
# restart). "-1" means "keep it loaded until Ollama itself restarts or the
# model is explicitly unloaded" — i.e. never pay reload latency during
# normal use. Set OLLAMA_KEEP_ALIVE to a duration like "30m" instead if you
# want it to still free the GPU after a long idle stretch (e.g. because
# something else on the same GPU — the ASR model, TTS — needs the headroom
# back); "-1" is the simplest choice for a demo that's being actively used.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "-1")

_BASE_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. The user's messages were produced by "
    "a speech-to-text + translation pipeline, so they may contain minor "
    "errors — interpret intent charitably. Reply in English, in 1-3 short "
    "sentences suitable for being read aloud (and then translated for the "
    "user). Do not use markdown formatting."
)

# Reported bug: asked something like "from here to Mumbai", the model
# silently assumed "here" meant Delhi and answered as if that were given
# information — just the LLM defaulting to the single most common Indian
# city in its training data whenever a question depends on a location it
# was never actually told. Explicit, repeated instruction not to do that
# — the earlier, softer phrasing ("the user's messages were produced by a
# speech-to-text pipeline...") apparently wasn't a strong enough signal to
# stop it defaulting to a specific city on its own, so this spells out the
# rule directly rather than relying on the model to infer it. Still
# correct now that get_weather exists below: the tool needs a real city
# argument, and this instruction is exactly what stops the model from
# silently passing a guessed one.
_LOCATION_INSTRUCTIONS = (
    "You have NO information about where the user actually is — no GPS, "
    "no IP location, nothing. If the user says something like 'here', "
    "'my city', 'near me', 'from here', or otherwise refers to their own "
    "location without naming a specific place, do NOT guess or assume any "
    "particular city (not Delhi, not any other) and do NOT answer — or "
    "call a tool — as if you were told one. Either ask which city/place "
    "they mean, or answer in a way that doesn't depend on knowing their "
    "location."
)



def _current_datetime_line() -> str:
    """A ground-truth date+time line injected fresh on every call — the
    model itself has no clock, so without this it would guess/hallucinate
    when asked what today's date or the current time is. Cheap and never
    wrong to include, so it stays as always-on context."""
    now = datetime.now(_IST) if _IST else datetime.now()
    return (
        f"Right now it is {now.strftime('%A, %d %B %Y, %H:%M')} "
        f"India Standard Time (IST)."
    )


# Reported bug: asked an UNRELATED question ("how long to travel from
# Bangalore to Chennai" — nothing to do with any tool here), the model
# answered correctly but prefaced it by narrating that the question
# "isn't related to PM-KISAN/SMAM" — naming schemes the user never
# mentioned. Root cause, confirmed by adding the tool_calls print in
# respond() below rather than guessed: the model called a Policies tool
# (get_policy_info or list_available_policies) ANYWAY for an unrelated
# question, got back a result naming PM-KISAN/SMAM (either real scheme
# data or get_policy_info's own "not found — available: [...]" fallback,
# see tools/policies/server.py), and then — with nothing telling it tool
# results/reasoning should stay invisible to the user — narrated that
# tool result straight into its reply. Two-part fix: (1) tool
# descriptions/USAGE_NOTES below are explicit that a tool should NOT be
# called for questions outside its own subject (also tightened in each
# tools/<name>/client.py's own TOOLS descriptions, e.g.
# tools/policies/client.py's get_policy_info), since the model deciding
# NOT to call anything is the cheaper, more direct fix than papering over
# a wrong call after the fact; (2) _NO_TOOL_NARRATION below as a backstop
# regardless of which tool, if any, actually got called.
_NO_TOOL_NARRATION = (
    "Never mention a tool's name, the examples in its description (e.g. "
    "scheme names), or your own reasoning about whether a tool applies, "
    "anywhere in your reply — not even to say a tool ISN'T relevant. If "
    "none of your tools are relevant to what the user asked, simply "
    "don't call any and answer directly and naturally, exactly as if you "
    "had no tools at all."
)


def _tool_usage_instructions() -> str:
    """Built from each registered package's OWN USAGE_NOTES (see
    tools/__init__.py's convention) rather than hand-maintained here —
    this is the extensibility payoff: a new tools/<name>/ package brings
    its own behavioral guidance for the model along with it, so this file
    never needs editing just because a tool was added/changed elsewhere.
    A package that omits USAGE_NOTES simply contributes nothing extra
    here — its TOOLS entries' own "description" fields are still visible
    to the model regardless, just without the sharper behavioral rules
    USAGE_NOTES is for (see tools/__init__.py). _NO_TOOL_NARRATION above
    is a global rule that applies regardless of which (if any) package is
    registered, so it's folded in here rather than left to each package
    to repeat."""
    notes = [getattr(pkg, "USAGE_NOTES", None) for pkg in _TOOL_PACKAGES]
    notes = [n.strip() for n in notes if n]
    if not notes:
        return "You have no tools available right now."
    return ("You have tools available. " + " ".join(notes)
            + " " + _NO_TOOL_NARRATION)


def _system_prompt() -> str:
    return (
        _BASE_SYSTEM_PROMPT + "\n\n" + _current_datetime_line() + "\n\n"
        + _LOCATION_INSTRUCTIONS + "\n\n" + _tool_usage_instructions() + "\n\n"
        "This is a multi-turn conversation — prior turns are included above "
        "the latest message. Use them for context."
    )


def _keep_alive_value():
    """Ollama accepts either a bare number of seconds (int, e.g. -1 or
    3600) or a duration string like "30m" — try the numeric form first so
    "-1"/"0"/"3600" are sent as real numbers (matches Ollama's own CLI/API
    examples), falling back to the raw string for anything that isn't just
    digits. Pulled into its own function since both the tool-calling round
    and the final round of respond() below now need it."""
    return (int(OLLAMA_KEEP_ALIVE)
            if OLLAMA_KEEP_ALIVE.lstrip("-").isdigit()
            else OLLAMA_KEEP_ALIVE)


def _execute_tool(name: str, args: dict) -> str:
    """Runs ONE model-requested tool call and returns its result as a
    STRING — Ollama's "tool" role message takes `content` as text (see the
    module docstring's link to Ollama's docs), even though the `arguments`
    coming IN from the model are a real JSON object, not a string.

    Routes purely by looking up which registered package owns `name`
    (built once into _TOOL_OWNER above) and delegating to that package's
    OWN call_tool() — this function has no per-tool knowledge at all, so
    a new tools/<name>/ package works here with zero changes to this
    file. A missing/unknown tool name, or a package's call_tool() itself
    raising, is reported back to the model as a normal tool-result string
    rather than propagated, so a single bad tool_calls entry can't blow
    up the rest of the turn.
    """
    args = args or {}
    pkg = _TOOL_OWNER.get(name)
    if pkg is None:
        return f"Error: no such tool '{name}'."
    try:
        return pkg.call_tool(name, args)
    except Exception as e:
        # Mirrors every other external call in this app: a tool blowing up
        # becomes a clear error STRING fed back to the model (which can
        # apologize / ask the user to try again) rather than an exception
        # that kills the whole turn. Packages are documented to catch
        # their own errors (see tools/__init__.py) — this is a second line
        # of defence, not the primary one.
        print(f"[LLM] tool '{name}' raised during execution: {e!r}")
        return f"Error: the '{name}' tool failed ({e})."


def _try_parse_tool_json(result_text) -> dict:
    """Tool results are always STRINGS (see tools/__init__.py) — most are
    json.dumps()'d dicts, but a package's own error path (or
    _execute_tool()'s "no such tool"/exception fallback above) can return
    a plain, non-JSON string instead. Returns the parsed dict, or None if
    `result_text` isn't one — used below to detect the "needs_input"
    multi-step convention (see tools/__init__.py) without assuming every
    tool result is JSON."""
    if not isinstance(result_text, str):
        return None
    try:
        parsed = json.loads(result_text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def status_text_for(name: str, args: dict) -> str:
    """Best-effort human-readable status line for `name` being called
    with `args` — for a CALLER (js_asr_demo_server_ai4bh_xslate.py's
    on_tool_call) to show while the tool/round-trip is in flight. Looks
    up the owning package's OWN status_text() (see tools/__init__.py);
    falls back to a generic line if the package didn't define one, or
    isn't recognized at all — a new tool package works immediately even
    if its author skipped status_text(), just with a less specific
    message, and this never raises regardless of what a package's
    status_text() does.
    """
    pkg = _TOOL_OWNER.get(name)
    if pkg is not None:
        fn = getattr(pkg, "status_text", None)
        if fn is not None:
            try:
                return fn(name, args or {})
            except Exception:
                pass
    return f"Calling {name}…"


class LlamaResponder:
    """Thin, mostly-stateless client for a local Ollama server running
    Llama 3.1, via /api/chat so a per-session message history can be
    passed in for a real multi-turn conversation. Tools come from the
    tools/<name>/ registry (_TOOL_PACKAGES above) — Llama decides per
    turn whether/which to call via Ollama's native tool-calling, there is
    no separate routing/classifier step (see respond()'s docstring for
    why). See indic_translator.py for the translation step that turns the
    reply into the user's selected language.
    """

    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL,
                 timeout_s: float = LLM_TIMEOUT_S):
        self.base_url  = base_url.rstrip("/")
        self.model     = model
        self.timeout_s = timeout_s
        self._check_available()

    def _check_available(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            if not any(self.model in n for n in names):
                print(f"[LLM] warning: model '{self.model}' not found via "
                      f"`ollama list` (have: {names}). Run `ollama pull {self.model}`.")
        except Exception as e:
            print(f"[LLM] warning: could not reach Ollama at {self.base_url} ({e}). "
                  f"Is `ollama serve` running?")

    def _one_chat_call(self, messages: list, with_tools: bool) -> dict:
        """One raw POST /api/chat, returning the response's "message" dict
        (never raises on an HTTP error — resp.raise_for_status() does
        that, same as before this method existed, just factored out since
        respond() below now needs to make this call more than once)."""
        payload = {
            "model":      self.model,
            "messages":   messages,
            "stream":     False,
            "options":    {"temperature": 0.7},
            # See OLLAMA_KEEP_ALIVE above — keeps the model resident in GPU
            # memory between turns instead of paying Ollama's default
            # 5-minute-idle reload penalty.
            "keep_alive": _keep_alive_value(),
        }
        if with_tools:
            payload["tools"] = _TOOLS
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}) or {}

    def respond(self, text: str, history: list = None, on_tool_call=None) -> dict:
        """Generate an ENGLISH reply to an ENGLISH `text` input, optionally
        continuing a conversation. One /api/chat round trip for a plain
        turn; one ADDITIONAL round trip per round of tool calls the model
        makes (bounded by MAX_TOOL_ROUNDS above) — see the module
        docstring for why this cost is now accepted.

        Returns {"text": str, "followup": dict|None} — NOT a bare string
        (see tools/__init__.py's "needs_input" convention). "followup"
        is None for a normal, finished reply. It is NON-None when a tool
        (e.g. GAS's register_user/authenticate_user) needs one more piece
        of information from the user before it can finish: "text" is then
        the exact question to speak to them, and "followup" is
        {"tool_name", "continuation"} — hang onto it (callers store it as
        session["pending_tool_followup"]) and pass it to
        resume_followup() below with the user's NEXT reply, rather than
        calling respond() again. respond() itself never recurses into a
        followup — it only ever starts one.

        text    : the English translation your ASR pipeline already
                  produced — not the native-script transcript.
        history : optional list of prior {"role": "user"/"assistant",
                  "content": str} turns for this session, oldest first.
                  Pass None (or []) for a fresh, single-turn exchange. Only
                  plain user/assistant turns belong here — tool-call
                  plumbing from a PAST respond() call is intentionally not
                  part of this (see module docstring), so don't reconstruct
                  it from an old turn and pass it back in.
        on_tool_call : optional callable (name: str, args: dict) -> None,
                  fired synchronously the instant the model decides to
                  call a tool, BEFORE that tool actually runs (so a slow
                  tool call, like the GAS network round trip, doesn't look
                  like a silent hang). Exceptions it raises are caught and
                  logged, never allowed to break the actual turn.

        Routing note — why there's no separate "does this need a tool"
        classifier ahead of this: with a small handful of tools (see
        _TOOL_PACKAGES above), each with a distinct, clearly-described
        purpose (a read-only weather lookup, a read-only scheme lookup, a
        grievance-filing action with real side effects, a status lookup),
        Ollama's native tool-calling is exactly the well-trodden case
        where letting the model itself decide is both simpler and more
        reliable than a hand-rolled router — it already has to understand
        the user's intent to answer at all, a separate classifier would
        duplicate that understanding in a second place that can drift out
        of sync as tools change, and the round-trip cost this
        avoided-separate-step argument used to be most worried about only
        actually happens on turns that need a tool in the first place (a
        plain chat turn is still one call, tools or not — see
        _one_chat_call() above). If a future tools/<name>/ addition pushes
        this into many tools, or tools with overlapping purposes the
        model confuses, that's the point to revisit this with a cheaper
        pre-filter; not needed yet.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("empty input text")

        messages = [{"role": "system", "content": _system_prompt()}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": text})

        for _round in range(MAX_TOOL_ROUNDS):
            message = self._one_chat_call(messages, with_tools=True)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return {"text": (message.get("content") or "").strip(), "followup": None}

            # Logged UNCONDITIONALLY (not just on error) specifically so
            # "why did it mention PM-KISAN for an unrelated question"
            # (or any future "why did it call X") is directly answerable
            # from the console instead of inferred from the model's own
            # reply text — that's indirect evidence at best, this is the
            # actual fact of what the model decided to call and with what
            # arguments.
            _names = [(c.get("function", {}) or {}).get("name", "?") for c in tool_calls]
            print(f"[LLM] model requested tool call(s): {_names} for input {text!r}")

            # The assistant's tool-call turn itself has to go into the
            # running message list before the tool results do, or the next
            # call has no record of what it asked for when the "tool" role
            # messages show up — same requirement as every other
            # provider's function-calling, confirmed against Ollama's own
            # docs rather than assumed.
            messages.append(message)

            for call in tool_calls:
                fn   = call.get("function", {}) or {}
                name = fn.get("name", "")
                args = fn.get("arguments") or {}   # already a dict, not a JSON string — Ollama-specific
                if on_tool_call:
                    try:
                        on_tool_call(name, args)
                    except Exception as e:
                        print(f"[LLM] on_tool_call callback raised ({e!r}) — continuing anyway")
                result_text = _execute_tool(name, args)

                # See tools/__init__.py's "needs_input" convention: a
                # tool can ask for one more piece of information instead
                # of finishing. Caught HERE, before this result ever goes
                # back to Ollama as a "tool" role message — there's no
                # sense asking the model to react to "I still need your
                # phone number" when the right answer is always exactly
                # the same (speak that prompt, wait for the reply), and
                # letting the model paraphrase it risks exactly the kind
                # of narration/hallucination bug _NO_TOOL_NARRATION above
                # already exists to avoid elsewhere. This returns
                # immediately, skipping any other tool_calls still left
                # in THIS round (a model asking for two slot-filling
                # tools in the same breath isn't a realistic case for the
                # tools registered here today).
                parsed = _try_parse_tool_json(result_text)
                if parsed and parsed.get("needs_input"):
                    prompt = (parsed.get("prompt") or "").strip() or "Could you say that again?"
                    print(f"[LLM] tool '{name}' needs more input: {prompt!r}")
                    return {
                        "text": prompt,
                        "followup": {"tool_name": name, "continuation": parsed.get("continuation")},
                    }

                messages.append({"role": "tool", "content": result_text, "tool_name": name})

        # Exhausted MAX_TOOL_ROUNDS rounds of tool calls without a plain
        # answer (shouldn't normally happen with two well-described
        # tools) — ask one final time with "tools" left OUT of the
        # payload, which forces a text answer instead of yet another
        # tool_calls response, using whatever tool results are already in
        # `messages` by now.
        message = self._one_chat_call(messages, with_tools=False)
        return {"text": (message.get("content") or "").strip(), "followup": None}

    def resume_followup(self, pending: dict, answer_text: str, on_tool_call=None) -> dict:
        """Continues a multi-step tool flow a PRIOR respond() call started
        (see tools/__init__.py's needs_input/continuation convention)
        with the user's raw next reply — e.g. the phone number they just
        spoke, in answer to a prompt respond() returned last turn.

        Deliberately does NOT go through Ollama at all: the model already
        decided to call this tool on an earlier turn; this just hands it
        the next piece of information directly, the same way submitting
        one field of a multi-step form doesn't require re-deciding which
        form you're filling out. That also means this can never
        accidentally drift onto a DIFFERENT tool mid-flow — it always
        re-enters the exact tool named in `pending`.

        Same {"text", "followup"} return shape as respond() (see that
        method's docstring) — "followup" non-None means still more
        input needed (call this again with the user's next reply and
        the new `followup`); None means the flow is finished, and "text"
        is the tool's own ready-to-speak success/failure message.

        `pending` : the exact {"tool_name", "continuation"} dict a prior
                  respond() (or resume_followup()) call returned as
                  "followup" — callers store this as
                  session["pending_tool_followup"].
        answer_text : the user's raw next reply, untouched (no parsing/
                  extraction attempted here — the tool itself validates
                  and, on a bad value like an unparseable phone number,
                  re-asks the SAME field; see tools/GAS/server.py).
        on_tool_call : same callback respond() takes, fired the same way
                  — existing status-line wiring needs no changes to cover
                  this path too.
        """
        name = (pending or {}).get("tool_name", "")
        continuation = (pending or {}).get("continuation")
        args = {"continuation": continuation, "answer": answer_text}

        if on_tool_call:
            try:
                on_tool_call(name, args)
            except Exception as e:
                print(f"[LLM] on_tool_call callback raised ({e!r}) — continuing anyway")

        result_text = _execute_tool(name, args)
        parsed = _try_parse_tool_json(result_text)

        if parsed and parsed.get("needs_input"):
            prompt = (parsed.get("prompt") or "").strip() or "Could you say that again?"
            print(f"[LLM] tool '{name}' needs more input: {prompt!r}")
            return {
                "text": prompt,
                "followup": {"tool_name": name, "continuation": parsed.get("continuation")},
            }

        # Flow finished — success or failure, either way speak the
        # TOOL's own message verbatim rather than looping this back
        # through Ollama to be rephrased: these are short,
        # already-written-for-a-citizen confirmation/error messages (see
        # tools/GAS/server.py) — re-narrating them through the model
        # would only add latency and a fresh chance to hallucinate over
        # a message that's already correct.
        text = None
        if parsed:
            text = parsed.get("message") or parsed.get("error")
        if not text:
            text = result_text if isinstance(result_text, str) else "Sorry, something went wrong with that."
        print(f"[LLM] tool '{name}' flow finished: {text!r}")
        return {"text": text, "followup": None}
