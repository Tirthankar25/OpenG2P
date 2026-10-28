# tts_text_normalize.py — pre-synthesis text cleanup for tts_mms.py (and,
# if ever revisited, tts_parler.py / tts_sooktam.py too — this is a
# standalone module precisely so any TTS backend can import it).
#
# --- Why this exists --------------------------------------------------------
# MMS-TTS's VitsTokenizer has a FIXED, per-language character vocabulary
# built from whatever characters happened to appear in that checkpoint's
# training text. A character that never showed up there — the degree sign
# °, %, ₹, etc. — isn't an error: it's simply not in the vocab, so the
# tokenizer silently drops it rather than raising. There's no text-
# normalization front-end built into VITS the way, say, eSpeak-NG expands
# "10-12" into "ten to twelve" for some other TTS stacks — VITS reads
# characters/phonemes, nothing more. Two concrete symptoms reported:
#   - "10°C"    → the ° vanishes; the model just says "10 C" (or worse).
#   - "10-12 hrs" → read character-by-character, never becomes "10 to 12
#     hours" the way a human would say it.
# This module expands exactly those patterns into plain words BEFORE the
# text reaches the tokenizer, so the vocab never has to see the symbol at
# all — same fix shape as uroman romanization in tts_mms.py: clean the
# text up front rather than hoping the model copes.
#
# --- Scope -------------------------------------------------------------------
# Fully implemented for English ("eng") only, since that's what was
# reported and these are English connector words ("degrees", "to",
# "percent", "rupees", "hours") that would sound wrong spliced into
# Hindi/Bengali/etc. output. Other languages currently pass straight
# through UNCHANGED (identical to behavior before this fix) — to extend
# to another language, add a same-shaped entry to _WORDS below with that
# language's own words; none of the regex logic needs to change, only the
# words it substitutes in.
#
# --- Known limitation ---------------------------------------------------------
# A leading minus sign on a number (e.g. "-5°C" for a sub-zero low) is not
# specially handled — the digits and ° still get expanded correctly, the
# minus sign itself is just left as-is. Not fixed here since it wasn't
# reported and temperatures in this app's domain (Indian weather) rarely
# go negative; flagging it so it's not mistaken for an oversight if it
# comes up later.
#
# --- Sanity check (standalone, no model/server needed) -----------------------
#   python3 -c "
#   from tts_text_normalize import normalize_for_tts
#   print(normalize_for_tts('High of 32°C, low of 18-20°C. Journey takes 10-12 hrs.', 'eng'))
#   "
# -----------------------------------------------------------------------------

import re

_WORDS = {
    "eng": {
        "degrees": "degrees",
        "to": "to",
        "percent": "percent",
        "rupees": "rupees",
        "hours": "hours",
        "minutes": "minutes",
    },
}

# Number-range hyphen, e.g. "10-12", "18-20.5" — deliberately NOT
# "2024-10-05" (a date) or "GAS-202410-00001" (a grievance ticket id):
# both of those are protected because the digit group on at least one
# side of their hyphens is itself immediately adjacent to ANOTHER hyphen,
# which the lookaround below excludes. A bare two-segment numeric id
# (e.g. "12-34") would still be mis-read as a range — not a pattern this
# app currently produces (ticket ids are GAS-YYYYMM-NNNNN, three
# segments), so left as a known tradeoff rather than over-engineered.
_RANGE_RE = re.compile(r"(?<!-)\b(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\b(?!-)")

# A number immediately followed by the degree sign, optionally followed
# by a unit letter (C/F) — "10°C", "10 ° C", "10°" are all handled.
_DEGREE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*°\s*([A-Za-z]?)")

# Any degree sign NOT already consumed above (e.g. a bare "°" with no
# leading number) — belt-and-suspenders so it never silently vanishes.
_STRAY_DEGREE_RE = re.compile(r"°")

_HRS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*hrs?\b", re.IGNORECASE)
_MIN_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*mins?\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_RUPEE_RE = re.compile(r"₹\s*(\d)")


def normalize_for_tts(text: str, language: str) -> str:
    """Expand symbols/abbreviations MMS-TTS's character-level tokenizer
    can't speak into words it CAN, for the given session language code
    (e.g. 'eng', 'hin' — one of tts_mms.SUPPORTED_LANGUAGES). Idempotent
    and safe to call on text that needs no changes — returns it
    unmodified. See module docstring for scope/limitations.
    """
    text = text or ""
    lang = (language or "eng").strip().lower()
    words = _WORDS.get(lang)
    if words is None:
        # No word table for this language yet — leave the text exactly as
        # it was before this fix, rather than splicing in English words.
        return text

    def _degree_sub(m: "re.Match") -> str:
        num, unit = m.group(1), m.group(2)
        if unit:
            return f"{num} {words['degrees']} {unit}"
        return f"{num} {words['degrees']}"

    # Order matters: degree expansion first, so a combined case like
    # "18-20°C" becomes "18-20 degrees C" before the range regex below
    # turns the now-unblocked "18-20" into "18 to 20" — giving "18 to 20
    # degrees C" rather than something the range regex would otherwise
    # have to parse around the ° sign for.
    text = _DEGREE_RE.sub(_degree_sub, text)
    text = _STRAY_DEGREE_RE.sub(f" {words['degrees']} ", text)
    text = _RANGE_RE.sub(lambda m: f"{m.group(1)} {words['to']} {m.group(2)}", text)
    text = _HRS_RE.sub(lambda m: f"{m.group(1)} {words['hours']}", text)
    text = _MIN_RE.sub(lambda m: f"{m.group(1)} {words['minutes']}", text)
    text = _PERCENT_RE.sub(lambda m: f"{m.group(1)} {words['percent']}", text)
    text = _RUPEE_RE.sub(lambda m: f"{words['rupees']} {m.group(1)}", text)

    # Collapse any double spaces the substitutions above may have left.
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text
