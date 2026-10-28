# tools/GAS/server.py — DUMMY external "Grievance Addressal System" (GAS),
# built as its own FastMCP server/process rather than a plain Python
# function, specifically because the real GAS this stands in for is an
# EXTERNAL system reached over the network — not something this app owns
# or could import directly, unlike tools/weather (a lookup this app does
# for itself against a public API, with no server.py at all — see that
# package's client.py docstring). Running GAS as a separate MCP server (a
# different process, possibly a different host entirely in production)
# and having llama_llm.py reach it as an MCP CLIENT (see
# tools/GAS/client.py) keeps that boundary honest even in this demo —
# swapping this dummy implementation for the REAL GAS later should only
# mean pointing GAS_MCP_URL at the real endpoint, not rewriting the app.
#
# Part of the tools/<name>/ convention — see tools/__init__.py for what
# every tool package exposes and how a new one gets added.
#
# This is a STUB: grievances live in an in-memory dict, lost on restart; a
# fake-but-plausible ticket ID is generated; there's no real routing,
# authentication, or persistence. Replace the body of file_grievance() /
# check_grievance_status() with real calls to an actual GAS backend when
# one exists — the MCP tool signatures below are the contract the rest of
# this app is written against, so a real backend can sit behind the same
# two functions without anything else needing to change.
#
# Verified against the actual installed `fastmcp` package (4.0.10) in a
# throwaway sandbox before writing this for real — @mcp.tool / mcp.run(
# transport="http", ...) and the matching fastmcp.Client(...).call_tool()
# shape are not guessed from memory, since this project has been bitten
# before by assuming a library's API instead of checking it.
#
# --- Prereqs (on whichever host runs GAS — can be the SAME host/venv as
#     the main app for this demo, or a genuinely separate machine) --------
#   pip install fastmcp
#
# --- Running ----------------------------------------------------------------
#   python3 tools/GAS/server.py          # or: tools/GAS/launch.sh
#   # Serves MCP over streamable-HTTP at http://<host>:8787/mcp — this is
#   # exactly what GAS_MCP_URL (see tools/GAS/client.py) must point at.
#   # Run this as its OWN process, separate from the main app — in a real
#   # deployment this would be someone else's server entirely, reachable
#   # only over the network.
#
# --- Sanity check (standalone, no main app needed) --------------------------
#   python3 -c "
#   import asyncio
#   from fastmcp import Client
#   async def main():
#       async with Client('http://127.0.0.1:8787/mcp') as c:
#           print(await c.call_tool('file_grievance', {
#               'complainant_name': 'Test User', 'contact_number': '9999999999',
#               'category': 'water_supply', 'description': 'No water for 3 days',
#           }))
#   asyncio.run(main())
#   "
# -----------------------------------------------------------------------------

import itertools
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("Grievance Addressal System (GAS) — DEMO STUB")

# Not an exhaustive or authoritative list — just enough categories for a
# demo to look realistic and for routing/reporting to be meaningful at
# all. Anything the model sends that isn't in here is filed under "other"
# rather than rejected (see file_grievance() below) since a real system's
# own category list will differ from this stub's, and a hard rejection
# here would just be a demo annoyance, not a useful validation.
CATEGORIES = [
    "pm_kisan", "ration_card", "pension", "electricity", "water_supply",
    "road_infrastructure", "healthcare", "education", "other",
]

_lock = threading.Lock()
_grievances: dict = {}          # ticket_id -> record, in-memory only
_ticket_counter = itertools.count(1)


def _new_ticket_id() -> str:
    # GAS-<YYYYMM>-<5-digit sequence> — plausible-looking, NOT a real
    # numbering scheme from any actual government system.
    stamp = datetime.now(timezone.utc).strftime("%Y%m")
    with _lock:
        n = next(_ticket_counter)
    return f"GAS-{stamp}-{n:05d}"


@mcp.tool
def file_grievance(complainant_name: str, contact_number: str, category: str,
                    description: str, location: str = "") -> dict:
    """File a new grievance with the Grievance Addressal System.

    Required: complainant_name, contact_number, category, description.
    location is optional but helps with routing to the right local office.

    Returns a dict with the new ticket_id, status, and a human-readable
    confirmation message the caller (the LLM) can relay to the citizen —
    or {"success": False, "error": ...} if a required field was missing.
    """
    complainant_name = (complainant_name or "").strip()
    contact_number   = (contact_number or "").strip()
    description      = (description or "").strip()
    category         = (category or "").strip().lower() or "other"
    if category not in CATEGORIES:
        category = "other"

    if not complainant_name or not contact_number or not description:
        # A real system would likely reject this with an HTTP 400; this
        # stub instead returns a normal-looking error dict so the calling
        # LLM can read exactly what's missing and go back to asking the
        # user for it, rather than having to narrate a raised exception.
        missing = [n for n, v in [("complainant_name", complainant_name),
                                   ("contact_number", contact_number),
                                   ("description", description)] if not v]
        return {"success": False,
                "error": f"Missing required field(s): {', '.join(missing)}"}

    ticket_id = _new_ticket_id()
    record = {
        "ticket_id":         ticket_id,
        "complainant_name":  complainant_name,
        "contact_number":    contact_number,
        "category":          category,
        "description":       description,
        "location":          location.strip(),
        "status":            "received",
        "filed_at":          datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        _grievances[ticket_id] = record
    print(f"[GAS] filed {ticket_id} ({category}) for {complainant_name!r}: {description!r}")

    return {
        "success":   True,
        "ticket_id": ticket_id,
        "status":    "received",
        "message": (
            f"Grievance registered. Reference number {ticket_id}. "
            f"You will be contacted on {contact_number} about this "
            f"{category.replace('_', ' ')} issue."
        ),
    }


@mcp.tool
def check_grievance_status(ticket_id: str) -> dict:
    """Look up a previously filed grievance by its ticket_id.

    This stub always reports "received" and never actually progresses a
    ticket's status — there's no real backend behind it. Wire this up to
    a real status-lookup API alongside file_grievance() when one exists;
    nothing about the MCP tool signature needs to change to do that.
    """
    ticket_id = (ticket_id or "").strip()
    with _lock:
        record = _grievances.get(ticket_id)
    if record is None:
        return {"success": False,
                "error": f"No grievance found with ticket ID '{ticket_id}'."}
    return {
        "success":   True,
        "ticket_id": ticket_id,
        "status":    record["status"],
        "category":  record["category"],
        "filed_at":  record["filed_at"],
    }


# ── Registration — demonstrates the needs_input/continuation convention ──
#
# See tools/__init__.py's "Multi-step ('slot-filling') tools" section for
# the convention this (and authenticate_user() below) implements.
# Partially-collected registrations live ONLY in memory here, keyed by a
# random token this function generates itself and hands back inside
# "continuation" — nothing outside this file ever looks at that token or
# relies on its shape, so it's free to be whatever's convenient (a dict
# with a "field" key alongside it, here, so a resumed call knows WHICH
# field an incoming "answer" is actually for, without having to guess
# from current state).
_reg_lock = threading.Lock()   # separate from _lock above purely so a slow
                                 # registration flow can never contend with
                                 # grievance filing for no real reason
_registrations: dict = {}   # token -> {"full_name":…, "phone_number":…, "address":…}

_REG_FIELDS = [   # order matters: the order fields get asked for
    ("full_name", "your full name"),
    ("phone_number", "your phone number"),
    ("address", "your address"),
]


def _validate_phone(value: str) -> str:
    """Returns an error string if `value` isn't a plausible phone
    number, or "" if it's fine. A real GAS backend would validate far
    more carefully (actual numbering plans, maybe an OTP round-trip of
    its own — see authenticate_user() below) — this stub just checks for
    10 digits, enough to demonstrate the "reject and RE-ASK THE SAME
    FIELD" path rather than silently accepting garbage."""
    digits = "".join(ch for ch in (value or "") if ch.isdigit())
    if len(digits) != 10:
        return ("That doesn't sound like a valid 10-digit phone number — "
                "please say just the 10 digits.")
    return ""


@mcp.tool
def register_user(full_name: str = "", phone_number: str = "", address: str = "",
                   continuation: Optional[dict] = None, answer: str = "") -> dict:
    """Register a new citizen with GAS (separate from filing a
    grievance) — demonstrating the needs_input/continuation multi-step
    convention (see tools/__init__.py).

    FIRST call (continuation empty/missing): pass whatever of
    full_name/phone_number/address the conversation ALREADY established
    — even none of them. Whichever of the three is still missing gets
    asked for, one at a time, via needs_input.

    EVERY call AFTER that is a resume: pass back the EXACT `continuation`
    value this tool returned last time, plus `answer` = the user's raw
    reply to whatever it just asked. Never construct/guess a
    continuation value yourself — always echo it back verbatim.
    """
    if not continuation:
        token = uuid.uuid4().hex[:12]
        collected = {
            "full_name":    (full_name or "").strip(),
            "phone_number": (phone_number or "").strip(),
            "address":      (address or "").strip(),
        }
        with _reg_lock:
            _registrations[token] = collected
    else:
        token = continuation.get("token", "")
        field = continuation.get("field", "")
        with _reg_lock:
            collected = _registrations.get(token)
        if collected is None:
            # Unknown token (server restarted since, or a bogus value) —
            # rather than error out with nothing the citizen can do
            # about it, just start the flow over cleanly.
            return register_user()

        answer = (answer or "").strip()
        if field == "phone_number":
            err = _validate_phone(answer)
            if err:
                # Re-ask the SAME field with an error-explaining prompt,
                # rather than silently accepting an unusable number or
                # moving on regardless.
                return {"needs_input": True, "prompt": err,
                        "continuation": {"token": token, "field": "phone_number"}}
            collected["phone_number"] = "".join(ch for ch in answer if ch.isdigit())
        elif field:
            collected[field] = answer
        with _reg_lock:
            _registrations[token] = collected

    for field, label in _REG_FIELDS:
        if not collected.get(field):
            return {"needs_input": True, "prompt": f"What is {label}?",
                     "continuation": {"token": token, "field": field}}

    registration_id = f"GASREG-{uuid.uuid4().hex[:8].upper()}"
    print(f"[GAS] registered {registration_id}: {collected!r}")
    with _reg_lock:
        _registrations.pop(token, None)
    return {
        "success": True,
        "registration_id": registration_id,
        "message": (
            f"You're registered with GAS. Your registration number is "
            f"{registration_id}."
        ),
    }


# ── Authentication — the SAME convention, shaped as challenge/response ──
_auth_lock = threading.Lock()
_authentications: dict = {}   # token -> {"phone_number": str, "attempts": int}

# STUB ONLY — a real backend would generate a fresh one-time code per
# attempt and actually send it via an SMS gateway. This demo always
# expects the same fixed code and just prints it to this server's own
# console each time, so the full flow can be exercised with no real SMS
# provider wired up. Swap the two spots marked below for real
# generate+send / check calls when a real OTP provider exists — the tool
# signature itself doesn't need to change.
_DEMO_OTP = "1234"
_MAX_OTP_ATTEMPTS = 3


@mcp.tool
def authenticate_user(phone_number: str = "", continuation: Optional[dict] = None,
                       answer: str = "") -> dict:
    """Authenticate an existing citizen with GAS via a phone number plus
    a one-time code — the SAME needs_input/continuation convention as
    register_user() above, but shaped as a challenge/response instead of
    plain data collection.

    STUB BEHAVIOR: no real SMS is ever sent — the expected code is
    always "1234" (printed to this server's console each time it's
    "sent", purely so this flow is testable without a real SMS gateway).
    Up to 3 wrong codes before the flow gives up and has to be restarted.
    """
    if not continuation:
        token = uuid.uuid4().hex[:12]
        phone_number = (phone_number or "").strip()
        with _auth_lock:
            _authentications[token] = {"phone_number": phone_number, "attempts": 0}
        if not phone_number:
            return {"needs_input": True, "prompt": "What is your phone number?",
                     "continuation": {"token": token, "field": "phone_number"}}
        print(f"[GAS] (stub) sending OTP {_DEMO_OTP} to {phone_number} …")   # <- real SMS send goes here
        return {"needs_input": True,
                "prompt": "I've sent a one-time code to your phone. What is the code?",
                "continuation": {"token": token, "field": "otp_code"}}

    token = continuation.get("token", "")
    field = continuation.get("field", "")
    with _auth_lock:
        state = _authentications.get(token)
    if state is None:
        return authenticate_user()

    answer = (answer or "").strip()

    if field == "phone_number":
        if not answer:
            return {"needs_input": True, "prompt": "I didn't catch a phone number — what is it?",
                     "continuation": {"token": token, "field": "phone_number"}}
        print(f"[GAS] (stub) sending OTP {_DEMO_OTP} to {answer} …")   # <- real SMS send goes here
        with _auth_lock:
            state["phone_number"] = answer
        return {"needs_input": True,
                "prompt": "I've sent a one-time code to your phone. What is the code?",
                "continuation": {"token": token, "field": "otp_code"}}

    if field == "otp_code":
        if answer == _DEMO_OTP:   # <- real OTP check goes here
            with _auth_lock:
                _authentications.pop(token, None)
            print(f"[GAS] authenticated {state['phone_number']}")
            return {"success": True, "message": "You're verified."}

        with _auth_lock:
            state["attempts"] = state.get("attempts", 0) + 1
            attempts = state["attempts"]
        if attempts >= _MAX_OTP_ATTEMPTS:
            with _auth_lock:
                _authentications.pop(token, None)
            return {"success": False, "error": "Too many incorrect codes.",
                     "message": "That code didn't match, too many times — please start again."}
        return {"needs_input": True, "prompt": "That code didn't match — please say the code again.",
                 "continuation": {"token": token, "field": "otp_code"}}

    return authenticate_user()


if __name__ == "__main__":
    host = os.getenv("GAS_HOST", "0.0.0.0")
    port = int(os.getenv("GAS_PORT", "8787"))
    print(f"[GAS] starting DEMO STUB Grievance Addressal System on {host}:{port}…")
    mcp.run(transport="http", host=host, port=port)
