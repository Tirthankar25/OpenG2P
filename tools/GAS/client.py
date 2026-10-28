# tools/GAS/client.py — thin synchronous wrapper around an MCP CLIENT call
# to the (dummy, external) Grievance Addressal System server — see
# tools/GAS/server.py's module docstring for why GAS is reached over MCP
# instead of being imported directly, unlike tools/weather.
#
# Part of the tools/<name>/ convention — see tools/__init__.py for what
# every tool package exposes (TOOLS/call_tool/status_text/USAGE_NOTES)
# and how a new one gets registered in llama_llm.py.
#
# llama_llm.py's respond() runs on a plain background thread (see
# _chat_turn_worker() in js_asr_demo_server_ai4bh_xslate.py), not inside
# an asyncio event loop, but FastMCP's Client is async-only. Each call
# here opens a fresh connection via asyncio.run(), does the one call, and
# tears the connection back down — the simplest correct thing for a
# demo's request volume; a long-lived connection/event loop would be the
# first thing to add if this were ever under real concurrent load.
#
# Never raises: every public function here returns a dict with
# "success" — a connection failure, GAS_MCP_URL not being reachable at
# all, or the server rejecting the call (e.g. a validation error) all
# come back as {"success": False, "error": str}.
#
# Verified against the actual installed `fastmcp` package (4.0.10) in a
# throwaway sandbox: a tool that returns a plain dict (as both
# tools/GAS/server.py tools do) comes back through
# fastmcp.Client.call_tool() as a CallToolResult whose `.data` attribute
# IS that same dict — not guessed from memory.
#
# --- Prereqs (in THIS app's own venv, same one llama_llm.py runs in) -------
#   pip install fastmcp
#   # And tools/GAS/server.py itself must actually be running and
#   # reachable at GAS_MCP_URL (see tools/GAS/launch.sh) — it's a
#   # SEPARATE process, same pattern this app already uses for e.g. the
#   # old separate Sooktam2 TTS service (see tts_service.py).
# -----------------------------------------------------------------------------

import asyncio
import json
import os

# If GAS runs on a SEPARATE server reachable only via SSH (not a port
# opened directly to this app's host), leave this URL exactly as-is —
# don't change it to the remote host's address. Instead run
# tools/GAS/launch_tunnel.sh (see that file), which opens an SSH local
# port-forward so localhost:8787 on THIS host transparently reaches the
# GAS server on the other one. launch_all.sh already picks that script
# over the local tools/GAS/launch.sh automatically whenever GAS_SSH_HOST
# is set — see launch_all.sh's own comments.
GAS_MCP_URL = os.getenv("GAS_MCP_URL", "http://localhost:8787/mcp")

try:
    from fastmcp import Client as _MCPClient
    from fastmcp.exceptions import ToolError as _MCPToolError
    _FASTMCP_AVAILABLE = True
    _fastmcp_import_error = None
except Exception as e:
    # Mirrors tts_mms.py's "no torch at all" handling: don't let a missing
    # optional dependency take down the whole LLM pipeline at import time
    # — only calls that actually need GAS fail, with a clear reason, while
    # every other tool package keeps working.
    _FASTMCP_AVAILABLE = False
    _fastmcp_import_error = e
    _MCPClient = None
    _MCPToolError = Exception
    print(f"[GAS] warning: could not import fastmcp ({e}). The "
          f"file_grievance/check_grievance_status tools will report a "
          f"clear error until `pip install fastmcp` is run.")


def _call_tool_sync(tool_name: str, arguments: dict) -> dict:
    """Runs ONE MCP tool call against GAS_MCP_URL and returns its `.data`
    payload (the tool's own return dict). Any failure — the GAS server
    not running, a network error, the server rejecting the arguments — is
    caught here and turned into {"success": False, "error"} so callers
    never need their own try/except for this."""
    if not _FASTMCP_AVAILABLE:
        return {"success": False,
                "error": f"GAS client unavailable: fastmcp isn't installed ({_fastmcp_import_error})."}

    async def _run():
        async with _MCPClient(GAS_MCP_URL) as client:
            result = await client.call_tool(tool_name, arguments)
            return result.data

    try:
        data = asyncio.run(_run())
    except _MCPToolError as e:
        # The GAS server itself rejected the call (e.g. an argument
        # validation error on its side). Printed with the [GAS] prefix
        # like every other external-call failure in this app, so it's
        # visible in the console without a separate repro step.
        print(f"[GAS] tool call '{tool_name}' rejected by server: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"[GAS] could not reach GAS server at {GAS_MCP_URL} for '{tool_name}': {e!r}")
        return {"success": False,
                "error": f"The grievance system is unavailable right now ({e})."}

    if isinstance(data, dict):
        return data
    # tools/GAS/server.py's tools always return dicts; a non-dict here
    # would mean a server-side contract change — surface it plainly
    # rather than silently mishandling it.
    return {"success": True, "result": data}


def file_grievance(complainant_name: str, contact_number: str, category: str,
                    description: str, location: str = "") -> dict:
    return _call_tool_sync("file_grievance", {
        "complainant_name": complainant_name,
        "contact_number":   contact_number,
        "category":         category,
        "description":      description,
        "location":         location,
    })


def check_grievance_status(ticket_id: str) -> dict:
    return _call_tool_sync("check_grievance_status", {"ticket_id": ticket_id})


# register_user/authenticate_user carry TWO distinct call shapes (see
# tools/__init__.py's needs_input convention): the model's own FIRST
# call passes some subset of full_name/phone_number/address (or
# phone_number, for auth) and leaves continuation/answer empty;
# llama_llm.py's resume_followup() — NOT the model — makes every call
# after that, passing continuation/answer and leaving the other fields
# empty. Both shapes land here as plain kwargs either way.
def register_user(full_name: str = "", phone_number: str = "", address: str = "",
                   continuation: dict = None, answer: str = "") -> dict:
    return _call_tool_sync("register_user", {
        "full_name":    full_name,
        "phone_number": phone_number,
        "address":      address,
        "continuation": continuation,
        "answer":       answer,
    })


def authenticate_user(phone_number: str = "", continuation: dict = None, answer: str = "") -> dict:
    return _call_tool_sync("authenticate_user", {
        "phone_number": phone_number,
        "continuation": continuation,
        "answer":       answer,
    })


# --- Registration contract — see tools/__init__.py --------------------------
#
# Tool schemas, Ollama's documented function-calling format. Verified
# against Ollama's own docs (github.com/ollama/ollama docs/api.md,
# registry.ollama.ai/blog/tool-support) rather than assumed — see
# llama_llm.py's module docstring for the same note, since this is the
# same shape every package's TOOLS entries must follow.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_grievance",
            "description": (
                "File a new grievance with the Grievance Addressal System "
                "(GAS) on the user's behalf — e.g. a complaint about a "
                "government service, a scheme benefit not received, or a "
                "civic issue (water, electricity, roads, etc.). Only call "
                "this once you have ALL required fields from the "
                "conversation. If any are missing, ASK the user for them "
                "first — never guess a name, phone number, or category, "
                "and never call this tool with a made-up or placeholder "
                "value just to fill a required field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "complainant_name": {
                        "type": "string",
                        "description": "Full name of the person filing the grievance.",
                    },
                    "contact_number": {
                        "type": "string",
                        "description": "A phone number GAS can use to follow up, digits only.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Which department or scheme the grievance is about.",
                        # Mirrors tools/GAS/server.py's own CATEGORIES list.
                        # Not imported from there on purpose — GAS is
                        # modeled as an external system reached only via
                        # this file's MCP call, so this app has no
                        # business importing the GAS server's own module.
                        # Keep these two lists in sync by hand; a category
                        # the model sends that ISN'T in this list still
                        # works fine (server.py files it under "other"
                        # rather than rejecting it), so drift here is a
                        # UX/accuracy issue, not a breakage.
                        "enum": [
                            "pm_kisan", "ration_card", "pension", "electricity",
                            "water_supply", "road_infrastructure", "healthcare",
                            "education", "other",
                        ],
                    },
                    "description": {
                        "type": "string",
                        "description": "What went wrong, in the user's own words.",
                    },
                    "location": {
                        "type": "string",
                        "description": "City, town, or district the grievance relates to, if known.",
                    },
                },
                "required": ["complainant_name", "contact_number", "category", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_grievance_status",
            "description": (
                "Look up the status of a grievance the user already filed, "
                "given its ticket/reference number (looks like "
                "'GAS-202610-00007'). Only call this if the user actually "
                "gives you a ticket number — if they don't have one, ask "
                "for it rather than guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The grievance's reference number, e.g. 'GAS-202610-00007'.",
                    },
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_user",
            "description": (
                "Register the user as a new citizen with GAS — separate "
                "from filing a grievance. Call this when the user asks "
                "to register, sign up, or create an account with GAS. "
                "Pass whatever of their full name, phone number, or "
                "address they've ALREADY given you in the conversation "
                "— even none of it. The tool itself will ask for "
                "whatever's still missing, one thing at a time, over "
                "the next few turns — never guess or invent a value for "
                "any of these fields yourself."
            ),
            # Deliberately only full_name/phone_number/address here —
            # "continuation"/"answer" (see tools/__init__.py) are NEVER
            # set by the model. llama_llm.py's resume_followup() sets
            # those directly, bypassing Ollama's tool-calling entirely
            # for every call after the first.
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {
                        "type": "string",
                        "description": "The user's full name, if already known from the conversation.",
                    },
                    "phone_number": {
                        "type": "string",
                        "description": "The user's phone number, if already known from the conversation.",
                    },
                    "address": {
                        "type": "string",
                        "description": "The user's address, if already known from the conversation.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "authenticate_user",
            "description": (
                "Verify the user's identity with GAS via a phone number "
                "and a one-time code sent to it. Call this when the "
                "user asks to log in, sign in, or verify/authenticate "
                "themselves with GAS. Pass their phone number if "
                "they've already given it in the conversation, "
                "otherwise leave it out — the tool will ask for it, "
                "then ask for the code it sends. Call this once per "
                "request; the tool itself handles the rest of the "
                "back-and-forth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "The user's phone number, if already known from the conversation.",
                    },
                },
            },
        },
    },
]


def call_tool(name: str, args: dict) -> str:
    if name == "file_grievance":
        result = file_grievance(
            complainant_name=args.get("complainant_name", ""),
            contact_number=args.get("contact_number", ""),
            category=args.get("category", "other"),
            description=args.get("description", ""),
            location=args.get("location", ""),
        )
        return json.dumps(result)

    elif name == "check_grievance_status":
        result = check_grievance_status(args.get("ticket_id", ""))
        return json.dumps(result)

    elif name == "register_user":
        result = register_user(
            full_name=args.get("full_name", ""),
            phone_number=args.get("phone_number", ""),
            address=args.get("address", ""),
            continuation=args.get("continuation"),
            answer=args.get("answer", ""),
        )
        return json.dumps(result)

    elif name == "authenticate_user":
        result = authenticate_user(
            phone_number=args.get("phone_number", ""),
            continuation=args.get("continuation"),
            answer=args.get("answer", ""),
        )
        return json.dumps(result)

    return f"Error: tools/GAS does not provide a tool named '{name}'."


def status_text(name: str, args: dict) -> str:
    if name == "file_grievance":
        return "Filing your grievance…"
    if name == "check_grievance_status":
        ticket = args.get("ticket_id") or "your grievance"
        return f"Checking status of {ticket}…"
    if name == "register_user":
        return "Registering you with GAS…"
    if name == "authenticate_user":
        return "Verifying your identity…"
    return f"Calling {name}…"


USAGE_NOTES = (
    "file_grievance: call it ONLY once you have, from the conversation, "
    "the complainant's name, a contact phone number, a category for the "
    "issue, and a description of what's wrong — if any of these is "
    "missing, ask the user for it in plain conversation first; do not "
    "call file_grievance with a blank, guessed, or placeholder value for "
    "a required field. After a successful file_grievance call, tell the "
    "user the exact reference number the tool returned — never invent a "
    "ticket/reference number yourself under any circumstances.\n"
    "check_grievance_status: call it when the user gives you a reference "
    "number and asks about an existing grievance; if they don't have a "
    "reference number, ask for it rather than guessing.\n"
    "register_user: call it when the user wants to register/sign up "
    "with GAS — unlike file_grievance, you do NOT need to collect every "
    "field yourself first; pass whatever of full_name/phone_number/"
    "address is already in the conversation (or none of it) and the "
    "tool will ask for the rest itself, one field at a time, across the "
    "next few turns.\n"
    "authenticate_user: call it when the user wants to log in/verify "
    "themselves with GAS — pass a phone number if already known, "
    "otherwise leave it out; call it once per request and let the tool "
    "handle asking for the phone number and the one-time code itself."
)
