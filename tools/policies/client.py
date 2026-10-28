# tools/policies/client.py — thin synchronous wrapper around an MCP
# CLIENT call to tools/policies/server.py — same pattern as
# tools/GAS/client.py (see that file's docstring for the fuller
# reasoning: FastMCP's Client is async-only, llama_llm.py's respond()
# runs on a plain background thread, so each call here opens a fresh
# connection via asyncio.run(), does the one call, and tears it down).
#
# Part of the tools/<name>/ convention — see tools/__init__.py.
#
# Never raises: every public function here returns a dict with
# "success" — a connection failure, POLICIES_MCP_URL not being reachable,
# or anything else all come back as {"success": False, "error": str}.
#
# --- Prereqs (in THIS app's own venv, same one llama_llm.py runs in) -------
#   pip install fastmcp
#   # And tools/policies/server.py itself must be running and reachable
#   # at POLICIES_MCP_URL (see tools/policies/launch.sh).
# -----------------------------------------------------------------------------

import asyncio
import json
import os

POLICIES_MCP_URL = os.getenv("POLICIES_MCP_URL", "http://localhost:8788/mcp")

try:
    from fastmcp import Client as _MCPClient
    from fastmcp.exceptions import ToolError as _MCPToolError
    _FASTMCP_AVAILABLE = True
    _fastmcp_import_error = None
except Exception as e:
    _FASTMCP_AVAILABLE = False
    _fastmcp_import_error = e
    _MCPClient = None
    _MCPToolError = Exception
    print(f"[POLICIES] warning: could not import fastmcp ({e}). The "
          f"get_policy_info/list_available_policies tools will report a "
          f"clear error until `pip install fastmcp` is run.")


def _call_tool_sync(tool_name: str, arguments: dict) -> dict:
    if not _FASTMCP_AVAILABLE:
        return {"success": False,
                "error": f"Policies client unavailable: fastmcp isn't installed ({_fastmcp_import_error})."}

    async def _run():
        async with _MCPClient(POLICIES_MCP_URL) as client:
            result = await client.call_tool(tool_name, arguments)
            return result.data

    try:
        data = asyncio.run(_run())
    except _MCPToolError as e:
        print(f"[POLICIES] tool call '{tool_name}' rejected by server: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"[POLICIES] could not reach Policies server at {POLICIES_MCP_URL} for '{tool_name}': {e!r}")
        return {"success": False,
                "error": f"The policies lookup is unavailable right now ({e})."}

    if isinstance(data, dict):
        return data
    return {"success": True, "result": data}


def get_policy_info(scheme: str) -> dict:
    return _call_tool_sync("get_policy_info", {"scheme": scheme})


def list_available_policies() -> dict:
    return _call_tool_sync("list_available_policies", {})


# --- Registration contract — see tools/__init__.py --------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_policy_info",
            "description": (
                "Look up information about a NAMED government scheme or "
                "subsidy program (e.g. PM-KISAN, SMAM) — eligibility, "
                "benefits, and how to apply. Call this ONLY when the "
                "user names a specific scheme or clearly asks about a "
                "government benefit/subsidy program, instead of "
                "answering from your own general knowledge, which may be "
                "outdated. Do NOT call this for anything else — travel, "
                "directions, weather, general knowledge, small talk, or "
                "any question that doesn't name or clearly imply a "
                "specific scheme. If the scheme isn't found, the result "
                "lists what IS available — offer the user that list "
                "rather than guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scheme": {
                        "type": "string",
                        "description": "The scheme name as the user said it, e.g. 'PM-KISAN' or 'SMAM'.",
                    },
                },
                "required": ["scheme"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_policies",
            "description": (
                "List every government scheme you currently have "
                "information for. Call this ONLY if the user explicitly "
                "asks what schemes/benefits you can help with, or "
                "immediately after get_policy_info reports a NAMED "
                "scheme wasn't found. Do NOT call this speculatively, "
                "and never for a question unrelated to government "
                "schemes."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def call_tool(name: str, args: dict) -> str:
    if name == "get_policy_info":
        return json.dumps(get_policy_info(args.get("scheme", "")))
    elif name == "list_available_policies":
        return json.dumps(list_available_policies())
    return f"Error: tools/policies does not provide a tool named '{name}'."


def status_text(name: str, args: dict) -> str:
    if name == "get_policy_info":
        return f"Looking up {args.get('scheme') or 'that scheme'}…"
    if name == "list_available_policies":
        return "Checking which schemes I can help with…"
    return f"Calling {name}…"


USAGE_NOTES = (
    "get_policy_info: call it ONLY when the user names a specific "
    "government scheme (e.g. PM-KISAN, SMAM) or clearly asks about a "
    "government benefit/subsidy program, to get accurate, current "
    "details instead of relying on your own possibly-outdated knowledge. "
    "Never call it for an unrelated question (directions, travel time, "
    "weather, general knowledge, etc.) just because it's unfamiliar or "
    "you're unsure how to answer. list_available_policies: call it only "
    "if the user explicitly asks what schemes you can help with, or "
    "immediately after get_policy_info reports a named scheme isn't "
    "found — never speculatively."
)
