# tools/policies/server.py — FETCH-ONLY "Policies" MCP server: scheme
# information (PM-KISAN, SMAM, ...) for the LLM to look up instead of
# relying on its own (possibly outdated or hallucinated) knowledge of
# government schemes.
#
# Run as its own MCP server/process, same shape as tools/GAS, even though
# this one has no real external system behind it — the data is just JSON
# files loaded at startup. That's a deliberate choice (confirmed with the
# person this was built for) for UNIFORMITY: every tool this app adds
# talks to llama_llm.py the exact same way (an MCP client calling a
# FastMCP server), so adding a tool is always the same shape of work
# regardless of whether it happens to need real external infrastructure
# yet. tools/weather is the one exception (no server.py at all) because
# it already existed in-process before this convention was introduced.
#
# Part of the tools/<name>/ convention — see tools/__init__.py.
#
# --- Adding a new scheme later ---------------------------------------------
# Drop a new <scheme_id>.json file into tools/policies/data/ (see
# pm_kisan.json / smam.json for the shape: id, name, aliases, summary,
# eligibility, benefits, how_to_apply, source_note, last_updated) and
# restart this server — _load_policies() picks up every *.json file in
# that directory, no code change needed. list_available_policies() and
# get_policy_info()'s "did you mean" suggestions both update automatically.
#
# --- Prereqs ----------------------------------------------------------------
#   pip install fastmcp
#
# --- Running ----------------------------------------------------------------
#   python3 tools/policies/server.py     # or: tools/policies/launch.sh
#   # Serves MCP over streamable-HTTP at http://<host>:8788/mcp — this is
#   # what POLICIES_MCP_URL (see client.py) must point at.
# -----------------------------------------------------------------------------

import glob
import json
import os

from fastmcp import FastMCP

mcp = FastMCP("Policies (scheme information) — fetch-only")

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_policies() -> dict:
    """Loads every *.json file in data/ into {scheme_id: record}. A
    malformed individual file is skipped with a printed warning rather
    than taking down the whole server — one bad JSON file shouldn't make
    every OTHER scheme unavailable too."""
    policies = {}
    for path in sorted(glob.glob(os.path.join(_DATA_DIR, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            scheme_id = record.get("id") or os.path.splitext(os.path.basename(path))[0]
            policies[scheme_id] = record
        except Exception as e:
            print(f"[POLICIES] warning: could not load {path}: {e!r}")
    print(f"[POLICIES] loaded {len(policies)} scheme(s): {', '.join(sorted(policies)) or '(none)'}")
    return policies


# Loaded ONCE at process start — this is fetch-only, read-only reference
# data, not something that changes while the server is running. Restart
# the process to pick up a newly-added/edited data/*.json file.
_POLICIES = _load_policies()


def _match(scheme: str) -> str:
    """Resolves free-text `scheme` to a known scheme_id via exact id
    match, then exact name match, then alias match, then a loose
    substring match against id/name/aliases — all case-insensitive.
    Returns "" if nothing matches at all."""
    needle = (scheme or "").strip().lower()
    if not needle:
        return ""
    if needle in _POLICIES:
        return needle
    for scheme_id, record in _POLICIES.items():
        if needle == (record.get("name") or "").lower():
            return scheme_id
        if needle in [a.lower() for a in (record.get("aliases") or [])]:
            return scheme_id
    for scheme_id, record in _POLICIES.items():
        haystack = " ".join([
            scheme_id, record.get("name", ""), *(record.get("aliases") or [])
        ]).lower()
        if needle in haystack or any(word in haystack for word in needle.split()):
            return scheme_id
    return ""


@mcp.tool
def get_policy_info(scheme: str) -> dict:
    """Look up information about one government scheme by name (e.g.
    "PM-KISAN", "pm kisan", "SMAM", "farm machinery subsidy" all resolve).

    Returns the scheme's full record on a match, or {"success": False,
    "error": ..., "available": [...]} listing known scheme names if
    nothing matches — so the caller (the LLM) can offer the user the
    actual list instead of just saying "not found".
    """
    scheme_id = _match(scheme)
    if not scheme_id:
        return {
            "success": False,
            "error": f"No policy information found for '{scheme}'.",
            "available": [r.get("name", sid) for sid, r in sorted(_POLICIES.items())],
        }
    record = dict(_POLICIES[scheme_id])
    record["success"] = True
    return record


@mcp.tool
def list_available_policies() -> dict:
    """List every scheme this server currently has information for —
    call this if the user asks what schemes you can help with, or as a
    fallback after get_policy_info() reports no match."""
    return {
        "success": True,
        "schemes": [
            {"id": sid, "name": r.get("name", sid)}
            for sid, r in sorted(_POLICIES.items())
        ],
    }


if __name__ == "__main__":
    host = os.getenv("POLICIES_HOST", "0.0.0.0")
    port = int(os.getenv("POLICIES_PORT", "8788"))
    print(f"[POLICIES] starting fetch-only Policies MCP server on {host}:{port}…")
    mcp.run(transport="http", host=host, port=port)
