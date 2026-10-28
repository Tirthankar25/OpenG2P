#!/usr/bin/env bash
# tools/GAS/launch.sh — starts the DUMMY Grievance Addressal System (GAS)
# MCP server as its OWN process, separate from the main app.
#
# Why separate: GAS is modeled as an EXTERNAL system the main app reaches
# over the network via an MCP client (client.py), not something it
# imports directly — see server.py's module docstring. Must be running
# and reachable at GAS_MCP_URL (default http://localhost:8787/mcp — see
# client.py) BEFORE the main app's file_grievance/check_grievance_status
# tool calls will succeed. If it's not running, those calls fail cleanly
# with a "grievance system is unavailable" message rather than crashing
# the main app or that turn.
#
# Run from the repo root (so `python3 tools/GAS/server.py` resolves) —
# or `cd` into this directory first; either works since server.py has no
# relative imports of its own.
#
# Prereq (in whichever venv runs this): pip install fastmcp

echo "[launch] starting GAS MCP server on port 8787…"
python3 "$(dirname "$0")/server.py"
