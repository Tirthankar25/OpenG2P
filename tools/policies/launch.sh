#!/usr/bin/env bash
# tools/policies/launch.sh — starts the fetch-only Policies MCP server
# (PM-KISAN, SMAM, ... — see data/*.json) as its OWN process, separate
# from the main app. Run alongside tools/GAS/launch.sh and the main
# app's own launch script.
#
# Must be running and reachable at POLICIES_MCP_URL (default
# http://localhost:8788/mcp — see client.py) BEFORE get_policy_info /
# list_available_policies tool calls will succeed.
#
# Prereq (in whichever venv runs this): pip install fastmcp

echo "[launch] starting Policies MCP server on port 8788…"
python3 "$(dirname "$0")/server.py"
