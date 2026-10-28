#!/usr/bin/env bash
# launch_all.sh — starts GAS, Policies, and the main app together as ONE
# command, instead of three separate shells. Equivalent to running
# tools/GAS/launch.sh, tools/policies/launch.sh, and
# launch_converse_index_asr_ai4bh_xslate.sh yourself in three terminals —
# the two tool servers run in the BACKGROUND (logged to files, see below)
# and the main app runs in the FOREGROUND, same as it would in its own
# shell: you see its output directly here, and Ctrl+C stops everything.
#
# Run from the app's root directory (same place you'd run the other
# launch scripts from): ./launch_all.sh
# (If it won't execute directly, `chmod +x launch_all.sh` once, or just
# run `bash launch_all.sh`.)
#
# Logs for the two background servers land in ./logs/ (created if it
# doesn't exist) — tail them if something's not responding:
#   tail -f logs/gas.log
#   tail -f logs/policies.log
# The main app's own output is NOT redirected — it prints straight to
# this terminal, exactly like running its launch script directly.
#
# Ctrl+C (or closing this shell) runs the cleanup trap below, which stops
# BOTH background servers too — without this, killing just the main
# app's terminal would leave GAS/Policies running and still holding ports
# 8787/8788, so the next launch_all.sh would fail to bind them.

set -u

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

LOG_DIR="$APP_DIR/logs"
mkdir -p "$LOG_DIR"
GAS_LOG="$LOG_DIR/gas.log"
POLICIES_LOG="$LOG_DIR/policies.log"

# GAS can run either locally (tools/GAS/server.py, the default) or on a
# separate remote host reached only via SSH (tools/GAS/launch_tunnel.sh)
# — set GAS_SSH_HOST (and GAS_SSH_USER) in the environment to switch to
# the latter; see tools/GAS/launch_tunnel.sh's own docstring for the
# full setup (key-based SSH auth especially — it must connect with NO
# prompt, since nothing here can type a password for it).
if [ -n "${GAS_SSH_HOST:-}" ]; then
    echo "[launch_all] GAS_SSH_HOST is set — tunneling to the remote GAS server at $GAS_SSH_HOST (log: $GAS_LOG)…"
    bash tools/GAS/launch_tunnel.sh > "$GAS_LOG" 2>&1 &
else
    echo "[launch_all] starting GAS MCP server locally (log: $GAS_LOG)…"
    bash tools/GAS/launch.sh > "$GAS_LOG" 2>&1 &
fi
GAS_PID=$!

echo "[launch_all] starting Policies MCP server (log: $POLICIES_LOG)…"
bash tools/policies/launch.sh > "$POLICIES_LOG" 2>&1 &
POLICIES_PID=$!

# Not a strict dependency — the main app's tool calls fail cleanly and
# recover on the NEXT call if GAS/Policies aren't up yet (see
# tools/GAS/client.py / tools/policies/client.py's "service unavailable"
# handling) — this just avoids a guaranteed-fail first tool call while
# they're still loading their models/data.
sleep 2

_CLEANED_UP=0
cleanup() {
    # Trapping both EXIT and INT/TERM means a signal normally runs this
    # TWICE (once for the signal's own trap, then again for the EXIT
    # trap as the script actually exits afterward) — this guard makes it
    # a no-op the second time rather than printing/killing twice.
    if [ "$_CLEANED_UP" = "1" ]; then
        return
    fi
    _CLEANED_UP=1
    echo
    echo "[launch_all] stopping GAS, Policies, and the main app…"
    # Kill both the wrapper launch.sh/launch script PIDs AND a
    # pattern-match on the actual underlying process (server.py /
    # js_asr_demo_server_ai4bh_xslate) — killing just the wrapper's PID
    # can leave its own child (the real python3/uvicorn process) running
    # orphaned, still holding the port, depending on exactly how the
    # shell delivers the signal; pkill -f catches that regardless of how
    # deep the wrapper nesting is.
    kill "$GAS_PID" "$POLICIES_PID" "$MAIN_PID" 2>/dev/null
    pkill -f "tools/GAS/server.py" 2>/dev/null
    # Covers the remote-GAS case too: killing $GAS_PID only signals the
    # `while true; do ssh …; done` wrapper (tools/GAS/launch_tunnel.sh)
    # itself — its ssh child can otherwise survive that, same orphan risk
    # as every other wrapped process here. Matches on the actual -L
    # forwarding spec so it's specific to the tunnel THIS app started,
    # using the same defaults tools/GAS/launch_tunnel.sh falls back to if
    # you didn't override GAS_LOCAL_PORT/GAS_REMOTE_PORT yourself.
    pkill -f "ssh.*-L ${GAS_LOCAL_PORT:-8787}:localhost:${GAS_REMOTE_PORT:-8885}" 2>/dev/null
    pkill -f "tools/policies/server.py" 2>/dev/null
    pkill -f "js_asr_demo_server_ai4bh_xslate" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[launch_all] starting main app…"
# Backgrounded (with output still going straight to this terminal, same
# as running it directly — nothing is redirected) rather than run as a
# normal foreground command, specifically so Ctrl+C is interruptible: a
# script blocked on a synchronous foreground command does NOT run its
# trap until that command finishes (documented bash behavior) — but a
# script blocked on `wait` for a backgrounded job IS interrupted
# immediately when a trapped signal arrives, which is the whole point of
# this script. Verified by actually testing both versions, not assumed —
# the first draft used a direct foreground call and Ctrl+C silently did
# nothing until the main app exited on its own.
bash launch_converse_index_asr_ai4bh_xslate.sh &
MAIN_PID=$!
wait "$MAIN_PID"
