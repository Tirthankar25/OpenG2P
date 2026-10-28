#!/usr/bin/env bash
# tools/GAS/launch_tunnel.sh — opens an SSH LOCAL port-forward so a GAS
# MCP server running on a SEPARATE remote machine becomes reachable at
# the exact same http://localhost:$GAS_LOCAL_PORT/mcp address
# tools/GAS/client.py already uses — so client.py / GAS_MCP_URL need
# ZERO code changes, whether GAS runs on this host or a remote one.
#
# Why a tunnel, instead of pointing GAS_MCP_URL straight at the remote
# host's own address (e.g. http://xxx.yy.zzz.ppp:8885/mcp): that only
# works if port 8885 is itself opened to this app's host in the GAS
# server's firewall. If the only access you have is SSH (as here — "ssh
# -p 32844 myname@xxx.yy.zzz.ppp"), that port very likely ISN'T open to
# the outside at all, only port 32844 (SSH) is. An SSH local
# port-forward (-L) rides over that SAME SSH connection: ssh connects out
# on port 32844 exactly like you would by hand, authenticates the same
# way, and then relays local 127.0.0.1:$GAS_LOCAL_PORT <-> (from the
# REMOTE box's OWN point of view) localhost:$GAS_REMOTE_PORT, which is
# where that box's GAS server is actually listening. Nothing new needs
# opening on that server's firewall.
#
# --- Required one-time setup (on THIS app's host) --------------------------
# This runs unattended, in the background, with no one watching — it
# CANNOT type a password or passphrase for you. Set up key-based auth
# FIRST, and confirm it actually works non-interactively, before relying
# on this script:
#   ssh-keygen -t ed25519                        # if you don't already have a key
#   ssh-copy-id -p 32844 myname@xxx.yy.zzz.ppp    # installs it on the GAS host
#   ssh -p 32844 myname@xxx.yy.zzz.ppp true       # must connect with NO prompt now
# If that last line still asks for anything, fix it before using this
# script — it will otherwise just hang silently, retrying forever,
# waiting for input that never arrives (see the retry loop below).
#
# --- Config (environment variables — see launch_all.sh) --------------------
#   GAS_SSH_USER     e.g. myname                      (required)
#   GAS_SSH_HOST     e.g. xxx.yy.zzz.ppp               (required)
#   GAS_SSH_PORT     the SSH port, default 32844
#   GAS_REMOTE_PORT  the port GAS listens on, ON THE REMOTE HOST, default 8885
#   GAS_LOCAL_PORT   the local port to forward it to, default 8787
#                    (must match the port in GAS_MCP_URL / client.py's
#                    default — they're 8787 on both sides so nothing
#                    else needs to change)
#
# Run directly to test: bash tools/GAS/launch_tunnel.sh
# (Ctrl+C stops it. launch_all.sh runs this the same way it runs any of
# the other tool launchers — backgrounded, log redirected, cleaned up on
# exit — see that script.)

set -u

GAS_SSH_USER="${GAS_SSH_USER:?set GAS_SSH_USER, e.g. export GAS_SSH_USER=myname}"
GAS_SSH_HOST="${GAS_SSH_HOST:?set GAS_SSH_HOST, e.g. export GAS_SSH_HOST=xxx.yy.zzz.ppp}"
GAS_SSH_PORT="${GAS_SSH_PORT:-32844}"
GAS_REMOTE_PORT="${GAS_REMOTE_PORT:-8885}"
GAS_LOCAL_PORT="${GAS_LOCAL_PORT:-8787}"

echo "[GAS-tunnel] forwarding localhost:${GAS_LOCAL_PORT} -> (ssh -p ${GAS_SSH_PORT} ${GAS_SSH_USER}@${GAS_SSH_HOST}) -> its own localhost:${GAS_REMOTE_PORT} …"

# -N: no remote command, just forward the port.
# -T: no pseudo-terminal — this is non-interactive, nothing will ever
#     read/write a terminal on the remote end.
# ExitOnForwardFailure=yes: fail FAST (instead of sitting there
#     connected-but-useless) if $GAS_LOCAL_PORT is already taken on this
#     host — the retry loop below will keep retrying either way, but a
#     fast failure shows up in the log immediately instead of looking
#     like a hang.
# ServerAlive*: detect a silently-dead connection (network blip, remote
#     host rebooting, etc.) within ~45s and exit, rather than hanging
#     forever — so the retry loop actually gets a chance to reconnect
#     instead of GAS looking "up" while quietly not working.
#
# The outer `while true` IS the retry loop: ssh exiting for ANY reason
# just gets retried after a short pause, rather than permanently losing
# GAS connectivity until someone notices this process died and restarts
# it by hand.
while true; do
    ssh -p "${GAS_SSH_PORT}" -N -T \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -L "${GAS_LOCAL_PORT}:localhost:${GAS_REMOTE_PORT}" \
        "${GAS_SSH_USER}@${GAS_SSH_HOST}"
    status=$?
    echo "[GAS-tunnel] ssh exited (status ${status}) — reconnecting in 3s…" >&2
    sleep 3
done
