# tools/ — self-contained tool packages this app's agentic LLM
# (llama_llm.py) can call. Each subfolder is ONE tool package. To add a
# new tool later: create tools/<name>/ following the convention below,
# then add ONE line to llama_llm.py's _TOOL_PACKAGES list. Nothing else in
# this app needs to know a new tool exists — not the system prompt text,
# not js_asr_demo_server_ai4bh_xslate.py's status-line handling, nothing.
#
# --- What every package must expose, from its own client.py ----------------
#
#   TOOLS : list[dict]
#       One Ollama tool-schema entry per tool this package provides — the
#       {"type": "function", "function": {"name", "description",
#       "parameters": {...}}} shape Ollama's /api/chat "tools" field
#       expects (see llama_llm.py's module docstring for where that shape
#       was verified). A package can provide more than one tool (GAS
#       provides file_grievance AND check_grievance_status).
#
#   call_tool(name: str, args: dict) -> str
#       Executes ONE of this package's own tool names (never a name
#       belonging to a different package — llama_llm.py's dispatcher
#       routes by name before calling this) and returns the result as a
#       STRING, since that's what Ollama's "tool" role message expects as
#       `content` — json.dumps(...) a dict result (see
#       tools/GAS/client.py for the pattern). Should never raise: catch
#       everything internally and return a clear error string instead.
#       llama_llm.py's dispatcher also catches, as a second line of
#       defence, but don't rely on that.
#
# --- Optional, but recommended -----------------------------------------
#
#   status_text(name: str, args: dict) -> str
#       A short, human-readable status line to show the user the INSTANT
#       this tool is about to be called — e.g. "Checking the weather in
#       Patna…", "Filing your grievance…" (see
#       js_asr_demo_server_ai4bh_xslate.py's _on_tool_call, and
#       llama_llm.py's status_text_for(), which calls this with a generic
#       fallback if a package omits it).
#
#   USAGE_NOTES : str
#       Plain-language guidance for the model about WHEN/HOW to use this
#       package's tool(s) — folded automatically into the system prompt
#       (see llama_llm.py's _tool_usage_instructions()). This is where
#       tool-specific judgment calls belong (e.g. GAS's "never guess a
#       required field") — the tool's own `description` field in TOOLS
#       already explains what it does; USAGE_NOTES is for the sharper
#       behavioral rules around calling it.
#
# --- Two shapes of package, both following the SAME convention above ------
#
# A package whose tool reaches something genuinely EXTERNAL (GAS,
# Policies) additionally has server.py (a FastMCP server — the thing
# actually being "reached") and launch.sh (starts that server as its own
# process, separate from the main app); client.py then talks to it over
# MCP rather than importing it directly — see tools/GAS/ for the
# reference shape, including the "why MCP, why a separate process" reasoning
# in tools/GAS/server.py's docstring.
#
# A package with no real external system behind it — weather is just a
# direct call to a public weather API, nothing this app could meaningfully
# run "as a service" on its own behalf — only needs client.py, with the
# real implementation living right there. See tools/weather/client.py.
#
# Either way, from llama_llm.py's point of view every package looks
# identical: TOOLS, call_tool(), optionally status_text()/USAGE_NOTES.
# Whether a given tool happens to be in-process or over MCP is an
# implementation detail of that one package, invisible to everything else.
