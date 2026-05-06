"""LangChain @tool wrappers for LightClaw's 8 built-in tools.

Each wrapper delegates to the original async implementation which returns
``LightClawToolResult``. Text-only results stay as plain strings; structured
media results are serialized to JSON so downstream renderers can preserve
images/files instead of collapsing them into text.

The original functions are NOT modified — this is a pure adapter layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.tools import tool


def _serialize_tool_result(resp) -> str:
    """Return plain text for text-only tool results, JSON for media results."""
    content = list(getattr(resp, "content", []) or [])
    if not content:
        return getattr(resp, "to_str", lambda: "")()

    has_non_text = any(isinstance(block, dict) and block.get("type") != "text" for block in content)
    if has_non_text:
        return json.dumps(content, ensure_ascii=False)

    return getattr(resp, "to_str", lambda: "")()


# ---------------------------------------------------------------------------
# 1. execute_shell_command
# ---------------------------------------------------------------------------
@tool
async def execute_shell_command(
    command: str,
    timeout: int = 60,
    cwd: str | None = None,
) -> str:
    """Execute a shell command and return stdout/stderr.

    Args:
        command: The shell command to execute.
        timeout: Maximum time in seconds (default 60).
        cwd: Working directory (default ~/.lightclaw/workspace).
    """
    from lightclaw.agent.tools.shell import execute_shell_command as _impl

    cwd_path = Path(cwd) if cwd else None
    resp = await _impl(command=command, timeout=timeout, cwd=cwd_path)
    return _serialize_tool_result(resp)


# ---------------------------------------------------------------------------
# 2. read_file
# ---------------------------------------------------------------------------
@tool
async def read_file(
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read a file. Relative paths resolve from ~/.lightclaw/workspace.

    Use start_line/end_line (1-based) to read a specific range,
    or omit both to read the full file.

    Args:
        file_path: Path to the file.
        start_line: First line to read (1-based, inclusive).
        end_line: Last line to read (1-based, inclusive).
    """
    from lightclaw.agent.tools.file_io import read_file as _impl

    resp = await _impl(file_path=file_path, start_line=start_line, end_line=end_line)
    return resp.to_str()


# ---------------------------------------------------------------------------
# 3. write_file
# ---------------------------------------------------------------------------
@tool
async def write_file(
    file_path: str,
    content: str,
) -> str:
    """Create or overwrite a file. Relative paths resolve from ~/.lightclaw/workspace.

    Args:
        file_path: Path to the file.
        content: Content to write.
    """
    from lightclaw.agent.tools.file_io import write_file as _impl

    resp = await _impl(file_path=file_path, content=content)
    return resp.to_str()


# ---------------------------------------------------------------------------
# 4. append_file
# ---------------------------------------------------------------------------
@tool
async def append_file(
    file_path: str,
    content: str,
) -> str:
    """Append content to a file without overwriting existing data.

    Use this instead of write_file when adding to daily memory notes
    (memory/YYYY-MM-DD.md) or any file where existing content must be preserved.
    Relative paths resolve from ~/.lightclaw/workspace.

    Args:
        file_path: Path to the file.
        content: Content to append.
    """
    from lightclaw.agent.tools.file_io import append_file as _impl

    resp = await _impl(file_path=file_path, content=content)
    return resp.to_str()


# ---------------------------------------------------------------------------
# 5. edit_file
# ---------------------------------------------------------------------------
@tool
async def edit_file(
    file_path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Find-and-replace text in a file. Relative paths resolve from ~/.lightclaw/workspace.

    Args:
        file_path: Path to the file.
        old_text: Exact text to find.
        new_text: Replacement text.
    """
    from lightclaw.agent.tools.file_io import edit_file as _impl

    resp = await _impl(file_path=file_path, old_text=old_text, new_text=new_text)
    return resp.to_str()


# ---------------------------------------------------------------------------
# 6. grep_search
# ---------------------------------------------------------------------------
@tool
async def grep_search(
    pattern: str,
    path: str | None = None,
    is_regex: bool = False,
    case_sensitive: bool = True,
    context_lines: int = 0,
) -> str:
    """Search file contents by pattern, recursively. Relative paths resolve from ~/.lightclaw/workspace.

    Output format: ``path:line_number: content``.

    Args:
        pattern: Search string (or regex when is_regex=True).
        path: File or directory to search in. Defaults to workspace root.
        is_regex: Treat pattern as a regular expression. Defaults to False.
        case_sensitive: Case-sensitive matching. Defaults to True.
        context_lines: Lines of context before and after each match (like grep -C). Defaults to 0.
    """
    from lightclaw.agent.tools.file_search import grep_search as _impl

    resp = await _impl(
        pattern=pattern,
        path=path,
        is_regex=is_regex,
        case_sensitive=case_sensitive,
        context_lines=context_lines,
    )
    return resp.to_str()


# ---------------------------------------------------------------------------
# 7. glob_search
# ---------------------------------------------------------------------------
@tool
async def glob_search(
    pattern: str,
    path: str | None = None,
) -> str:
    """Find files matching a glob pattern (e.g. ``"*.py"``, ``"**/*.json"``).
    Relative paths resolve from ~/.lightclaw/workspace.

    Args:
        pattern: Glob pattern to match.
        path: Root directory to search from. Defaults to workspace root.
    """
    from lightclaw.agent.tools.file_search import glob_search as _impl

    resp = await _impl(pattern=pattern, path=path)
    return resp.to_str()


# ---------------------------------------------------------------------------
# 8. browser_control
# ---------------------------------------------------------------------------
@tool
async def browser_control(
    action: str,
    page_id: str = "default",
    selector: str = "",
    ref: str = "",
    code: str = "",
    text: str = "",
    url: str = "",
    x: float | None = None,
    y: float | None = None,
    max_depth: int = 8,
    interactive_only: bool = False,
    delta_x: float = 0,
    delta_y: float = 300,
    screenshot_format: str = "png",
    screenshot_quality: int = 85,
    await_promise: bool = False,
    paths_json: str = "",
) -> str:
    """Control the browser via Chrome DevTools Protocol (CDP).

    This is the **primary** browser automation tool.  It launches and manages
    a shared Chromium instance, providing full DOM inspection, interaction,
    and screenshot capabilities through the CDP protocol.

    Actions:
    - start        — launch / attach to the shared browser (auto-starts if needed)
    - dom_tree     — return a full DOM tree with node refs (use ref in click/type)
    - eval_js      — evaluate JavaScript expression, return result
    - query_nodes  — find DOM elements by CSS selector, return node refs
    - screenshot   — capture a screenshot; result includes preview_url — use ONLY that exact URL, never invent one; also call file_preview(file_path=...) with the saved path to display it
    - click        — click element by ref / selector / (x,y) coordinates
    - type         — type text into focused element or element matched by selector
    - navigate     — navigate to a URL
    - go_back      — navigate back in history
    - go_forward   — navigate forward in history
    - reload       — reload the current page
    - scroll       — scroll the page (delta_y pixels, positive = down)
    - tabs         — list all open tabs with page_ids
    - get_metrics  — return Chrome performance metrics
    - file_upload  — upload files to <input type="file"> via CDP
    - close        — detach CDP sessions and destroy browser session
    - stop         — full browser shutdown (close CDP, destroy session, stop Chrome)

    Args:
        action: Action to perform (see above).
        page_id: Target page identifier; "default" resolves to current page.
        selector: CSS selector for element targeting (query_nodes, click, type, file_upload).
        ref: DOM node ref from dom_tree output, e.g. "button_42" or "node_42".
        code: JavaScript expression to evaluate (eval_js action).
        text: Text to type (type action).
        url: URL to navigate to (navigate action).
        x: X coordinate in browser logical pixels (click/scroll actions).
        y: Y coordinate in browser logical pixels (click/scroll actions).
        max_depth: DOM tree traversal depth — default 8 (dom_tree action).
        interactive_only: Only show interactive elements in dom_tree.
        delta_x: Horizontal scroll amount in pixels (scroll action).
        delta_y: Vertical scroll amount in pixels, positive = down (scroll action).
        screenshot_format: "png" or "jpeg" (screenshot action).
        screenshot_quality: JPEG quality 1-100 (screenshot action).
        await_promise: Await a Promise result (eval_js action).
        paths_json: JSON array of file paths for file_upload action.
    """
    from lightclaw.agent.tools.chrome_devtools.devtools_control import devtools_control_impl

    # Inject conversation_id and channel_source from the LangGraph
    # RunnableConfig so browser sessions are correctly bound to the
    # conversation that triggered the tool call.
    _conversation_id = ""
    _channel_source = "dashboard"
    try:
        from langchain_core.runnables.config import var_child_runnable_config

        cfg = var_child_runnable_config.get({})
        if isinstance(cfg, dict):
            configurable = cfg.get("configurable") or {}
            _conversation_id = configurable.get("chat_id", "") or configurable.get("session_id", "")
            _channel_source = configurable.get("channel", "dashboard") or "dashboard"
    except Exception:
        pass

    resp = await devtools_control_impl(
        action=action,
        page_id=page_id,
        selector=selector,
        ref=ref,
        code=code,
        text=text,
        url=url,
        x=x,
        y=y,
        max_depth=max_depth,
        interactive_only=interactive_only,
        delta_x=delta_x,
        delta_y=delta_y,
        screenshot_format=screenshot_format,
        screenshot_quality=screenshot_quality,
        await_promise=await_promise,
        paths_json=paths_json,
        conversation_id=_conversation_id,
        channel_source=_channel_source,
    )
    return _serialize_tool_result(resp)


# ---------------------------------------------------------------------------
# 9. desktop_screenshot
# ---------------------------------------------------------------------------
@tool
async def desktop_screenshot(
    path: str = "",
    capture_window: bool = False,
) -> str:
    """Capture a screenshot of the entire desktop or a single window.

    Args:
        path: Optional path to save the screenshot (default: temp file).
        capture_window: If True on macOS, click a window to capture it.
    """
    from lightclaw.agent.tools.desktop_screenshot import desktop_screenshot as _impl

    resp = await _impl(path=path, capture_window=capture_window)
    return _serialize_tool_result(resp)


# ---------------------------------------------------------------------------
# 10. file_preview
# ---------------------------------------------------------------------------
@tool
async def file_preview(
    file_path: str,
) -> str:
    """Present a file to the user by generating a previewable URL or inline content.

    For images, audio, and video files this returns a signed preview URL that
    renders inline in the chat window. For other file types it returns a
    download link.

    Args:
        file_path: Absolute or relative path to the file to present.
    """
    from lightclaw.agent.tools.file_preview import file_preview as _impl

    resp = await _impl(file_path=file_path)
    return _serialize_tool_result(resp)


# ---------------------------------------------------------------------------
# 11. get_current_time
# ---------------------------------------------------------------------------
@tool
async def get_current_time() -> str:
    """Get the current system time with timezone info (e.g. '2026-02-13 19:30:45 CST (UTC+0800)')."""
    from lightclaw.agent.tools.get_current_time import get_current_time as _impl

    resp = await _impl()
    return resp.to_str()


# ---------------------------------------------------------------------------
# 12. set_user_timezone
# ---------------------------------------------------------------------------
@tool
async def set_user_timezone(timezone: str) -> str:
    """Set and persist the user's preferred timezone.

    Validates the IANA timezone name, saves it to config
    (agents.defaults.userTimezone in ~/.lightclaw/lightclaw.json), and
    confirms with the current time rendered in that zone.

    Args:
        timezone: IANA timezone name, e.g. "Asia/Shanghai", "America/New_York", "Europe/London", "UTC".
    """
    from lightclaw.agent.tools.get_current_time import set_user_timezone as _impl

    resp = await _impl(timezone=timezone)
    return resp.to_str()


# ---------------------------------------------------------------------------
# 13. tavily_search  (optional — requires TAVILY_API_KEY)
# ---------------------------------------------------------------------------
@tool
async def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
) -> str:
    """Search the web using Tavily AI search engine and return relevant results.

    This tool requires the TAVILY_API_KEY environment variable to be set.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).
        search_depth: Search depth, either "basic" or "advanced" (default "basic").
        include_answer: Whether to include a direct AI-generated answer (default True).
    """
    from langchain_tavily import TavilySearch

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."

    search = TavilySearch(
        max_results=max_results,
        search_depth=search_depth,
        include_answer=include_answer,
        api_key=api_key,
    )
    result = await search.ainvoke({"query": query})
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 14. brave_search  (optional — requires BRAVE_API_KEY)
# ---------------------------------------------------------------------------
@tool
async def brave_search(
    query: str,
    count: int = 5,
) -> str:
    """Search the web using Brave Search and return relevant results.

    Brave Search is a privacy-focused search engine with its own independent index.
    This tool requires the BRAVE_API_KEY environment variable to be set.

    Args:
        query: The search query string.
        count: Number of results to return (default 5, max 20).
    """
    import asyncio

    from langchain_community.tools import BraveSearch

    api_key = os.getenv("BRAVE_API_KEY", "")
    if not api_key:
        return "Error: BRAVE_API_KEY environment variable is not set."

    search = BraveSearch.from_api_key(
        api_key=api_key,
        search_kwargs={"count": count},
    )
    # BraveSearch only exposes a sync .run(); run it in a thread pool
    result = await asyncio.get_event_loop().run_in_executor(None, search.run, query)
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 15. google_search  (optional — requires GOOGLE_API_KEY + GOOGLE_CSE_ID)
# ---------------------------------------------------------------------------
@tool
async def google_search(
    query: str,
    num_results: int = 5,
) -> str:
    """Search the web using Google Custom Search and return relevant results.

    This tool requires both GOOGLE_API_KEY and GOOGLE_CSE_ID environment
    variables to be set. Get your API key from Google Cloud Console and
    create a Custom Search Engine at https://programmablesearchengine.google.com/.

    Args:
        query: The search query string.
        num_results: Number of results to return (default 5).
    """
    import asyncio

    from langchain_community.utilities import GoogleSearchAPIWrapper

    api_key = os.getenv("GOOGLE_API_KEY", "")
    cse_id = os.getenv("GOOGLE_CSE_ID", "")
    if not api_key or not cse_id:
        missing = []
        if not api_key:
            missing.append("GOOGLE_API_KEY")
        if not cse_id:
            missing.append("GOOGLE_CSE_ID")
        return f"Error: Missing environment variable(s): {', '.join(missing)}"

    search = GoogleSearchAPIWrapper(
        google_api_key=api_key,
        google_cse_id=cse_id,
        k=num_results,
    )
    results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: search.results(query, num_results=num_results)
    )
    if not results:
        return "No results found."
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 16. kimi_search  (optional — requires MOONSHOT_API_KEY)
# ---------------------------------------------------------------------------
@tool
async def kimi_search(
    query: str,
) -> str:
    """Search the web using Kimi (Moonshot AI) built-in web search capability.

    Kimi's moonshot-v1 model has a native web_search tool that retrieves
    up-to-date information from the internet and synthesizes an answer.
    This tool requires the MOONSHOT_API_KEY environment variable to be set.

    Args:
        query: The search query or question to answer using web search.
    """
    from openai import AsyncOpenAI

    api_key = os.getenv("MOONSHOT_API_KEY", "")
    if not api_key:
        return "Error: MOONSHOT_API_KEY environment variable is not set."

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )
    response = await client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=[{"role": "user", "content": query}],
        tools=[{"type": "web_search"}],
        temperature=0.3,
    )
    message = response.choices[0].message
    return message.content or "No answer returned."


# ---------------------------------------------------------------------------
# 17. lightclaw_search (built-in, always available)
# ---------------------------------------------------------------------------
@tool
async def lightclaw_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> str:
    """Search the web using SearchFree AI search engine and return relevant results.

    SearchFree provides AI-powered web search with advanced filtering and result
    optimization.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).
        search_depth: Search depth strategy, either "basic" or "advanced" (default "advanced").
    """
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://searchfree.site/api/search",
                json={
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                },
                timeout=30.0,
            )

            if response.status_code >= 400:
                return f"Error: SearchFree API error (HTTP {response.status_code})."

            result = response.json()
            if isinstance(result, dict) and "results" in result:
                return json.dumps(result.get("results", []), ensure_ascii=False)
            return json.dumps(result, ensure_ascii=False) if result else "No results found."
    except Exception as e:
        return f"Error: Failed to search with SearchFree: {e!s}"


# ---------------------------------------------------------------------------
# 18. read_status_file (built-in, always available)
# ---------------------------------------------------------------------------
@tool
async def read_status_file(
    target: str,
    max_chars: int = 4000,
) -> str:
    """Read a heartbeat/monitoring status file by logical name — no arbitrary paths.

    Allowed *target* values:
    - ``"heartbeat_state"`` — heartbeat cycle state
    - ``"heartbeat_today_log"`` — today's heartbeat log
    - ``"heartbeat_error_fingerprints"`` — error fingerprint database
    - ``"dreaming_status"`` — dreaming pipeline status
    - ``"heartbeat_prompt"`` — HEARTBEAT.md task prompt

    Rejects raw paths, ``..`` traversal, and absolute paths.  Returns at most
    *max_chars* characters (hard-cap 8000).

    Args:
        target:    Logical target name (one of the values above).
        max_chars: Max characters to return (default 4000).
    """
    from lightclaw.agent.tools.read_status_file import read_status_file as _impl

    resp = await _impl(target=target, max_chars=max_chars)
    return resp.to_str()


# ---------------------------------------------------------------------------
# 19. read_recent_heartbeat_errors (built-in, always available)
# ---------------------------------------------------------------------------
@tool
async def read_recent_heartbeat_errors(
    max_items: int = 10,
    max_chars: int = 4000,
) -> str:
    """Return recent heartbeat error summary — fingerprint state + daily log.

    Does **not** read the full lightclaw.log.  Returns at most *max_items*
    entries and *max_chars* total characters.  Secret-like patterns are
    redacted automatically.

    Args:
        max_items: Max error entries (default 10, hard-cap 20).
        max_chars: Max total chars returned (default 4000, hard-cap 8000).
    """
    from lightclaw.agent.tools.read_recent_heartbeat_errors import read_recent_heartbeat_errors as _impl

    resp = await _impl(max_items=max_items, max_chars=max_chars)
    return resp.to_str()


# 20. delegate_readonly_tasks (P3.1, built-in, always available)
# ---------------------------------------------------------------------------
@tool
async def delegate_readonly_tasks(
    tasks_json: str,
    max_concurrent: int = 4,
    per_worker_timeout: int = 120,
) -> str:
    """Delegate multiple read-only research tasks to concurrent isolated workers.

    Use this for parallel investigation of code, logs, config, or docs.
    Each worker runs in a fresh context with readonly tools ONLY (no write,
    shell, browser, memory mutation).  Worker tool outputs are NOT returned —
    only structured summaries.

    Args:
        tasks_json: JSON array string of task objects.  Each task:
          - goal (required): what to accomplish
          - task_id (optional): human-readable label
          - context (optional): file paths, constraints, background
          - expected_output (optional): desired answer format
          - allowed_tools (optional): subset of readonly tools to use
        max_concurrent: max parallel workers (1-4, default 4).
        per_worker_timeout: seconds per worker (30-120, default 120).

    Returns:
        XML-wrapped JSON with per-task summaries, evidence, risks, and
        recommended next actions.  Total output ≤ 12000 chars.
    """
    import json

    from lightclaw.agent.tools.delegate_readonly_tasks import (
        delegate_readonly_tasks as _impl,
    )

    try:
        tasks = json.loads(tasks_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({
            "status": "rejected",
            "summary": "tasks_json is not valid JSON. Provide a JSON array of task objects.",
            "recommended_next_action": "Fix JSON syntax and retry.",
        }, ensure_ascii=False)
    if not isinstance(tasks, list):
        return json.dumps({
            "status": "rejected",
            "summary": "tasks_json must be a JSON array.",
            "recommended_next_action": "Provide a list of task objects.",
        }, ensure_ascii=False)

    return await _impl(
        tasks=tasks,
        max_concurrent=max_concurrent,
        per_worker_timeout=per_worker_timeout,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Registry of optional tools: (env_var_names, tool_object)
# All env vars in the tuple must be non-empty for the tool to be enabled.
_OPTIONAL_TOOLS: list[tuple[tuple[str, ...], object]] = [
    (("TAVILY_API_KEY",), tavily_search),
    (("BRAVE_API_KEY",), brave_search),
    (("GOOGLE_API_KEY", "GOOGLE_CSE_ID"), google_search),
    (("MOONSHOT_API_KEY",), kimi_search),
]


def load_builtin_tools() -> list:
    """Return built-in tools as LangChain tool instances.

    Always includes the 13 core tools (including grep_search, glob_search,
    lightclaw_search, browser_control, and set_user_timezone).
    Additionally includes optional search tools when their required
    environment variables are set:

    * ``tavily_search``     — requires ``TAVILY_API_KEY``
    * ``brave_search``      — requires ``BRAVE_API_KEY``
    * ``google_search``     — requires ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID``
    * ``kimi_search``       — requires ``MOONSHOT_API_KEY``
    """
    tools = [
        execute_shell_command,
        read_file,
        write_file,
        append_file,
        edit_file,
        grep_search,
        glob_search,
        browser_control,
        desktop_screenshot,
        file_preview,
        get_current_time,
        set_user_timezone,
        lightclaw_search,
        read_status_file,
        read_recent_heartbeat_errors,
        delegate_readonly_tasks,
    ]
    for env_vars, tool_obj in _OPTIONAL_TOOLS:
        if all(os.getenv(v) for v in env_vars):
            tools.append(tool_obj)
    return tools
