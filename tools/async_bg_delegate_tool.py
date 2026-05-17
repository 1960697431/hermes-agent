#!/usr/bin/env python3
"""
Async Background Delegate Tool

Adds a new tool `delegate_task_async` that runs a child agent in a
daemon thread and returns immediately.  When the child finishes its
result is pushed back into the current chat via the gateway runner.

This is a zero-modification extension:
  - No existing Hermes source files are changed.
  - The new file is discovered automatically by tools/registry.py.
  - `hermes update` will stash/restore local changes but will not delete
    this file because it is untracked.  If upstream ever adds a file with
    the same name a git conflict will occur (very unlikely with this name).
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# In-memory registry of background delegations (lost on gateway restart).
_background_delegations: Dict[str, Dict[str, Any]] = {}
_bg_lock = threading.Lock()


def _notify_gateway(runner, source, loop, goal, summary, status, duration):
    """Schedule a coroutine on the gateway's event loop to send the result."""
    import asyncio

    async def _send():
        try:
            adapter = runner.adapters.get(source.platform)
            if not adapter:
                return
            _thread_metadata = (
                {"thread_id": source.thread_id} if source.thread_id else None
            )
            preview = goal[:60] + ("..." if len(goal) > 60 else "")
            icon = "[OK]" if status == "completed" else "[ERR]"
            header = (
                f"{icon} 后台任务完成\n"
                f'任务: "{preview}"\n'
                f"耗时: {duration:.1f}s\n\n"
            )
            content = header + (summary or "(无输出)")

            # Some adapters (BlueBubbles/iMessage, WeChat) cannot handle local
            # MEDIA paths.  Strip them so the message at least arrives as text.
            media_files, text_only = adapter.extract_media(content)
            images, text_without_images = adapter.extract_images(text_only)

            # Send text first
            if text_without_images.strip():
                await adapter.send(
                    chat_id=source.chat_id,
                    content=text_without_images,
                    metadata=_thread_metadata,
                )
            # Then images (if any inline image URLs remain)
            if images:
                for img in images:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=img,
                        metadata=_thread_metadata,
                    )
            # Finally media attachments (only for adapters that support them)
            if media_files:
                for mf in media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=mf,
                        metadata=_thread_metadata,
                    )
        except Exception as exc:
            logger.warning("Failed to send async delegation result: %s", exc)

    try:
        asyncio.run_coroutine_threadsafe(_send(), loop)
    except Exception as exc:
        logger.warning("Failed to schedule gateway notification: %s", exc)


def delegate_task_async(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    max_iterations: Optional[int] = None,
    role: Optional[str] = None,
    parent_agent=None,
) -> str:
    """
    Launch a child agent in a background daemon thread.
    Returns immediately with a task_id.  The result is pushed to chat
    automatically when the child finishes.
    """
    if not goal:
        return json.dumps({"error": "goal is required"})
    if parent_agent is None:
        return json.dumps(
            {"error": "delegate_task_async requires a parent agent context."}
        )

    # Deferred imports avoid circular-import problems during registry discovery.
    from tools.delegate_tool import (
        DEFAULT_MAX_ITERATIONS,
        _build_child_agent,
        _get_max_spawn_depth,
        _load_config,
        _normalize_role,
        _resolve_delegation_credentials,
        _run_single_child,
    )

    # Depth guard
    depth = getattr(parent_agent, "_delegate_depth", 0)
    max_spawn = _get_max_spawn_depth()
    if depth >= max_spawn:
        return json.dumps(
            {
                "error": (
                    f"Delegation depth limit reached "
                    f"(depth={depth}, max={max_spawn})"
                )
            }
        )

    # Gateway context (set by gateway/run.py when running in gateway mode)
    gateway_runner = getattr(parent_agent, "_gateway_runner", None)
    gateway_source = getattr(parent_agent, "_gateway_source", None)
    gateway_loop = getattr(parent_agent, "_gateway_loop", None)

    cfg = _load_config()
    effective_max_iter = max_iterations or cfg.get(
        "max_iterations", DEFAULT_MAX_ITERATIONS
    )
    creds = _resolve_delegation_credentials(cfg, parent_agent)
    top_role = _normalize_role(role)

    task_id = f"async_del_{os.urandom(4).hex()}"

    # Build child on the main thread (AIAgent construction is not thread-safe).
    child = _build_child_agent(
        task_index=0,
        goal=goal,
        context=context,
        toolsets=toolsets,
        model=creds["model"],
        max_iterations=effective_max_iter,
        task_count=1,
        parent_agent=parent_agent,
        override_provider=creds["provider"],
        override_base_url=creds["base_url"],
        override_api_key=creds["api_key"],
        override_api_mode=creds["api_mode"],
        role=top_role,
    )

    def _run_in_thread():
        start = time.monotonic()
        try:
            result = _run_single_child(0, goal, child, parent_agent)
            duration = time.monotonic() - start

            summary = ""
            status = "completed"
            if isinstance(result, dict):
                summary = result.get("summary", "")
                status = result.get("status", "completed")

            with _bg_lock:
                _background_delegations[task_id] = {
                    "status": status,
                    "summary": summary,
                    "duration_seconds": round(duration, 2),
                    "completed_at": time.time(),
                    "goal": goal,
                }

            if gateway_runner and gateway_source and gateway_loop:
                _notify_gateway(
                    gateway_runner,
                    gateway_source,
                    gateway_loop,
                    goal,
                    summary,
                    status,
                    duration,
                )
        except Exception as exc:
            logger.exception("Background delegation %s failed", task_id)
            with _bg_lock:
                _background_delegations[task_id] = {
                    "status": "error",
                    "error": str(exc),
                    "completed_at": time.time(),
                    "goal": goal,
                }

    thread = threading.Thread(
        target=_run_in_thread, daemon=True, name=f"async-del-{task_id}"
    )
    thread.start()

    with _bg_lock:
        _background_delegations[task_id] = {
            "status": "running",
            "started_at": time.time(),
            "goal": goal,
            "thread": thread,
        }

    return json.dumps(
        {
            "status": "dispatched",
            "background_task_id": task_id,
            "goal": goal,
            "message": (
                f"后台任务已启动 (ID: {task_id})。\n"
                f"完成后结果将自动推送到当前聊天。你可以继续对话。"
            ),
        }
    )


def check_async_task(task_id: str) -> str:
    """Check the status of a previously dispatched async task."""
    with _bg_lock:
        record = _background_delegations.get(task_id)
    if not record:
        return json.dumps({"error": f"Task {task_id} not found"})
    # Exclude non-serialisable Thread object
    return json.dumps({k: v for k, v in record.items() if k != "thread"})


# ---------------------------------------------------------------------------
# Register the tool so Hermes discovers it automatically.
# ---------------------------------------------------------------------------
registry.register(
    name="delegate_task_async",
    toolset="delegation",
    schema={
        "name": "delegate_task_async",
        "description": (
            "启动一个后台异步子 agent 执行任务，立即返回不阻塞主对话。"
            "完成后结果自动推送到当前聊天。仅支持单任务模式 (goal)。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "子 agent 的任务目标",
                },
                "context": {
                    "type": "string",
                    "description": "可选的上下文信息",
                },
                "toolsets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "子 agent 可用工具集列表",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "子 agent 最大迭代次数",
                },
                "role": {
                    "type": "string",
                    "enum": ["leaf", "orchestrator"],
                    "description": "子 agent 角色，leaf 不能继续委派",
                },
            },
            "required": ["goal"],
        },
    },
    handler=lambda args, **kw: delegate_task_async(
        goal=args.get("goal"),
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        max_iterations=args.get("max_iterations"),
        role=args.get("role"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=lambda: True,
    requires_env=[],
)

# Also register a small helper to query task status.
registry.register(
    name="check_async_task",
    toolset="delegation",
    schema={
        "name": "check_async_task",
        "description": "查询一个后台异步任务的状态和结果",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "后台任务 ID",
                },
            },
            "required": ["task_id"],
        },
    },
    handler=lambda args, **kw: check_async_task(args.get("task_id", "")),
    check_fn=lambda: True,
    requires_env=[],
)
