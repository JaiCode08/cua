from __future__ import annotations

import asyncio
import copy
import dataclasses
import importlib.util
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
from PIL import Image

from flakiness.classifier import classify_failure
from flakiness.reporter import RunResult, generate_report
from replay.cache import Cache
from replay.executor import (
    ActionResult,
    ContinueReplayStep,
    ReplayExecutor,
    normalize_replay_action,
    validate_trajectory,
)
from replay.tool import continue_replay
from replay.waits import wait_for_screenshot_stable
from verifiers.base import VerificationResult, Verifier


class DriverWrapper:
    """Thin async wrapper exposing cua-driver calls to task verifiers."""

    def __init__(self, executor: ReplayExecutor):
        self.executor = executor

    async def call_tool(self, tool: str, arguments: dict, timeout: Optional[float] = None) -> dict:
        return await self.executor.call_tool(tool, arguments, timeout)

    async def screenshot(self):
        return (await self.executor.capture_observation()).screenshot

    async def launch_app(self, app_name: str, **kwargs):
        owned = bool(kwargs.pop("_owned", True))
        arguments = {"name": app_name, **kwargs}
        payload = await self.call_tool("launch_app", arguments, self.executor.launch_timeout)
        await self.executor.register_launch_payload(payload, owned=owned)
        return payload

    async def list_windows(self, pid: Optional[int] = None) -> list[dict]:
        return await self.executor.list_windows(pid)

    async def adopt_pid(self, pid: int, *, owned: bool = False) -> None:
        await self.executor.adopt_pid(pid, owned=owned)

    async def cleanup_owned(self) -> list[dict]:
        """Close only processes/sessions registered as owned by this run."""
        return await self.executor.teardown()


def load_verifier(path: str | Path) -> Verifier:
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location("custom_verifier", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load verifier module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, Verifier) and value is not Verifier
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one Verifier subclass in {path}, found {len(candidates)}"
        )
    return candidates[0]()


def _command_output(command: list[str], timeout: float = 10.0) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_environment_metadata(agent: str, driver_command: str) -> dict:
    resolution = "unknown"
    try:
        import ctypes

        user32 = ctypes.windll.user32
        resolution = f"{user32.GetSystemMetrics(0)}x{user32.GetSystemMetrics(1)}"
    except Exception:
        pass
    agent_binary = shutil.which(f"{agent}.cmd") or shutil.which(agent) or agent
    packages = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    return {
        "os": platform.platform(),
        "image": os.environ.get("CUA_IMAGE", "local-desktop"),
        "screen_resolution": resolution,
        "cua_driver_version": _command_output([driver_command, "--version"]),
        "agent_cli": agent,
        "agent_cli_version": _command_output([agent_binary, "--version"]),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "python_version": platform.python_version(),
        "packages": packages,
    }


def _extract_action(text: str) -> dict:
    try:
        envelope = json.loads(text)
        if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
            text = envelope["result"]
    except json.JSONDecodeError:
        pass
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("tool"), str)
            and isinstance(value.get("arguments", {}), dict)
        ):
            candidates.append(value)
    if not candidates:
        raise ValueError(f"Claude did not return a tool/arguments JSON object: {text[-1000:]}")
    action = candidates[-1]
    args = action.setdefault("arguments", {})
    # Agent interventions must be reusable in later sessions.
    fresh_reference = "element_index" in args or "element_token" in args
    keys_to_remove = ("session",) if fresh_reference else ("pid", "window_id", "session")
    for key in keys_to_remove:
        args.pop(key, None)
    return action


def _recorded_action_failed(action: dict) -> bool:
    summary = str(action.get("result_summary", "")).casefold()
    failure_markers = (
        " rejected ",
        " failed",
        " failure",
        " error",
        " unable to ",
        " cannot ",
        " couldn't ",
    )
    padded = f" {summary} "
    return any(marker in padded for marker in failure_markers)


async def _run_agent(
    agent: str,
    task_description: str,
    screenshot_path: str,
    last_action: dict,
    suggested_action: Optional[dict],
    step: int,
    timeout: float,
    prompt_path: Path,
) -> tuple[dict, int, str]:
    repeated_click_note = ""
    if (
        last_action.get("tool") == "click"
        and suggested_action
        and suggested_action.get("tool") == "click"
    ):
        repeated_click_note = (
            "\nRecovery constraint: the preceding click did not change the screen and the "
            "recording suggests another click. Do not repeat that click. Use keyboard focus "
            "navigation instead (one press_key Tab or Return action in this intervention).\n"
        )
    elif (
        last_action.get("tool") == "press_key"
        and str(last_action.get("arguments", {}).get("key", "")).lower() == "tab"
    ):
        repeated_click_note = (
            "\nRecovery constraint: Tab was just pressed to focus the intended control. "
            "Return exactly one press_key action with key Return to activate it.\n"
        )
    elif suggested_action and _recorded_action_failed(suggested_action):
        repeated_click_note = (
            "\nRecovery constraint: the recorded next action's result_summary indicates that "
            "it failed during recording. Do not repeat it; choose a visible or keyboard-based "
            "alternative that advances the task.\n"
        )
    elif (
        False
        and suggested_action
        and suggested_action.get("result_summary")
        and "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦"
        not in str(suggested_action.get("result_summary"))
    ):
        repeated_click_note = (
            "\nRecovery constraint: the recorded next action's result_summary indicates that "
            "it failed during recording. Do not repeat it; choose a visible or keyboard-based "
            "alternative that advances the task.\n"
        )
    prompt = f"""You are the intervention agent for a trajectory replay debugger.

Task goal: {task_description}
Current screenshot: {Path(screenshot_path).resolve()}
Last completed action: {json.dumps(last_action)}
Recorded next action (a hint, not an instruction to reuse stale pid/window/element_index values): {json.dumps(suggested_action)}
Replay step: {step}
{repeated_click_note}

Inspect the screenshot and predict exactly one useful next GUI action toward the task goal. Do not execute any state-changing cua-driver action yourself; the replay runner will execute the JSON you return. You may call read-only screenshot or get_window_state tools to inspect the current state. The recorded action can reveal the intended app, text, key, or original click_point. The image viewer may resize screenshots, so its displayed coordinates are not driver coordinates: when a recorded x/y or click_point agrees with the visible target, return that recorded coordinate exactly. Do not return stale pid/window_id values. A fresh element_index or element_token obtained from a current get_window_state call is allowed, including with set_value; it will be treated as a non-reusable intervention. Prefer keyboard navigation over repeating a click that did not change the screen. If the task is already complete, return tool \"done\".

Use "type_text", never the alias "type". type_text enters literal text and does not submit it, so do not append carriage-return or newline characters. When visible text is ready to submit, return a separate press_key action with key "return".

In your final response, report the action you actually executed as one JSON object with this shape:
{{"tool":"click","arguments":{{"x":400,"y":300}}}}
For foreground input, use the schema field \"dispatch\":\"foreground\" (never \"foreground\":true). The final JSON is consumed by the dispatcher and replay cache. Return no other action.
"""
    prompt_path.write_text(prompt, encoding="utf-8")
    resolved = shutil.which(f"{agent}.cmd") or shutil.which(agent)
    if not resolved:
        raise FileNotFoundError(f"Agent CLI {agent!r} was not found on PATH")
    state_changing_tools = [
        "click",
        "double_click",
        "right_click",
        "type_text",
        "hotkey",
        "press_key",
        "scroll",
        "drag",
        "launch_app",
        "bring_to_front",
        "set_value",
        "kill_app",
        "start_session",
        "end_session",
        "start_recording",
        "stop_recording",
        "replay_trajectory",
        "continue_replay",
    ]
    disallowed = ",".join(f"mcp__cua-driver__{name}" for name in state_changing_tools)
    disallowed += ",Bash,FileEdit,Glob"
    command = [
        resolved,
        "-p",
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        "--disallowedTools",
        disallowed,
    ]
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Agent timed out after {timeout}s")
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    duration_ms = int((time.monotonic() - start) * 1000)
    output = stdout.decode("utf-8", errors="replace")
    error = stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Agent CLI exited with {proc.returncode}: {error or output}")
    return _extract_action(output), duration_ms, output


def _write_jsonl(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")


def _matches_suggestion(actual: dict, suggested: Optional[dict]) -> bool:
    if not suggested or actual.get("tool") != suggested.get("tool"):
        return False
    actual_args = actual.get("arguments", {})
    suggested_args = suggested.get("arguments", {})
    ignored = {"pid", "window_id", "session", "dispatch", "element_index"}
    for key, value in suggested_args.items():
        if key not in ignored and actual_args.get(key) != value:
            return False
    if "element_index" in suggested_args:
        point = suggested.get("click_point", {})
        if point and all(key in actual_args for key in ("x", "y")):
            return (
                abs(actual_args["x"] - point["x"]) <= 30
                and abs(actual_args["y"] - point["y"]) <= 30
            )
        return False
    return True


def _normalize_agent_action(actual: dict, suggested: Optional[dict]) -> dict:
    """Snap resized-image coordinates to a nearby recorded driver click point."""
    if actual.get("tool") == "type":
        actual["tool"] = "type_text"
    actual_args = actual.setdefault("arguments", {})
    if actual_args.pop("foreground", False):
        actual_args["dispatch"] = "foreground"

    suggested_args = suggested.get("arguments", {}) if suggested else {}
    if "pid" in suggested_args and "pid" not in actual_args:
        actual_args["pid"] = suggested_args["pid"]
    if "window_id" in suggested_args and "window_id" not in actual_args:
        actual_args["window_id"] = suggested_args["window_id"]

    if not suggested or actual.get("tool") not in {"click", "double_click", "right_click"}:
        if actual.get("tool") in {"type", "type_text"}:
            actual_args.pop("dispatch", None)
        return actual
    if actual.get("tool") != suggested.get("tool"):
        return actual
    suggested_args = suggested.get("arguments", {})
    point = suggested.get("click_point") or {
        "x": suggested_args.get("x"),
        "y": suggested_args.get("y"),
    }
    numeric = (int, float)
    if not all(isinstance(point.get(key), numeric) for key in ("x", "y")):
        return actual
    if not all(isinstance(actual_args.get(key), numeric) for key in ("x", "y")):
        return actual
    if abs(actual_args["x"] - point["x"]) <= 100 and abs(actual_args["y"] - point["y"]) <= 100:
        actual_args["x"] = point["x"]
        actual_args["y"] = point["y"]
        if suggested_args.get("dispatch") == "foreground":
            actual_args["dispatch"] = "foreground"
    return actual


class DriverUnavailableError(RuntimeError):
    pass


def _resolve_driver_command() -> str:
    sibling_driver = Path(sys.executable).with_name(
        "cua-driver.exe" if os.name == "nt" else "cua-driver"
    )
    return (
        str(sibling_driver)
        if sibling_driver.exists()
        else shutil.which("cua-driver") or "cua-driver"
    )


async def driver_daemon_is_running(driver_command: str, timeout: float = 2.0) -> bool:
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [driver_command, "status"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return False
    output = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}".lower()
    return result.returncode == 0 and "daemon is running" in output


async def wait_for_driver_ready(
    driver_command: str, timeout: float = 10.0, poll: float = 0.2
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "driver did not respond"
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [driver_command, "call", "list_windows", "{}"],
                capture_output=True,
                text=True,
                timeout=min(2.0, remaining),
                check=False,
            )
            if result.returncode == 0:
                return
            last_error = (result.stderr or result.stdout or "driver returned an error").strip()
        except Exception as exc:
            last_error = str(exc)
        await asyncio.sleep(poll)
    raise DriverUnavailableError(
        f"cua-driver daemon was not ready after {timeout:.1f}s: {last_error}"
    )


def _write_daemon_warning(output_dir: Path, warning: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "daemon.jsonl", warning)


async def run_flakiness(
    trajectory: str,
    verifier_path: str,
    agent: str,
    runs: int,
    mode: str,
    max_agent_interventions: int,
    output_dir: str,
    action_timeout: float,
    launch_timeout: float,
    replay_timeout: float,
    agent_timeout: float,
    cleanup_timeout: float,
    report: Optional[str] = None,
) -> list[RunResult]:
    """Start a private driver daemon and guarantee cleanup around the batch."""
    output_path = Path(output_dir).resolve()
    driver_command = _resolve_driver_command()
    daemon_proc = None
    try:
        # Status does not auto-start the daemon. Reuse an existing daemon
        # without claiming ownership; otherwise start and own this process.
        if not await driver_daemon_is_running(driver_command):
            daemon_proc = subprocess.Popen(
                [driver_command, "serve", "--no-overlay"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            try:
                await wait_for_driver_ready(
                    driver_command, timeout=min(max(2.0, launch_timeout), 30.0)
                )
            except DriverUnavailableError as exc:
                _write_daemon_warning(
                    output_path,
                    {
                        "event": "driver_unavailable",
                        "category": "environment_drift",
                        "message": str(exc),
                    },
                )
                raise
        return await _run_flakiness_core(
            trajectory=trajectory,
            verifier_path=verifier_path,
            agent=agent,
            runs=runs,
            mode=mode,
            max_agent_interventions=max_agent_interventions,
            output_dir=output_dir,
            action_timeout=action_timeout,
            launch_timeout=launch_timeout,
            replay_timeout=replay_timeout,
            agent_timeout=agent_timeout,
            cleanup_timeout=cleanup_timeout,
            report=report,
            driver_command_override=driver_command,
        )
    finally:
        if daemon_proc is not None:
            try:
                daemon_proc.terminate()
                daemon_proc.wait(timeout=cleanup_timeout)
            except subprocess.TimeoutExpired:
                try:
                    daemon_proc.kill()
                    daemon_proc.wait(timeout=cleanup_timeout)
                except Exception as exc:
                    _write_daemon_warning(
                        output_path, {"event": "daemon_cleanup_warning", "message": str(exc)}
                    )
            except Exception as exc:
                _write_daemon_warning(
                    output_path, {"event": "daemon_cleanup_warning", "message": str(exc)}
                )


async def _run_flakiness_core(
    trajectory: str,
    verifier_path: str,
    agent: str,
    runs: int,
    mode: str,
    max_agent_interventions: int,
    output_dir: str,
    action_timeout: float,
    launch_timeout: float,
    replay_timeout: float,
    agent_timeout: float,
    cleanup_timeout: float,
    report: Optional[str] = None,
    driver_command_override: Optional[str] = None,
) -> list[RunResult]:
    trajectory_path = Path(trajectory).resolve()
    verifier_file = Path(verifier_path).resolve()
    output_path = Path(output_dir).resolve()
    validate_trajectory(trajectory_path)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    output_path.mkdir(parents=True, exist_ok=True)
    verifier = load_verifier(verifier_file)
    strict = mode == "strict"
    match_mode = mode.removeprefix("forgiving-") if not strict else "exact"
    cache = Cache(match_mode)
    cache.load_from_dir(trajectory_path)
    sibling_driver = Path(sys.executable).with_name(
        "cua-driver.exe" if os.name == "nt" else "cua-driver"
    )
    driver_command = driver_command_override or (
        str(sibling_driver)
        if sibling_driver.exists()
        else shutil.which("cua-driver") or "cua-driver"
    )

    results: list[RunResult] = []

    for run_index in range(1, runs + 1):
        run_dir = output_path / "runs" / f"run_{run_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "log.jsonl"
        log_path.write_text("", encoding="utf-8")
        executor = ReplayExecutor(
            trajectory_path,
            action_timeout=action_timeout,
            launch_timeout=launch_timeout,
            replay_timeout=replay_timeout,
            action_transform=verifier.prepare_action,
            driver_command=driver_command,
        )
        driver = DriverWrapper(executor)
        started = time.monotonic()
        initial_stats = cache.stats()
        steps: list[ContinueReplayStep] = []
        action_results: list[ActionResult] = []
        verifier_result: Optional[VerificationResult] = None
        error: Optional[Exception] = None
        context = "setup"
        interventions = 0
        last_failed_action: Optional[ActionResult] = None
        reset_succeeded = False
        pending_terminal: Optional[tuple[dict, str, str]] = None

        def commit_pending_terminal(current_step: int) -> None:
            if not (verifier_result and verifier_result.passed and pending_terminal):
                return
            terminal_action, terminal_path, terminal_scope = pending_terminal
            try:
                with Image.open(terminal_path) as terminal_image:
                    cache.store_terminal(terminal_action, terminal_image, terminal_scope)
                _write_jsonl(log_path, {"event": "terminal_state_cached", "step": current_step})
            except Exception as terminal_error:
                _write_jsonl(
                    log_path,
                    {
                        "event": "cache_store_warning",
                        "message": f"Could not store terminal state: {terminal_error}",
                    },
                )

        async def execute_run() -> None:
            nonlocal context, interventions, verifier_result, last_failed_action
            nonlocal reset_succeeded, pending_terminal
            await executor.setup()
            context = "reset"
            _write_jsonl(log_path, {"event": "reset_started", "run": run_index})
            await verifier.reset(driver)
            reset_succeeded = True
            _write_jsonl(log_path, {"event": "reset_completed", "run": run_index})
            context = "replay"
            step = 1
            last_action = {"tool": "__start__"}
            while True:
                status, step, last_action, replay_results, replay_steps = await continue_replay(
                    executor, cache, run_dir, step, last_action
                )
                action_results.extend(replay_results)
                steps.extend(replay_steps)
                for replay_step in replay_steps:
                    _write_jsonl(
                        log_path, {"event": "replay_step", **dataclasses.asdict(replay_step)}
                    )
                    for warning in replay_step.warnings:
                        _write_jsonl(
                            log_path,
                            {"event": "artifact_warning", "step": replay_step.step, **warning},
                        )
                if status == "replay_complete":
                    break
                if status == "error":
                    last_failed_action = replay_results[-1] if replay_results else None
                    raise RuntimeError(
                        last_failed_action.error
                        if last_failed_action
                        else "Observation capture failed"
                    )
                if strict:
                    last_failed_action = ActionResult(
                        step, False, "cache_miss", 0, "cache_miss (strict mode)"
                    )
                    action_results.append(last_failed_action)
                    raise RuntimeError(last_failed_action.error)

                if interventions > 0:
                    try:
                        completion_probe = await asyncio.wait_for(
                            verifier.verify(driver), timeout=cleanup_timeout
                        )
                    except Exception as probe_error:
                        _write_jsonl(
                            log_path,
                            {"event": "completion_probe_error", "message": str(probe_error)},
                        )
                    else:
                        _write_jsonl(
                            log_path,
                            {"event": "completion_probe", **dataclasses.asdict(completion_probe)},
                        )
                        if completion_probe.passed:
                            verifier_result = completion_probe
                            pending_terminal = (
                                copy.deepcopy(last_action),
                                replay_steps[-1].screenshot_path,
                                replay_steps[-1].capture_scope,
                            )
                            break
                if interventions >= max_agent_interventions:
                    last_failed_action = ActionResult(
                        step,
                        False,
                        "cache_miss",
                        0,
                        "cache_miss_unrecovered: intervention limit reached",
                    )
                    action_results.append(last_failed_action)
                    raise RuntimeError(last_failed_action.error)
                interventions += 1
                context = "agent"
                miss_path = replay_steps[-1].screenshot_path
                source_suggestion = cache.suggest(last_action)
                prompt_suggestion = (
                    verifier.prepare_action(copy.deepcopy(source_suggestion))
                    if source_suggestion
                    else None
                )
                action, duration_ms, raw_output = await _run_agent(
                    agent,
                    verifier.task_description,
                    miss_path,
                    last_action,
                    prompt_suggestion,
                    step,
                    agent_timeout,
                    run_dir / f"prompt_step_{step:05d}.txt",
                )
                (run_dir / f"agent_step_{step:05d}.json").write_text(raw_output, encoding="utf-8")
                if action["tool"] == "done":
                    pending_terminal = (
                        copy.deepcopy(last_action),
                        miss_path,
                        replay_steps[-1].capture_scope,
                    )
                    break
                action = _normalize_agent_action(action, prompt_suggestion)
                action = normalize_replay_action(action, source=f"agent action at step {step}")
                action["_intervention_id"] = f"run-{run_index}-step-{step}"

                turn_dir = run_dir / f"turn-{step:05d}"
                context = "replay"
                result = await executor.execute_action(action, step, run_dir)
                if not result.ok:
                    _write_jsonl(
                        log_path,
                        {"event": "agent_action_failed", "error": result.error, "step": step},
                    )
                    last_failed_action = result
                    action_results.append(result)
                    step += 1
                    context = "replay"
                    continue
                try:
                    await wait_for_screenshot_stable(driver, timeout=min(action_timeout, 2.0))
                except TimeoutError as settle_error:
                    _write_jsonl(
                        log_path,
                        {"event": "settle_timeout", "step": step, "message": str(settle_error)},
                    )
                screenshot_path, tree_path, capture_warnings = await executor.capture_artifacts(
                    turn_dir
                )
                result.duration_ms += duration_ms
                result.screenshot_path = screenshot_path
                result.accessibility_tree_path = tree_path
                result.warnings.extend(capture_warnings)
                action_results.append(result)
                for warning in capture_warnings:
                    _write_jsonl(log_path, {"event": "artifact_warning", "step": step, **warning})
                try:
                    with Image.open(miss_path) as miss_image:
                        cache.store(
                            last_action,
                            miss_image,
                            source_suggestion
                            if _matches_suggestion(action, prompt_suggestion)
                            else action,
                            capture_scope=replay_steps[-1].capture_scope,
                        )
                except FileNotFoundError:
                    _write_jsonl(
                        log_path,
                        {
                            "event": "cache_store_warning",
                            "message": f"miss_path not found: {miss_path}",
                        },
                    )
                intervention_step = ContinueReplayStep(
                    step,
                    False,
                    action["tool"],
                    action["arguments"],
                    duration_ms,
                    screenshot_path or miss_path,
                    warnings=list(result.warnings),
                    capture_scope=replay_steps[-1].capture_scope,
                )
                steps[-1] = intervention_step
                _write_jsonl(
                    log_path,
                    {"event": "agent_intervention", **dataclasses.asdict(intervention_step)},
                )
                last_action = (
                    source_suggestion if _matches_suggestion(action, prompt_suggestion) else action
                )
                step += 1
                context = "replay"
            context = "verify"
            if verifier_result is None:
                try:
                    verifier_result = await asyncio.wait_for(
                        verifier.verify(driver), timeout=cleanup_timeout
                    )
                except Exception as verify_error:
                    verifier_result = VerificationResult.fail(
                        f"Verifier raised {type(verify_error).__name__}: {verify_error}",
                        {"error_type": type(verify_error).__name__},
                    )
                _write_jsonl(
                    log_path, {"event": "verification", **dataclasses.asdict(verifier_result)}
                )

            commit_pending_terminal(step)

        try:
            await asyncio.wait_for(execute_run(), timeout=replay_timeout)
        except Exception as exc:
            error = exc
            _write_jsonl(
                log_path,
                {
                    "event": "run_error",
                    "context": context,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        finally:
            # A successful reset establishes a valid run. Always attempt final
            # verification, even when replay or recovery failed; preserve the
            # original replay error as the primary failure classification.
            if reset_succeeded and verifier_result is None:
                try:
                    verifier_result = await asyncio.wait_for(
                        verifier.verify(driver), timeout=cleanup_timeout
                    )
                except Exception as verify_error:
                    verifier_result = VerificationResult.fail(
                        f"Verifier raised {type(verify_error).__name__}: {verify_error}",
                        {"error_type": type(verify_error).__name__},
                    )
                _write_jsonl(
                    log_path, {"event": "verification", **dataclasses.asdict(verifier_result)}
                )

            try:
                await asyncio.wait_for(verifier.teardown(driver), timeout=cleanup_timeout)
            except Exception as cleanup_error:
                _write_jsonl(
                    log_path,
                    {
                        "event": "cleanup_warning",
                        "code": "verifier_cleanup_failed",
                        "message": str(cleanup_error),
                    },
                )
            for cleanup_warning in (await executor.teardown()) or []:
                _write_jsonl(log_path, {"event": "cleanup_warning", **cleanup_warning})

        passed = error is None and verifier_result is not None and verifier_result.passed
        failure = (
            None
            if passed
            else classify_failure(error, last_failed_action, verifier_result, context)
        )
        current_stats = cache.stats()
        cache_hits = current_stats["hits"] - initial_stats["hits"]
        cache_misses = current_stats["misses"] - initial_stats["misses"]
        result_payload = (
            dataclasses.asdict(verifier_result)
            if verifier_result
            else {
                "passed": False,
                "reason": failure.reason if failure else "Verifier did not run",
                "details": failure.detail if failure else {},
            }
        )
        (run_dir / "verifier_result.json").write_text(
            json.dumps(result_payload, indent=2), encoding="utf-8"
        )
        results.append(
            RunResult(
                run_index=run_index,
                passed=passed,
                failure=failure,
                steps=steps,
                action_results=action_results,
                verifier_result=verifier_result,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                agent_interventions=interventions,
                duration_ms=int((time.monotonic() - started) * 1000),
                log_path=str(log_path),
                screenshot_dir=str(run_dir),
            )
        )

    metadata = get_environment_metadata(agent, driver_command)
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for result in results:
        (Path(result.log_path).parent / "environment.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    report_path = generate_report(
        results,
        output_path,
        trajectory_path.name,
        str(trajectory_path),
        str(verifier_file),
        Path(report).resolve() if report else None,
        mode=mode,
        agent=agent,
        max_agent_interventions=max_agent_interventions,
    )
    passes = sum(result.passed for result in results)
    hits = sum(result.cache_hits for result in results)
    misses = sum(result.cache_misses for result in results)
    total_interventions = sum(result.agent_interventions for result in results)
    first_failure = next((result for result in results if not result.passed), None)
    print(f"Task: {trajectory_path.name}")
    print(f"Runs: {runs}")
    print(f"Passed: {passes}")
    print(f"Failed: {runs - passes}")
    print(f"Flake rate: {(runs - passes) / runs:.0%}")
    print(
        f"Cache hit rate: {hits / (hits + misses):.0%} ({hits} hits / {misses} misses across {runs} runs)"
        if hits + misses
        else "Cache hit rate: 0% (0 hits / 0 misses)"
    )
    print(
        f"Agent interventions: {total_interventions} (avg {total_interventions / runs:.1f} per run)"
    )
    if first_failure and first_failure.failure:
        print(
            f"First failure: run: {first_failure.run_index} reason: {first_failure.failure.category} detail: {first_failure.failure.reason}"
        )
    print(f"Report: {report_path}")
    return results


@click.command()
@click.option("--trajectory", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--verifier", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--agent", default="claude", show_default=True)
@click.option("--runs", default=10, type=click.IntRange(min=1), show_default=True)
@click.option(
    "--mode",
    default="forgiving-perceptual",
    type=click.Choice(["strict", "forgiving-perceptual", "forgiving-crop", "forgiving-downsample"]),
)
@click.option("--max-agent-interventions", default=5, type=click.IntRange(min=0), show_default=True)
@click.option("--report", default=None, type=click.Path(path_type=Path))
@click.option("--output-dir", default=None, type=click.Path(path_type=Path))
@click.option("--action-timeout", default=15.0, type=click.FloatRange(min=0.1))
@click.option("--launch-timeout", default=30.0, type=click.FloatRange(min=0.1))
@click.option("--replay-timeout", default=600.0, type=click.FloatRange(min=1.0))
@click.option("--agent-timeout", default=300.0, type=click.FloatRange(min=1.0))
@click.option("--cleanup-timeout", default=30.0, type=click.FloatRange(min=0.1))
def cli(
    trajectory: Path,
    verifier: Path,
    agent: str,
    runs: int,
    mode: str,
    max_agent_interventions: int,
    report: Optional[Path],
    output_dir: Optional[Path],
    action_timeout: float,
    launch_timeout: float,
    replay_timeout: float,
    agent_timeout: float,
    cleanup_timeout: float,
) -> list[RunResult]:
    """Replay a trajectory repeatedly and report flakiness."""
    if output_dir is None:
        if report:
            output_dir = report.parent
        else:
            output_dir = Path("out")

    kwargs = {
        "trajectory": str(trajectory),
        "verifier_path": str(verifier),
        "agent": agent,
        "runs": runs,
        "mode": mode,
        "max_agent_interventions": max_agent_interventions,
        "report": str(report) if report else None,
        "output_dir": str(output_dir),
        "action_timeout": action_timeout,
        "launch_timeout": launch_timeout,
        "replay_timeout": replay_timeout,
        "agent_timeout": agent_timeout,
        "cleanup_timeout": cleanup_timeout,
    }
    return asyncio.run(run_flakiness(**kwargs))


def main(args=None) -> int:
    try:
        results = cli(args=args, standalone_mode=False)
        # Click returns the exit code directly for eager options such as
        # ``--help`` when standalone mode is disabled. Only normal command
        # execution returns the list of run results.
        if isinstance(results, int):
            return results
        return 1 if any(not result.passed for result in results) else 0
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
