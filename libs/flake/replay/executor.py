from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from PIL import Image


SUPPORTED_ACTIONS = {
    "click",
    "right_click",
    "double_click",
    "type_text",
    "hotkey",
    "press_key",
    "scroll",
    "drag",
    "launch_app",
    "set_value",
}
TARGETED_ACTIONS = SUPPORTED_ACTIONS - {"launch_app"}

NON_REPLAYABLE_TOOLS = {
    "start_session",
    "end_session",
    "set_recording",
    "start_recording",
    "stop_recording",
    "screenshot",
    "get_window_state",
    "replay_trajectory",
    "continue_replay",
}


@dataclass
class ActionResult:
    step: int
    ok: bool
    action_type: str
    duration_ms: int
    error: Optional[str] = None
    screenshot_path: Optional[str] = None
    accessibility_tree_path: Optional[str] = None
    warnings: list[dict] = field(default_factory=list)


@dataclass
class ContinueReplayStep:
    step: int
    cache_hit: bool
    tool: Optional[str]
    args: Optional[dict]
    duration_ms: int
    screenshot_path: str
    warnings: list[dict] = field(default_factory=list)
    capture_scope: str = "window"


@dataclass
class Observation:
    screenshot: Image.Image
    capture_scope: str
    timestamp: str
    warnings: list[dict] = field(default_factory=list)


class ObservationCaptureError(RuntimeError):
    def __init__(self, message: str, *, code: str = "screenshot_capture_failed"):
        super().__init__(message)
        self.warning = {"code": code, "message": message, "artifact": "screenshot"}


def _turn_dirs(trajectory_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in trajectory_dir.iterdir()
            if path.is_dir() and path.name.startswith("turn-")
        ),
        key=lambda path: path.name,
    )


def validate_trajectory(trajectory_dir: Path) -> None:
    if not trajectory_dir.is_dir():
        raise ValueError(f"Invalid trajectory: directory does not exist: {trajectory_dir}")
    turns = _turn_dirs(trajectory_dir)
    if not turns:
        raise ValueError(f"Invalid trajectory at {trajectory_dir}: no turn-NNNNN folders found")
    for turn in turns:
        if not (turn / "action.json").is_file():
            raise ValueError(f'Invalid trajectory at {turn}/: missing required file "action.json"')
        try:
            action = json.loads((turn / "action.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid trajectory at {turn}/action.json: {exc}") from exc
        if not isinstance(action.get("tool"), str) or not isinstance(
            action.get("arguments", {}), dict
        ):
            raise ValueError(
                f"Invalid trajectory at {turn}/action.json: expected tool string and arguments object"
            )
        if action.get("tool") in NON_REPLAYABLE_TOOLS:
            continue
        if action.get("tool") not in SUPPORTED_ACTIONS:
            raise ValueError(
                f"Invalid trajectory at {turn}/action.json: tool '{action.get('tool')}' is not supported for replay"
            )
        normalize_replay_action(action, source=str(turn / "action.json"))


def normalize_replay_action(action: dict, *, source: str = "action") -> dict:
    """Return a session-independent action without destroying its target."""
    normalized = copy.deepcopy(action)
    arguments = normalized.setdefault("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Invalid {source}: arguments must be an object")

    has_unstable_element = "element_index" in arguments or "element_token" in arguments
    has_coordinates = all(
        isinstance(arguments.get(key), (int, float)) and not isinstance(arguments.get(key), bool)
        for key in ("x", "y")
    )
    if has_unstable_element and not has_coordinates:
        raise ValueError(f"Invalid {source}: element-indexed action has no x/y coordinate fallback")
    if has_unstable_element:
        arguments.pop("element_index", None)
        arguments.pop("element_token", None)

    arguments.pop("window_id", None)
    arguments.pop("pid", None)
    return normalized


def load_trajectory(trajectory_dir: Path) -> list[dict]:
    actions = []
    for turn in _turn_dirs(trajectory_dir):
        action = json.loads((turn / "action.json").read_text(encoding="utf-8"))
        if action.get("tool") in NON_REPLAYABLE_TOOLS:
            continue
        actions.append(normalize_replay_action(action, source=str(turn / "action.json")))
    return actions


def _extract_payload(stdout: str) -> dict:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return {"text": stdout}
    if isinstance(value, dict) and isinstance(value.get("structuredContent"), dict):
        return value["structuredContent"]
    if isinstance(value, dict):
        return value
    raise ValueError("cua-driver returned a non-object response")


def _decode_screenshot(state: dict) -> Optional[Image.Image]:
    encoded = state.get("screenshot_png_b64") or state.get("_legacy_screenshot_png_b64")
    if not isinstance(encoded, str) or not encoded:
        return None
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.load()
    return image.convert("RGB")


class ReplayExecutor:
    def __init__(
        self,
        trajectory_dir: Path,
        action_timeout: float = 15.0,
        launch_timeout: float = 30.0,
        replay_timeout: float = 600.0,
        action_transform: Optional[Callable[[dict], dict]] = None,
        driver_command: Optional[str] = None,
    ):
        self.trajectory_dir = trajectory_dir
        self.action_timeout = action_timeout
        self.launch_timeout = launch_timeout
        self.replay_timeout = replay_timeout
        self.action_transform = action_transform
        sibling_driver = Path(sys.executable).with_name(
            "cua-driver.exe" if os.name == "nt" else "cua-driver"
        )
        self.driver_command = driver_command or (
            str(sibling_driver)
            if sibling_driver.exists()
            else shutil.which("cua-driver") or "cua-driver"
        )
        self.actions: list[dict] = []
        self._processes: set[asyncio.subprocess.Process] = set()
        self._recorded_pid_map: dict[int, int] = {}
        self._active_pid: Optional[int] = None
        self._active_window_id: Optional[int] = None
        self._owned_pids: set[int] = set()
        self._active_sessions: set[str] = set()
        self._session_map: dict[str, str] = {}

    async def setup(self) -> None:
        validate_trajectory(self.trajectory_dir)
        self.actions = load_trajectory(self.trajectory_dir)
        await self._refresh_target()

    async def call_tool(self, tool: str, arguments: dict, timeout: Optional[float] = None) -> dict:
        command = [self.driver_command, "call", tool, json.dumps(arguments, separators=(",", ":"))]
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        self._processes.add(proc)
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout or self.action_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"{tool} timed out after {timeout or self.action_timeout}s")
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
        finally:
            self._processes.discard(proc)
        text = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(err or text or f"{tool} exited with code {proc.returncode}")
        return _extract_payload(text)

    async def list_windows(self, pid: Optional[int] = None) -> list[dict]:
        payload = await self.call_tool("list_windows", {"pid": pid} if pid else {})
        return payload.get("windows", payload.get("_legacy_windows", []))

    async def _refresh_target(self, preferred_pid: Optional[int] = None) -> None:
        windows = await self.list_windows(preferred_pid)
        if not windows and preferred_pid:
            windows = await self.list_windows()
        windows = [
            window for window in windows if window.get("title") != "Cua.AgentCursorOverlay.default"
        ]
        if not windows:
            return
        # Match the recorder: list_windows is ordered front-to-back and the
        # recording hook captures the first window for a pid.
        target = windows[0]
        self._active_pid = int(target["pid"])
        self._active_window_id = int(target["window_id"])

    def adopt_window(self, window: dict, *, owned: bool = False) -> None:
        self._active_pid = int(window["pid"])
        self._active_window_id = int(window["window_id"])
        if owned:
            self._owned_pids.add(self._active_pid)

    async def adopt_pid(self, pid: int, *, owned: bool = False) -> None:
        self._active_pid = int(pid)
        self._active_window_id = None
        if owned:
            self._owned_pids.add(self._active_pid)
        windows = await self.list_windows(self._active_pid)
        if windows:
            # Match the recorder's first-window selection for a pid.
            target = windows[0]
            self._active_window_id = int(target["window_id"])

    async def register_launch_payload(self, payload: dict, *, owned: bool = True) -> None:
        pid = payload.get("pid") or payload.get("process_id")
        if isinstance(pid, int):
            await self.adopt_pid(pid, owned=owned)
            return
        windows = payload.get("windows", [])
        if windows:
            self.adopt_window(windows[0], owned=owned)
        else:
            await self._refresh_target()

    def _prepare_action(self, action: dict) -> dict:
        prepared = copy.deepcopy(action)
        if self.action_transform:
            prepared = self.action_transform(prepared)
        args = prepared.setdefault("arguments", {})
        recorded_session = args.get("session")
        if isinstance(recorded_session, str):
            args["session"] = self._session_map.setdefault(
                recorded_session, f"{recorded_session}-replay-{uuid.uuid4().hex[:8]}"
            )
        recorded_pid = args.get("pid")
        if prepared.get("tool") in TARGETED_ACTIONS:
            if isinstance(recorded_pid, int) and recorded_pid in self._recorded_pid_map:
                args["pid"] = self._recorded_pid_map[recorded_pid]
            elif self._active_pid is not None:
                args["pid"] = self._active_pid
            if self._active_window_id is not None and (
                "window_id" in args or "element_index" in args or "element_token" in args
            ):
                args["window_id"] = self._active_window_id
        return prepared

    async def execute_action(self, action: dict, step: int, output_dir: Path) -> ActionResult:
        original = copy.deepcopy(action)
        prepared = self._prepare_action(action)
        tool = prepared.get("tool", "")
        owned_process = bool(prepared.pop("_owned_process", True))
        args = prepared.get("arguments", {})
        start = time.monotonic()
        error: Optional[str] = None
        payload: dict = {}

        if tool not in SUPPORTED_ACTIONS:
            error = f"Unsupported recorded action: {tool}"
        elif "element_index" in args or "element_token" in args:
            error = "cache_miss: unstable element reference cannot replay deterministically"
        else:
            try:
                payload = await self.call_tool(
                    tool,
                    args,
                    self.launch_timeout if tool == "launch_app" else self.action_timeout,
                )
                if tool == "launch_app":
                    live_pid = payload.get("pid") or payload.get("process_id")
                    recorded_pid = original.get("arguments", {}).get("pid") or original.get("pid")
                    if isinstance(live_pid, int) and isinstance(recorded_pid, int):
                        self._recorded_pid_map[recorded_pid] = live_pid
                    await self.register_launch_payload(payload, owned=owned_process)
            except Exception as exc:
                error = str(exc)

        turn_dir = output_dir / f"turn-{step:05d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "action.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
        screenshot_path, tree_path, capture_warnings = await self.capture_artifacts(turn_dir)
        return ActionResult(
            step=step,
            ok=error is None,
            action_type=tool,
            duration_ms=int((time.monotonic() - start) * 1000),
            error=error,
            screenshot_path=screenshot_path,
            accessibility_tree_path=tree_path,
            warnings=capture_warnings,
        )

    async def _capture_window_state(self, capture_mode: str) -> dict:
        if self._active_pid is None or self._active_window_id is None:
            await self._refresh_target()
        if self._active_pid is None or self._active_window_id is None:
            raise ObservationCaptureError("No active target window is available for capture")
        return await self.call_tool(
            "get_window_state",
            {
                "pid": self._active_pid,
                "window_id": self._active_window_id,
                "capture_mode": capture_mode,
            },
        )

    async def _capture_desktop_image(self) -> Image.Image:
        """Capture the display through the recorder-compatible screenshot tool."""
        state = await self.call_tool("screenshot", {"format": "png"})
        image = _decode_screenshot(state)
        if image is None:
            raise ObservationCaptureError(
                "screenshot returned no full-display image",
                code="screenshot_missing",
            )
        return image.convert("RGB").copy()

    async def capture_observation(self, capture_scope: str = "window") -> Observation:
        """Capture an observation in the same coordinate domain as its recording."""
        if capture_scope not in {"window", "desktop"}:
            raise ObservationCaptureError(
                f"Unsupported capture scope: {capture_scope}",
                code="capture_scope_invalid",
            )
        if capture_scope == "desktop":
            warnings: list[dict] = []
            try:
                image = await self._capture_desktop_image()
            except Exception as exc:
                try:
                    from PIL import ImageGrab

                    image = await asyncio.to_thread(ImageGrab.grab, all_screens=True)
                except Exception as fallback_exc:
                    raise ObservationCaptureError(
                        "Desktop screenshot capture failed through both cua-driver "
                        f"({exc}) and ImageGrab ({fallback_exc})"
                    ) from fallback_exc
                warnings.append(
                    {
                        "code": "desktop_capture_fallback",
                        "message": (
                            "cua-driver desktop capture was unavailable; used full-desktop "
                            f"ImageGrab for a desktop-scoped recording: {exc}"
                        ),
                        "artifact": "screenshot",
                    }
                )
            return Observation(
                screenshot=image,
                capture_scope="desktop",
                timestamp=datetime.now(timezone.utc).isoformat(),
                warnings=warnings,
            )

        try:
            state = await self._capture_window_state("vision")
            image = _decode_screenshot(state)
        except ObservationCaptureError:
            raise
        except Exception as exc:
            raise ObservationCaptureError(
                f"Target-window screenshot capture failed: {exc}"
            ) from exc
        if image is None:
            raise ObservationCaptureError(
                "get_window_state returned no target-window screenshot",
                code="screenshot_missing",
            )
        return Observation(
            screenshot=image,
            capture_scope="window",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def capture_artifacts(
        self, turn_dir: Path
    ) -> tuple[Optional[str], Optional[str], list[dict]]:
        screenshot_path: Optional[str] = None
        tree_path: Optional[str] = None
        warnings: list[dict] = []
        try:
            state = await self._capture_window_state("som")
        except Exception as exc:
            message = f"Window state capture failed: {exc}"
            warnings.extend(
                [
                    {
                        "code": "screenshot_capture_failed",
                        "message": message,
                        "artifact": "screenshot",
                    },
                    {
                        "code": "accessibility_capture_failed",
                        "message": message,
                        "artifact": "app_state",
                    },
                ]
            )
            return screenshot_path, tree_path, warnings

        try:
            tree_file = turn_dir / "app_state.json"
            tree_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tree_path = str(tree_file)
        except Exception as exc:
            warnings.append(
                {
                    "code": "accessibility_capture_failed",
                    "message": f"Could not save accessibility state: {exc}",
                    "artifact": "app_state",
                }
            )

        try:
            image = _decode_screenshot(state)
            if image is None:
                raise ValueError("get_window_state returned no screenshot bytes")
            screenshot_file = turn_dir / "screenshot.png"
            image.save(screenshot_file)
            screenshot_path = str(screenshot_file)
        except Exception as exc:
            warnings.append(
                {
                    "code": "screenshot_capture_failed",
                    "message": f"Could not save target-window screenshot: {exc}",
                    "artifact": "screenshot",
                }
            )
        return screenshot_path, tree_path, warnings

    async def teardown(self) -> list[dict]:
        """Clean resources owned by this executor and return structured warnings."""
        warnings: list[dict] = []
        for session in list(self._active_sessions):
            try:
                await self.call_tool("end_session", {"session": session}, self.action_timeout)
            except Exception as exc:
                warnings.append(
                    {"code": "session_cleanup_failed", "message": str(exc), "session": session}
                )
        self._active_sessions.clear()

        for pid in sorted(self._owned_pids):
            try:
                if sys.platform == "win32":
                    try:
                        await self.call_tool(
                            "hotkey",
                            {"pid": pid, "keys": ["alt", "f4"]},
                            min(self.action_timeout, 3.0),
                        )
                        await asyncio.sleep(0.25)
                    except Exception:
                        # Force cleanup below remains scoped to this owned PID.
                        pass
                    try:
                        await self.call_tool(
                            "kill_app", {"pid": pid}, min(self.action_timeout, 5.0)
                        )
                    except Exception as driver_error:
                        import subprocess

                        result = subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=False,
                        )
                        if result.returncode not in (0, 128):
                            detail = result.stderr.strip() or result.stdout.strip()
                            raise RuntimeError(
                                f"driver cleanup failed: {driver_error}; "
                                f"taskkill fallback failed: {detail}"
                            )
                else:
                    import signal

                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as exc:
                warnings.append({"code": "process_cleanup_failed", "message": str(exc), "pid": pid})
        self._owned_pids.clear()

        processes = list(self._processes)
        for proc in processes:
            if proc.returncode is None:
                try:
                    proc.kill()
                except Exception as exc:
                    warnings.append({"code": "subprocess_cleanup_failed", "message": str(exc)})
        if processes:
            await asyncio.gather(*(proc.wait() for proc in processes), return_exceptions=True)
        self._processes.clear()
        return warnings
