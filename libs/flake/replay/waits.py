import asyncio
import hashlib
import json
import logging
import subprocess
import time
import inspect
from pathlib import Path


async def wait_for_window(driver, title: str, timeout: float = 10.0, poll: float = 0.2) -> None:
    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for window '{title}' after {timeout}s")

        logging.debug(
            json.dumps({"event": "wait_poll", "type": "window", "title": title, "elapsed": elapsed})
        )

        try:
            windows = driver.list_windows()
            if inspect.isawaitable(windows):
                windows = await windows
            if windows is None:
                windows = []
        except Exception as e:
            logging.warning(json.dumps({"event": "list_windows_error", "error": str(e)}))
            windows = []
        for w in windows:
            if isinstance(w, str) and title in w:
                return
            elif title in (getattr(w, "title", None) or ""):
                return
            elif isinstance(w, dict) and title in w.get("title", ""):
                return

        await asyncio.sleep(poll)


async def wait_for_file(path: Path, timeout: float = 10.0, poll: float = 0.2) -> None:
    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            try:
                files_in_dir = list(path.parent.iterdir())
            except FileNotFoundError:
                files_in_dir = []
            raise TimeoutError(
                f"Timed out waiting for file '{path}' after {timeout}s. Files in dir: {files_in_dir}"
            )

        logging.debug(
            json.dumps(
                {"event": "wait_poll", "type": "file", "path": str(path), "elapsed": elapsed}
            )
        )

        if path.exists():
            return

        await asyncio.sleep(poll)


async def wait_for_screenshot_stable(driver, timeout: float = 10.0, poll: float = 0.2) -> None:
    start_time = time.monotonic()
    last_hash = None

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for stable screenshot after {timeout}s")

        logging.debug(
            json.dumps({"event": "wait_poll", "type": "screenshot_stable", "elapsed": elapsed})
        )

        screenshot = driver.screenshot()
        if inspect.isawaitable(screenshot):
            screenshot = await screenshot

        # Determine how to hash the screenshot based on its type (e.g. PIL.Image vs bytes)
        if hasattr(screenshot, "tobytes"):
            data = screenshot.tobytes()
        elif isinstance(screenshot, bytes):
            data = screenshot
        else:
            data = str(screenshot).encode("utf-8")

        current_hash = hashlib.md5(data).hexdigest()

        if last_hash is not None and current_hash == last_hash:
            return

        last_hash = current_hash
        await asyncio.sleep(poll)


async def wait_for_js(pid: int, expr: str, timeout: float = 10.0, poll: float = 0.2) -> None:
    try:
        from bench_ui import execute_javascript
    except ImportError:
        execute_javascript = None

    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for JS expression '{expr}' after {timeout}s")

        logging.debug(
            json.dumps(
                {"event": "wait_poll", "type": "js", "pid": pid, "expr": expr, "elapsed": elapsed}
            )
        )

        if execute_javascript and execute_javascript(pid, expr):
            return

        await asyncio.sleep(poll)


async def wait_for_subprocess(cmd: list[str], timeout: float = 10.0, poll: float = 0.2) -> None:
    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for subprocess '{cmd}' after {timeout}s")

        logging.debug(
            json.dumps({"event": "wait_poll", "type": "subprocess", "cmd": cmd, "elapsed": elapsed})
        )

        remaining = max(0.1, timeout - elapsed)
        try:
            res = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=min(2.0, remaining),
                check=False,
            )
        except subprocess.TimeoutExpired:
            res = None
        if res is not None and res.returncode == 0:
            return

        await asyncio.sleep(poll)
