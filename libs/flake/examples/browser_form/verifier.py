import asyncio
import ctypes
import sys
from pathlib import Path
from typing import Any

from bench_ui import execute_javascript
from cua_driver.flakiness import VerificationResult, Verifier


class BrowserFormVerifier(Verifier):
    task_description = (
        "Fill the local Contact Us form with name 'John Doe', email "
        "'john@test.com', and comment 'hello world', then submit it."
    )

    def __init__(self) -> None:
        self.pid: int | None = None

    async def reset(self, driver: Any) -> None:
        from bench_ui import launch_window
        from replay.waits import wait_for_screenshot_stable

        html_content = (Path(__file__).parent / "task.html").read_text(encoding="utf-8")
        self.pid = launch_window(
            html=html_content, title="Form Task", x=0, y=0, width=804, height=809
        )
        await driver.adopt_pid(self.pid, owned=True)
        for _ in range(50):
            windows = await driver.list_windows(self.pid)
            matches = [w for w in windows if "Form Task" in w.get("title", "")]
            if matches:
                break
            await asyncio.sleep(0.2)
        else:
            raise TimeoutError(f"Form Task window for owned pid {self.pid} did not appear")
        if sys.platform == "win32":
            hwnd = int(matches[0]["window_id"])
            swp_no_zorder = 0x0004
            swp_no_activate = 0x0010
            ctypes.windll.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
            resized = ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 1000, 1000, swp_no_zorder | swp_no_activate
            )
            if not resized:
                raise OSError(f"Failed to size owned Form Task window {hwnd} deterministically")
        await driver.adopt_pid(self.pid, owned=True)
        await driver.call_tool(
            "bring_to_front",
            {"pid": self.pid, "window_id": int(matches[0]["window_id"])},
        )
        await wait_for_screenshot_stable(driver, timeout=10)

    async def verify(self, driver: Any) -> VerificationResult:
        windows = await driver.list_windows()
        matches = [w for w in windows if "Form Task" in w.get("title", "")]
        if not matches:
            return VerificationResult.fail("Window 'Form Task' is not found")
        pid = int(matches[0]["pid"])

        try:
            clicked = execute_javascript(pid, "window.__submitted === true")
            form_data = execute_javascript(pid, "window.__formData")
        except Exception as e:
            return VerificationResult.fail(f"Failed to query javascript state: {e}", {"pid": pid})

        if not clicked:
            return VerificationResult.fail(
                "Submit button was not clicked or form was not submitted", {"pid": pid}
            )

        expected = {"name": "John Doe", "email": "john@test.com", "comment": "hello world"}

        if not form_data:
            return VerificationResult.fail(
                "Form data was not found in javascript state", {"pid": pid}
            )

        missing_or_wrong = {}
        for k, v in expected.items():
            if form_data.get(k) != v:
                missing_or_wrong[k] = {"expected": v, "actual": form_data.get(k)}

        if missing_or_wrong:
            return VerificationResult.fail(
                "Form data did not match expected values", {"pid": pid, "diff": missing_or_wrong}
            )

        return VerificationResult.pass_({"pid": pid})

    async def teardown(self, driver: Any) -> None:
        # The runner closes the exact PID returned by launch_window.
        return None
