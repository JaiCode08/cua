import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

from cua_driver.flakiness import Verifier, VerificationResult


class ZipFolderVerifier(Verifier):
    def __init__(
        self,
        desktop: Path | None = None,
        verify_timeout: float = 10.0,
        verify_poll: float = 0.2,
    ) -> None:
        self.desktop = desktop or Path.home() / "Desktop"
        self.folder_path = self.desktop / "test_folder"
        self.zip_path = self.desktop / "test_folder.zip"
        self.verify_timeout = verify_timeout
        self.verify_poll = verify_poll
        self.task_description = (
            f"Create {self.folder_path}, add a text file, and create {self.zip_path}."
        )

    async def _cleanup_owned_apps(self, driver: Any) -> None:
        cleanup = getattr(driver, "cleanup_owned", None)
        if cleanup is None:
            return
        warnings = await cleanup()
        if warnings:
            raise RuntimeError(f"Owned application cleanup failed: {json.dumps(warnings)}")

    async def reset(self, driver: Any) -> None:
        # Idempotently close only apps registered as owned by this run. Never
        # kill all cmd, PowerShell, Explorer, or Nautilus processes by name.
        await self._cleanup_owned_apps(driver)

        self.desktop.mkdir(parents=True, exist_ok=True)
        for path in (self.zip_path, self.folder_path):
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)
            if path.exists():
                raise RuntimeError(f"Failed to reset task-owned path: {path}")

    def prepare_action(self, action: dict) -> dict:
        # This trajectory launches PowerShell and cmd.exe itself. Their returned
        # PIDs are safe for the replay executor to own and close after the run.
        if action.get("tool") == "launch_app":
            action["_owned_process"] = True
        return action

    async def verify(self, driver: Any) -> VerificationResult:
        deadline = time.monotonic() + self.verify_timeout
        saw_empty_zip = False

        while True:
            try:
                size = self.zip_path.stat().st_size
            except FileNotFoundError:
                size = None

            if size is not None and size > 0:
                return VerificationResult.pass_(details={"size": size})
            if size == 0:
                saw_empty_zip = True

            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(self.verify_poll)

        if saw_empty_zip:
            return VerificationResult.fail(
                reason=f"Zip file remained empty after {self.verify_timeout:g}s."
            )

        contents = [p.name for p in self.desktop.iterdir()]
        return VerificationResult.fail(
            reason=(
                "Expected ~/Desktop/test_folder.zip, not found after "
                f"{self.verify_timeout:g}s. Desktop contents: {contents}"
            ),
            details={"desktop_contents": contents},
        )

    async def teardown(self, driver: Any) -> None:
        await self._cleanup_owned_apps(driver)
