import csv
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from cua_driver.flakiness import VerificationResult, Verifier


class CsvEditVerifier(Verifier):
    def __init__(self) -> None:
        self.working_dir = Path(tempfile.gettempdir()) / f"cua-flake-csv-{uuid.uuid4().hex}"
        self.working_csv = self.working_dir / "data.csv"
        self.task_description = f"Open {self.working_csv}, change one value to 999, and save it."
        self.original_content = "id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n"

    async def reset(self, driver: Any) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.working_csv.write_text(self.original_content, encoding="utf-8")

    def prepare_action(self, action: dict) -> dict:
        if action.get("tool") == "launch_app":
            args = action.setdefault("arguments", {})
            args["path"] = str(self.working_csv)
            args["bounds"] = {"x": 0, "y": 0, "width": 1000, "height": 800}
        return action

    async def verify(self, driver: Any) -> VerificationResult:
        if not self.working_csv.exists():
            return VerificationResult.fail(
                f"Expected working CSV was not found: {self.working_csv}"
            )

        content = self.working_csv.read_text(encoding="utf-8")
        if content == self.original_content:
            return VerificationResult.fail("CSV file was not modified.")

        try:
            with self.working_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            return VerificationResult.fail(f"Failed to parse CSV: {exc}")

        if not rows:
            return VerificationResult.fail("CSV is empty after editing.")

        edited_cells = [
            {"id": row.get("id"), "column": key}
            for row in rows
            for key, value in row.items()
            if value == "999"
        ]
        if not edited_cells:
            return VerificationResult.fail(
                "The expected edited value 999 was not saved to the CSV.",
                {"expected_value": "999", "rows": rows},
            )

        return VerificationResult.pass_(
            {"path": str(self.working_csv), "rows": rows, "edited_cells": edited_cells}
        )

    async def teardown(self, driver: Any) -> None:
        # The runner owns and closes processes it launched. The verifier must not
        # kill unrelated user applications.
        shutil.rmtree(self.working_dir, ignore_errors=True)
