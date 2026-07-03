import dataclasses
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader

from flakiness.classifier import FailureInfo
from replay.executor import ActionResult, ContinueReplayStep
from verifiers.base import VerificationResult


def _artifact_href(path: str | Path, report_dir: Path) -> str:
    """Return a browser-safe link relative to the actual report location."""
    relative = Path(os.path.relpath(Path(path).resolve(), report_dir.resolve())).as_posix()
    return quote(relative, safe="/:")


@dataclass
class RunResult:
    run_index: int
    passed: bool
    failure: Optional[FailureInfo]
    steps: List[ContinueReplayStep]
    action_results: List[ActionResult]
    verifier_result: Optional[VerificationResult]
    cache_hits: int
    cache_misses: int
    agent_interventions: int
    duration_ms: int
    log_path: str
    screenshot_dir: str


def generate_report(
    results: List[RunResult],
    output_dir: Path,
    task_id: str,
    trajectory_path: str,
    verifier_path: str,
    report_file: Optional[Path] = None,
    mode: str = "forgiving-perceptual",
    agent: Optional[str] = None,
    max_agent_interventions: int = 5,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = len(results)
    passes = sum(1 for r in results if r.passed)
    failures = runs - passes
    flake_rate = failures / runs if runs > 0 else 0.0

    total_hits = sum(r.cache_hits for r in results)
    total_misses = sum(r.cache_misses for r in results)
    cache_hit_rate = (
        total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
    )

    agent_interventions_total = sum(r.agent_interventions for r in results)

    failure_categories = {}
    for r in results:
        if not r.passed and r.failure:
            cat = r.failure.category
            failure_categories[cat] = failure_categories.get(cat, 0) + 1

    first_failed_result = next((result for result in results if not result.passed), None)
    first_failure = None
    if first_failed_result and first_failed_result.failure:
        first_failure = {
            "run": first_failed_result.run_index,
            "category": first_failed_result.failure.category,
            "reason": first_failed_result.failure.reason,
            "details": first_failed_result.failure.detail,
        }

    summary = {
        "task_id": task_id,
        "runs": runs,
        "passes": passes,
        "failures": failures,
        "flake_rate": flake_rate,
        "cache_hit_rate": cache_hit_rate,
        "agent_interventions_total": agent_interventions_total,
        "cache_hits": total_hits,
        "cache_misses": total_misses,
        "failure_categories": failure_categories,
        "first_failure": first_failure,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")), autoescape=True
    )

    template = env.get_template("report.html.j2")

    report_target = report_file if report_file else output_dir / "report.html"
    report_dir = Path(report_target).resolve().parent
    repro_cmd = (
        f'cua-driver flakiness --trajectory "{trajectory_path}" --verifier "{verifier_path}" '
        f"--runs {runs} --mode {mode} --max-agent-interventions {max_agent_interventions} "
        f'--report "{report_target}"'
    )
    if agent and mode != "strict":
        repro_cmd += f' --agent "{agent}"'

    for r in results:
        run_dir = Path(r.screenshot_dir)
        r.timeline = [{"kind": "replay_step", **dataclasses.asdict(step)} for step in r.steps] + [
            {"kind": "action_result", **dataclasses.asdict(result)} for result in r.action_results
        ]
        r.run_dir_link = _artifact_href(run_dir, report_dir)
        r.log_link = _artifact_href(r.log_path, report_dir)
        verifier_artifact = run_dir / "verifier_result.json"
        r.verifier_link = (
            _artifact_href(verifier_artifact, report_dir) if verifier_artifact.exists() else ""
        )

    failed_runs = []
    for r in results:
        if not r.passed:
            failing_step = None
            if r.action_results:
                for a in r.action_results:
                    if not a.ok:
                        failing_step = a.step
                        break

            before_img = ""
            after_img = ""

            if failing_step is not None:
                before_idx = max(1, failing_step - 1)
                before_result = next(
                    (item for item in r.action_results if item.step == before_idx), None
                )
                after_result = next(
                    (item for item in r.action_results if item.step == failing_step), None
                )
                before_path = (
                    Path(before_result.screenshot_path)
                    if before_result and before_result.screenshot_path
                    else None
                )
                after_path = (
                    Path(after_result.screenshot_path)
                    if after_result and after_result.screenshot_path
                    else None
                )
                if after_path is None:
                    failed_step = next(
                        (item for item in r.steps if item.step == failing_step), None
                    )
                    if failed_step and failed_step.screenshot_path:
                        after_path = Path(failed_step.screenshot_path)

                # Check if screenshots exist and create relative paths for HTML
                if before_path and before_path.exists():
                    before_img = _artifact_href(before_path, report_dir)
                if after_path and after_path.exists():
                    after_img = _artifact_href(after_path, report_dir)

            elif r.verifier_result and not r.verifier_result.passed and r.action_results:
                failing_step = r.action_results[-1].step

                before_idx = max(1, failing_step - 1)
                before_result = next(
                    (item for item in r.action_results if item.step == before_idx), None
                )
                before_path = (
                    Path(before_result.screenshot_path)
                    if before_result and before_result.screenshot_path
                    else None
                )
                if before_path and before_path.exists():
                    before_img = _artifact_href(before_path, report_dir)

                last_result = r.action_results[-1]
                after_path = (
                    Path(last_result.screenshot_path) if last_result.screenshot_path else None
                )
                if after_path and after_path.exists():
                    after_img = _artifact_href(after_path, report_dir)

            failed_runs.append(
                {
                    "run_index": r.run_index,
                    "failing_step": failing_step,
                    "failure": r.failure,
                    "verifier_result": r.verifier_result,
                    "before_img": before_img,
                    "after_img": after_img,
                    "log_path": _artifact_href(r.log_path, report_dir),
                }
            )

    metadata = (
        json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        if (output_dir / "metadata.json").exists()
        else {}
    )
    html_content = template.render(
        task_id=task_id,
        trajectory_path=trajectory_path,
        verifier_path=verifier_path,
        date=datetime.now(timezone.utc).isoformat(),
        os_info=platform.platform(),
        image_info=metadata.get("image", "unknown"),
        python_version=platform.python_version(),
        cua_driver_version=metadata.get("cua_driver_version", "unknown"),
        summary=summary,
        results=results,
        failed_runs=failed_runs,
        repro_cmd=repro_cmd,
    )

    report_path = report_file if report_file else output_dir / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path
