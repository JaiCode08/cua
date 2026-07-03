import json
from flakiness.reporter import generate_report, RunResult
from flakiness.classifier import FailureInfo
from replay.executor import ActionResult
from verifiers.base import VerificationResult


def test_generate_report_success_and_failure(tmp_path):
    results = [
        RunResult(
            run_index=1,
            passed=True,
            failure=None,
            steps=[],
            action_results=[ActionResult(step=1, ok=True, action_type="click", duration_ms=10)],
            verifier_result=VerificationResult.pass_(),
            cache_hits=1,
            cache_misses=0,
            agent_interventions=0,
            duration_ms=100,
            log_path="log1.txt",
            screenshot_dir="run1",
        ),
        RunResult(
            run_index=2,
            passed=False,
            failure=FailureInfo(category="verifier_failed", reason="missing element", detail={}),
            steps=[],
            action_results=[ActionResult(step=1, ok=True, action_type="click", duration_ms=10)],
            verifier_result=VerificationResult.fail("missing element"),
            cache_hits=0,
            cache_misses=1,
            agent_interventions=1,
            duration_ms=200,
            log_path="log2.txt",
            screenshot_dir="run2",
        ),
    ]

    report_path = generate_report(
        results=results,
        output_dir=tmp_path,
        task_id="test_task_123",
        trajectory_path="path/to/traj",
        verifier_path="path/to/verifier.py",
    )

    assert report_path.exists()
    assert (tmp_path / "summary.json").exists()

    with open(tmp_path / "summary.json") as f:
        summary = json.load(f)

    assert summary["runs"] == 2
    assert summary["passes"] == 1
    assert summary["failures"] == 1
    assert summary["flake_rate"] == 0.5
    assert summary["cache_hit_rate"] == 0.5
    assert summary["failure_categories"]["verifier_failed"] == 1
    assert summary["first_failure"]["run"] == 2
    assert summary["first_failure"]["category"] == "verifier_failed"

    with open(report_path) as f:
        html = f.read()

    assert "Flakiness Report: test_task_123" in html
    assert "Failure Details" in html
    assert "verifier_failed" in html
    assert "<td>2</td>" in html  # run count
    assert "run directory" in html
    assert "log.jsonl" in html
    assert "--mode forgiving-perceptual" in html
    assert "--max-agent-interventions 5" in html
