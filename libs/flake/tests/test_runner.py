import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from flakiness import runner
from replay.executor import ActionResult, ContinueReplayStep
from verifiers.base import VerificationResult, Verifier


class DummyProcess:
    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class FakeExecutor:
    def __init__(self, trajectory_dir, *args, **kwargs):
        self.trajectory_dir = trajectory_dir
        self.action_timeout = kwargs.get("action_timeout", 15)
        self.launch_timeout = kwargs.get("launch_timeout", 30)
        self.actions = []

    async def setup(self):
        pass

    async def teardown(self):
        pass

    async def execute_action(self, action, step, output_dir):
        turn = output_dir / f"turn-{step:05d}"
        turn.mkdir(parents=True, exist_ok=True)
        (turn / "action.json").write_text(json.dumps(action), encoding="utf-8")
        screenshot = turn / "screenshot.png"
        Image.new("RGB", (20, 20), "white").save(screenshot)
        return ActionResult(step, True, action["tool"], 1, screenshot_path=str(screenshot))

    async def capture_artifacts(self, turn_dir):
        turn_dir.mkdir(parents=True, exist_ok=True)
        screenshot = turn_dir / "screenshot.png"
        Image.new("RGB", (20, 20), "white").save(screenshot)
        return str(screenshot), None, []

    async def call_tool(self, *args, **kwargs):
        return {}

    async def capture_observation(self, capture_scope="window"):
        return {"screenshot": Image.new("RGB", (20, 20), "white")}


class CountingVerifier(Verifier):
    def __init__(self, *, verify_error=None, reset_error=None):
        self.reset_calls = 0
        self.verify_calls = 0
        self.verify_error = verify_error
        self.reset_error = reset_error

    async def reset(self, driver):
        self.reset_calls += 1
        if self.reset_error:
            raise self.reset_error

    async def verify(self, driver):
        self.verify_calls += 1
        if self.verify_error:
            raise self.verify_error
        return VerificationResult.fail("task incomplete", {"checked": True})


def test_main_returns_nonzero_when_any_run_fails(monkeypatch):
    monkeypatch.setattr(
        runner,
        "cli",
        lambda **kwargs: [SimpleNamespace(passed=True), SimpleNamespace(passed=False)],
    )

    assert runner.main([]) == 1


def test_main_returns_zero_when_all_runs_pass(monkeypatch):
    monkeypatch.setattr(
        runner,
        "cli",
        lambda **kwargs: [SimpleNamespace(passed=True), SimpleNamespace(passed=True)],
    )

    assert runner.main([]) == 0


def test_main_help_returns_zero(capsys):
    assert runner.main(["--help"]) == 0
    assert "Replay a trajectory repeatedly" in capsys.readouterr().out


def test_agent_type_alias_normalizes_to_replayable_type_text():
    action = runner._normalize_agent_action(
        {"tool": "type", "arguments": {"text": "echo hello", "dispatch": "foreground"}},
        None,
    )

    assert action == {"tool": "type_text", "arguments": {"text": "echo hello"}}
    assert runner._recorded_action_failed(
        {"result_summary": "Foreground swap to target HWND was rejected by Windows"}
    )
    assert not runner._recorded_action_failed(
        {"result_summary": "Typed 10 characters successfully"}
    )


def _trajectory(path: Path):
    turn = path / "turn-00001"
    turn.mkdir(parents=True)
    (turn / "action.json").write_text(
        json.dumps({"tool": "click", "arguments": {"x": 1, "y": 1}}), encoding="utf-8"
    )


def _patch_runtime(monkeypatch, verifier):
    monkeypatch.setattr(runner, "ReplayExecutor", FakeExecutor)
    monkeypatch.setattr(runner, "load_verifier", lambda path: verifier)
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="")
    )
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: DummyProcess())
    monkeypatch.setattr(runner, "get_environment_metadata", lambda *a: {})

    async def stable(*args, **kwargs):
        return None

    monkeypatch.setattr(runner, "wait_for_screenshot_stable", stable)


async def _run(tmp_path, **kwargs):
    trajectory = tmp_path / "trajectory"
    _trajectory(trajectory)
    verifier_file = tmp_path / "verifier.py"
    verifier_file.write_text("# mocked", encoding="utf-8")
    return await runner.run_flakiness(
        trajectory=str(trajectory),
        verifier_path=str(verifier_file),
        agent="fake-agent",
        runs=kwargs.pop("runs", 1),
        mode=kwargs.pop("mode", "strict"),
        max_agent_interventions=kwargs.pop("max_agent_interventions", 0),
        output_dir=str(tmp_path / "out"),
        action_timeout=1,
        launch_timeout=1,
        replay_timeout=5,
        agent_timeout=1,
        cleanup_timeout=1,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_strict_miss_never_calls_agent_and_verifies_after_failure(tmp_path, monkeypatch):
    verifier = CountingVerifier()
    _patch_runtime(monkeypatch, verifier)
    agent_called = False

    async def fake_agent(*args, **kwargs):
        nonlocal agent_called
        agent_called = True
        raise AssertionError("strict mode invoked agent")

    async def miss(executor, cache, run_dir, step, last_action):
        miss_path = run_dir / f"turn-{step:05d}" / "screenshot_miss.png"
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20)).save(miss_path)
        return (
            "cache_miss",
            step,
            last_action,
            [],
            [ContinueReplayStep(step, False, None, None, 0, str(miss_path))],
        )

    monkeypatch.setattr(runner, "_run_agent", fake_agent)
    monkeypatch.setattr(runner, "continue_replay", miss)
    results = await _run(tmp_path)

    assert not agent_called
    assert verifier.reset_calls == 1
    assert verifier.verify_calls == 1
    assert results[0].failure.category == "cache_miss_unrecovered"
    assert results[0].verifier_result.details == {"checked": True}
    report = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "screenshot_miss.png" in report


@pytest.mark.asyncio
async def test_verify_exception_is_structured_without_replacing_primary_failure(
    tmp_path, monkeypatch
):
    verifier = CountingVerifier(verify_error=RuntimeError("verifier exploded"))
    _patch_runtime(monkeypatch, verifier)

    async def replay_error(executor, cache, run_dir, step, last_action):
        failed = ActionResult(step, False, "click", 1, "dispatch broke")
        return "error", step, last_action, [failed], []

    monkeypatch.setattr(runner, "continue_replay", replay_error)
    results = await _run(tmp_path)

    assert verifier.verify_calls == 1
    assert results[0].failure.category == "action_dispatch_failed"
    assert results[0].verifier_result.passed is False
    assert "verifier exploded" in results[0].verifier_result.reason


@pytest.mark.asyncio
async def test_forgiving_novel_agent_action_advances_replay_state(tmp_path, monkeypatch):
    verifier = CountingVerifier()
    _patch_runtime(monkeypatch, verifier)
    seen_last_actions = []

    async def sequence(executor, cache, run_dir, step, last_action):
        seen_last_actions.append(last_action)
        if len(seen_last_actions) == 1:
            miss_path = run_dir / f"turn-{step:05d}" / "screenshot_miss.png"
            miss_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), "white").save(miss_path)
            return (
                "cache_miss",
                step,
                last_action,
                [],
                [ContinueReplayStep(step, False, None, None, 0, str(miss_path))],
            )
        return "replay_complete", step, last_action, [], []

    async def agent(*args, **kwargs):
        return {"tool": "hotkey", "arguments": {"keys": ["ctrl", "s"]}}, 2, "{}"

    monkeypatch.setattr(runner, "continue_replay", sequence)
    monkeypatch.setattr(runner, "_run_agent", agent)
    results = await _run(tmp_path, mode="forgiving-perceptual", max_agent_interventions=2)

    assert len(seen_last_actions) == 2
    assert seen_last_actions[1]["tool"] == "hotkey"
    assert results[0].agent_interventions == 1


@pytest.mark.asyncio
async def test_intervention_cap_prevents_agent_call(tmp_path, monkeypatch):
    verifier = CountingVerifier()
    _patch_runtime(monkeypatch, verifier)

    async def miss(executor, cache, run_dir, step, last_action):
        path = run_dir / "miss.png"
        Image.new("RGB", (10, 10)).save(path)
        return (
            "cache_miss",
            step,
            last_action,
            [],
            [ContinueReplayStep(step, False, None, None, 0, str(path))],
        )

    async def forbidden(*args, **kwargs):
        raise AssertionError("agent called above cap")

    monkeypatch.setattr(runner, "continue_replay", miss)
    monkeypatch.setattr(runner, "_run_agent", forbidden)
    results = await _run(tmp_path, mode="forgiving-crop", max_agent_interventions=0)
    assert results[0].agent_interventions == 0
    assert results[0].failure.category == "cache_miss_unrecovered"


@pytest.mark.asyncio
async def test_verify_not_called_when_reset_fails(tmp_path, monkeypatch):
    verifier = CountingVerifier(reset_error=RuntimeError("reset exploded"))
    _patch_runtime(monkeypatch, verifier)

    async def should_not_replay(*args, **kwargs):
        raise AssertionError("replay ran after reset failure")

    monkeypatch.setattr(runner, "continue_replay", should_not_replay)
    results = await _run(tmp_path)
    assert verifier.reset_calls == 1
    assert verifier.verify_calls == 0
    assert results[0].failure.category == "reset_failed"


@pytest.mark.asyncio
async def test_multi_run_resets_and_verifies_each_run(tmp_path, monkeypatch):
    verifier = CountingVerifier()
    _patch_runtime(monkeypatch, verifier)

    async def complete(executor, cache, run_dir, step, last_action):
        return "replay_complete", step, last_action, [], []

    monkeypatch.setattr(runner, "continue_replay", complete)
    results = await _run(tmp_path, runs=3)
    assert len(results) == 3
    assert verifier.reset_calls == 3
    assert verifier.verify_calls == 3


class PassingVerifier(CountingVerifier):
    async def verify(self, driver):
        self.verify_calls += 1
        return VerificationResult.pass_({"checked": True})


@pytest.mark.asyncio
async def test_agent_done_terminal_reduces_interventions_across_runs(tmp_path, monkeypatch):
    verifier = PassingVerifier()
    _patch_runtime(monkeypatch, verifier)
    recovery = {"tool": "hotkey", "arguments": {"keys": ["ctrl", "s"]}}
    image = Image.new("RGB", (20, 20), "green")
    agent_calls = 0
    replay_calls = 0

    async def replay(executor, cache, run_dir, step, last_action):
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls == 1:
            miss_path = run_dir / f"turn-{step:05d}" / "screenshot_miss.png"
            miss_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(miss_path)
            return (
                "cache_miss",
                step,
                recovery,
                [],
                [ContinueReplayStep(step, False, None, None, 0, str(miss_path))],
            )
        assert cache.is_terminal(recovery, image, "window")
        return "replay_complete", step, recovery, [], []

    async def done(*args, **kwargs):
        nonlocal agent_calls
        agent_calls += 1
        return {"tool": "done", "arguments": {}}, 1, "{}"

    monkeypatch.setattr(runner, "continue_replay", replay)
    monkeypatch.setattr(runner, "_run_agent", done)
    results = await _run(
        tmp_path,
        runs=2,
        mode="forgiving-perceptual",
        max_agent_interventions=2,
    )
    assert [result.agent_interventions for result in results] == [1, 0]
    assert agent_calls == 1
    assert verifier.verify_calls == 2


class CleanupFailingVerifier(PassingVerifier):
    async def teardown(self, driver):
        raise RuntimeError("cleanup exploded")


@pytest.mark.asyncio
async def test_cleanup_failure_is_logged_without_failing_batch(tmp_path, monkeypatch):
    verifier = CleanupFailingVerifier()
    _patch_runtime(monkeypatch, verifier)

    async def complete(executor, cache, run_dir, step, last_action):
        return "replay_complete", step, last_action, [], []

    monkeypatch.setattr(runner, "continue_replay", complete)
    results = await _run(tmp_path)
    assert results[0].passed is True
    events = [
        json.loads(line)
        for line in (tmp_path / "out" / "runs" / "run_001" / "log.jsonl").read_text().splitlines()
    ]
    warning = next(event for event in events if event["event"] == "cleanup_warning")
    assert warning["code"] == "verifier_cleanup_failed"
