import json

import pytest
from PIL import Image

from replay.cache import Cache
from replay.executor import ActionResult, ReplayExecutor, validate_trajectory
from replay.tool import continue_replay


class FakeExecutor:
    def __init__(self, image):
        self.image = image
        self.executed = []

    async def capture_observation(self, capture_scope="window"):
        return {"screenshot": self.image, "capture_scope": capture_scope}

    async def execute_action(self, action, step, run_dir):
        self.executed.append(action)
        return ActionResult(step, True, action["tool"], 1)


def _write_trajectory(root, actions, screenshots):
    root.mkdir(exist_ok=True)
    for index, (action, screenshot) in enumerate(zip(actions, screenshots, strict=True), 1):
        turn = root / f"turn-{index:05d}"
        turn.mkdir()
        (turn / "action.json").write_text(json.dumps(action), encoding="utf-8")
        screenshot.save(turn / "screenshot.png")


@pytest.mark.asyncio
async def test_continue_replay_reaches_terminal_action(tmp_path):
    image = Image.new("RGB", (20, 20), "white")
    first = {"tool": "click", "arguments": {"x": 1, "y": 2}}
    second = {"tool": "type_text", "arguments": {"text": "ok"}}
    _write_trajectory(tmp_path / "trajectory", [first, second], [image, image])
    cache = Cache("perceptual")
    cache.load_from_dir(tmp_path / "trajectory")
    executor = FakeExecutor(image)
    run_dir = tmp_path / "run"
    status, step, last_action, results, steps = await continue_replay(
        executor, cache, run_dir, 1, {"tool": "__start__"}
    )
    assert status == "replay_complete"
    assert [action["tool"] for action in executor.executed] == ["click", "type_text"]
    assert step == 3
    assert len(results) == len(steps) == 2


@pytest.mark.asyncio
async def test_continue_replay_returns_cache_miss_artifact(tmp_path):
    image = Image.new("RGB", (20, 20), "white")
    executor = FakeExecutor(image)
    status, _, _, _, steps = await continue_replay(
        executor, Cache("perceptual"), tmp_path, 1, {"tool": "__start__"}
    )
    assert status == "cache_miss"
    assert not steps[0].cache_hit
    assert (tmp_path / "turn-00001" / "screenshot_miss.png").exists()


@pytest.mark.asyncio
async def test_failed_cached_action_preserves_pre_intervention_screenshot(tmp_path):
    miss_image = Image.new("RGB", (20, 20), "red")
    first = {"tool": "launch_app", "arguments": {"name": "cmd"}}
    failed = {"tool": "click", "arguments": {"x": 1, "y": 2}}
    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    _write_trajectory(trajectory, [first, failed], [miss_image, miss_image])
    cache = Cache("perceptual")
    cache.load_from_dir(trajectory)

    class FailingExecutor(FakeExecutor):
        async def execute_action(self, action, step, run_dir):
            self.executed.append(action)
            if action["tool"] == "click":
                turn = run_dir / f"turn-{step:05d}"
                turn.mkdir(parents=True, exist_ok=True)
                screenshot = turn / "screenshot.png"
                miss_image.save(screenshot)
                return ActionResult(
                    step, False, "click", 1, "dispatch rejected", screenshot_path=str(screenshot)
                )
            return ActionResult(step, True, action["tool"], 1)

    run_dir = tmp_path / "run"
    status, _, _, _, steps = await continue_replay(
        FailingExecutor(miss_image), cache, run_dir, 1, {"tool": "__start__"}
    )

    assert status == "cache_miss"
    preserved = run_dir / "turn-00002" / "screenshot_miss.png"
    assert steps[-1].screenshot_path == str(preserved)
    Image.new("RGB", (20, 20), "blue").save(run_dir / "turn-00002" / "screenshot.png")
    with Image.open(preserved) as screenshot:
        assert screenshot.getpixel((0, 0)) == (255, 0, 0)


def test_validate_trajectory_rejects_missing_action(tmp_path):
    turn = tmp_path / "turn-00001"
    turn.mkdir()
    Image.new("RGB", (5, 5)).save(turn / "screenshot.png")
    with pytest.raises(ValueError, match='missing required file "action.json"'):
        validate_trajectory(tmp_path)
