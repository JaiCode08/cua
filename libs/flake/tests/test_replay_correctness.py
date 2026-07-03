import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from replay.cache import Cache
from replay.executor import (
    load_trajectory,
    normalize_replay_action,
    validate_trajectory,
)


def _pattern(kind: str, size=(256, 256)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    if kind == "vertical":
        draw.ellipse((20, 20, 100, 100), fill="black")
    else:
        image.paste("black", (0, 0, size[0], size[1]))
    return image


def _trajectory(root: Path) -> list[dict]:
    actions = []
    for index, screenshot in enumerate(("vertical", "horizontal"), 1):
        turn = root / f"turn-{index:05d}"
        turn.mkdir()
        action = {
            "tool": "click",
            "arguments": {"x": index, "y": index, "pid": 99},
            "timestamp": str(index),
        }
        actions.append(action)
        (turn / "action.json").write_text(json.dumps(action), encoding="utf-8")
        _pattern(screenshot).save(turn / "screenshot.png")
    return actions


def test_cache_transition_uses_post_action_screenshot(tmp_path):
    actions = _trajectory(tmp_path)
    cache = Cache("exact")
    cache.load_from_dir(tmp_path)

    assert (
        cache.lookup(normalize_replay_action(actions[0]), _pattern("vertical"))["timestamp"] == "2"
    )


def test_element_index_with_coordinates_normalizes_safely():
    normalized = normalize_replay_action(
        {
            "tool": "click",
            "arguments": {"element_index": 7, "window_id": 8, "pid": 9, "x": 10, "y": 11},
        }
    )
    assert normalized["arguments"] == {"x": 10, "y": 11}


def test_element_index_without_coordinates_is_rejected(tmp_path):
    turn = tmp_path / "turn-00001"
    turn.mkdir()
    (turn / "action.json").write_text(
        json.dumps({"tool": "click", "arguments": {"element_index": 7}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no x/y coordinate fallback"):
        validate_trajectory(tmp_path)


def test_control_actions_are_filtered_from_loaded_trajectory(tmp_path):
    actions = [
        {"tool": "start_session", "arguments": {"session": "fixture"}},
        {"tool": "click", "arguments": {"x": 10, "y": 11}},
        {"tool": "get_window_state", "arguments": {}},
    ]
    for index, action in enumerate(actions, 1):
        turn = tmp_path / f"turn-{index:05d}"
        turn.mkdir()
        (turn / "action.json").write_text(json.dumps(action), encoding="utf-8")

    validate_trajectory(tmp_path)

    assert load_trajectory(tmp_path) == [actions[1]]


def test_unknown_action_is_rejected(tmp_path):
    turn = tmp_path / "turn-00001"
    turn.mkdir()
    (turn / "action.json").write_text(
        json.dumps({"tool": "unknown_action", "arguments": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not supported for replay"):
        validate_trajectory(tmp_path)


def test_committed_trajectories_validate():
    trajectories = Path(__file__).parents[1] / "trajectories"

    for trajectory in (path for path in trajectories.iterdir() if path.is_dir()):
        validate_trajectory(trajectory)
