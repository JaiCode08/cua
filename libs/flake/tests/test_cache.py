import json

import pytest
from PIL import Image, ImageEnhance
from replay.cache import Cache


def _write_turn(root, index, action, image):
    turn = root / f"turn-{index:05d}"
    turn.mkdir()
    (turn / "action.json").write_text(json.dumps(action), encoding="utf-8")
    image.save(turn / "screenshot.png")


@pytest.mark.parametrize("mode", ["exact", "perceptual", "crop", "downsample"])
def test_same_observation_hits_in_every_match_mode(mode):
    image = Image.new("RGB", (100, 100), color="white")
    action = {"tool": "click", "arguments": {"x": 50, "y": 50}}
    next_action = {"tool": "type_text", "arguments": {"text": "hello"}}
    cache = Cache(match_mode=mode)

    cache.store(action, image, next_action)

    assert cache.lookup(action, image) == next_action
    assert cache.stats() == {"hits": 1, "misses": 0, "size": 1}


def test_load_from_dir(tmp_path):
    actions = []
    screenshots = []
    for i in range(3):
        img = Image.new("RGB", (100, 100), color="white")
        action = {"tool": "click", "arguments": {"x": i, "y": i}, "timestamp": str(i)}
        actions.append(action)
        screenshots.append(img)
        _write_turn(tmp_path, i, action, img)

    cache = Cache(match_mode="perceptual")
    cache.load_from_dir(tmp_path)

    assert cache.suggest({"tool": "__start__"}) == actions[0]
    assert cache.lookup(actions[0], screenshots[0], capture_scope="desktop") == actions[1]
    assert cache.lookup(actions[1], screenshots[1], capture_scope="desktop") == actions[2]
    assert cache.stats()["size"] == 4  # start, two edges, and visual terminal
    assert cache.is_terminal(actions[2], screenshots[2], capture_scope="desktop")


def test_lookup_and_stats():
    img = Image.new("RGB", (100, 100), color="white")
    action = {"tool": "click"}
    next_action = {"tool": "type_text"}

    cache = Cache(match_mode="perceptual")

    # Lookup on empty cache -> Miss
    assert cache.lookup(action, img) is None
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 0

    # Store and lookup -> Hit
    cache.store(action, img, next_action)
    res = cache.lookup(action, img)

    assert res == {"tool": "type_text", "arguments": {}}
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 1


def test_perceptual_lookup_accepts_small_visual_change():
    img = Image.linear_gradient("L").resize((100, 100)).convert("RGB")
    similar = ImageEnhance.Brightness(img).enhance(0.98)
    action = {"tool": "click", "arguments": {"pid": 1, "x": 50, "y": 50}}
    next_action = {"tool": "type_text", "arguments": {"pid": 1, "text": "hello"}}
    cache = Cache("perceptual")
    cache.store(action, img, next_action)
    # Ephemeral target ids do not prevent reuse in a fresh session.
    replay_action = {"tool": "click", "arguments": {"pid": 99, "x": 50, "y": 50}}
    assert cache.lookup(replay_action, similar) == {
        "tool": "type_text",
        "arguments": {"text": "hello"},
    }


def test_element_indexed_next_action_is_a_cache_miss():
    img = Image.new("RGB", (100, 100), color="white")
    cache = Cache("perceptual")
    with pytest.raises(ValueError, match="no x/y coordinate fallback"):
        cache.store(
            {"tool": "hotkey", "arguments": {"keys": ["ctrl", "h"]}},
            img,
            {"tool": "click", "arguments": {"element_index": 7}},
        )


def test_crop_mode_similar_input_same_key():
    """Similar screenshots with background changes map to the same key in crop mode."""
    # Base image: mostly gray
    base = Image.new("RGB", (300, 300), color=(128, 128, 128))
    # Tweaked: background changed far from the click point
    tweaked = base.copy()
    for x in range(0, 20):
        for y in range(0, 20):
            tweaked.putpixel((x, y), (200, 200, 200))

    action = {
        "tool": "click",
        "arguments": {"x": 150, "y": 150},
        "click_point": {"x": 150, "y": 150},
    }
    next_action = {"tool": "type_text", "arguments": {"text": "hello"}}
    cache = Cache("crop")
    cache.store(action, base, next_action)
    # Background change away from click point should still hit
    assert cache.lookup(action, tweaked) == next_action


def test_downsample_mode_similar_input_same_key():
    """Single-pixel change maps to the same key in downsample mode."""
    base = Image.new("RGB", (200, 200), color=(128, 128, 128))
    slightly_different = base.copy()
    # A single pixel change should be negligible after 64x64 downsample
    slightly_different.putpixel((100, 100), (130, 130, 130))

    action = {
        "tool": "click",
        "arguments": {"x": 50, "y": 50},
        "click_point": {"x": 50, "y": 50},
    }
    next_action = {"tool": "type_text", "arguments": {"text": "world"}}
    cache = Cache("downsample")
    cache.store(action, base, next_action)
    assert cache.lookup(action, slightly_different) == next_action
