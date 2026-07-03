from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imagehash
from PIL import Image, ImageChops, ImageStat

from replay.executor import NON_REPLAYABLE_TOOLS, SUPPORTED_ACTIONS, normalize_replay_action


EPHEMERAL_ARGUMENTS = {"pid", "window_id", "session", "dispatch"}


def recorded_capture_scope(action: dict) -> str:
    """Mirror cua-driver recording: no pid/window target falls back to desktop."""
    arguments = action.get("arguments", {})
    return "window" if "pid" in arguments or "window_id" in arguments else "desktop"


def action_fingerprint(action: dict) -> str:
    """Return a stable identity for an action across desktop sessions."""
    arguments = {
        key: value
        for key, value in action.get("arguments", {}).items()
        if key not in EPHEMERAL_ARGUMENTS
    }
    identity = {"tool": action.get("tool", ""), "arguments": arguments}
    if action.get("timestamp") is not None:
        identity["timestamp"] = action["timestamp"]
    if action.get("_intervention_id") is not None:
        identity["intervention_id"] = action["_intervention_id"]
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CacheKey:
    last_action_type: str
    screenshot_hash: str
    capture_scope: str = "window"


@dataclass
class CacheEntry:
    action_fingerprint: str
    screenshot_hash: str
    screenshot: Image.Image
    next_action: Optional[dict]
    capture_scope: str = "window"


def perceptual_hash(img: Image.Image) -> str:
    return str(imagehash.phash(img.convert("RGB")))


def crop_image(img: Image.Image, click_point: dict) -> Image.Image:
    """Crop around a window-local action point in window screenshot coordinates."""
    width, height = img.size
    x = int(click_point.get("x", width // 2))
    y = int(click_point.get("y", height // 2))
    left = max(0, x - 100)
    top = max(0, y - 100)
    right = min(width, x + 100)
    bottom = min(height, y + 100)
    if left >= right or top >= bottom:
        return img.resize((64, 64)).convert("L")
    return img.crop((left, top, right, bottom))


def crop_hash(img: Image.Image, click_point: dict) -> str:
    return str(imagehash.phash(crop_image(img, click_point)))


def downsample_image(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize((64, 64)).convert("L")


def downsample_hash(img: Image.Image) -> str:
    return downsample_image(img).tobytes().hex()


class Cache:
    def __init__(
        self, match_mode: str, perceptual_threshold: int = 10, downsample_threshold: float = 40.0
    ):
        if match_mode not in ("exact", "perceptual", "crop", "downsample"):
            raise ValueError(f"Invalid match_mode: {match_mode}")
        self.match_mode = match_mode
        self.perceptual_threshold = perceptual_threshold
        self.downsample_threshold = downsample_threshold
        self._cache: dict[CacheKey, Optional[dict]] = {}
        self._entries: list[CacheEntry] = []
        self._terminal_entries: list[CacheEntry] = []
        self._start_actions: list[dict] = []
        self._terminal_actions_without_screenshot: set[str] = set()
        # Kept as an alias for compatibility with older callers/tests.
        self._terminal_actions = self._terminal_actions_without_screenshot
        self._hits = 0
        self._misses = 0

    def _image_for(self, action: dict, screenshot: Image.Image) -> Image.Image:
        if self.match_mode == "crop":
            point = action.get("click_point") or action.get("arguments", {})
            return crop_image(screenshot, point)
        if self.match_mode == "downsample":
            return downsample_image(screenshot)
        return screenshot.convert("RGB")

    def _hash_for(self, action: dict, screenshot: Image.Image) -> str:
        if self.match_mode == "exact":
            image = screenshot.convert("RGB")
            return hashlib.sha256(image.tobytes()).hexdigest()
        if self.match_mode == "perceptual":
            return perceptual_hash(screenshot)
        if self.match_mode == "crop":
            point = action.get("click_point") or action.get("arguments", {})
            return crop_hash(screenshot, point)
        return downsample_hash(screenshot)

    def _compute_key(
        self, last_action: dict, screenshot: Image.Image, capture_scope: str = "window"
    ) -> CacheKey:
        return CacheKey(
            action_fingerprint(last_action),
            self._hash_for(last_action, screenshot),
            capture_scope,
        )

    def _similar(
        self, current: Image.Image, entry: CacheEntry, *, capture_scope: str = "window"
    ) -> bool:
        if capture_scope != entry.capture_scope:
            return False
        if self.match_mode == "exact":
            reference = entry.screenshot.convert("RGB")
            candidate = current.convert("RGB")
            return (
                candidate.size == reference.size
                and ImageChops.difference(candidate, reference).getbbox() is None
            )
        if self.match_mode in ("perceptual", "crop"):
            return (
                imagehash.phash(current) - imagehash.hex_to_hash(entry.screenshot_hash)
                <= self.perceptual_threshold
            )
        candidate = downsample_image(current)
        reference = entry.screenshot
        if reference.mode != "L" or reference.size != (64, 64):
            reference = downsample_image(reference)
        return (
            ImageStat.Stat(ImageChops.difference(candidate, reference)).mean[0]
            <= self.downsample_threshold
        )

    def is_terminal(
        self,
        action: dict,
        screenshot: Optional[Image.Image] = None,
        capture_scope: str = "window",
        *,
        count: bool = False,
    ) -> bool:
        fingerprint = action_fingerprint(action)
        if fingerprint in self._terminal_actions_without_screenshot:
            return True
        if screenshot is None:
            return False
        current = self._image_for(action, screenshot)
        for entry in self._terminal_entries:
            if entry.action_fingerprint != fingerprint:
                continue
            if self._similar(current, entry, capture_scope=capture_scope):
                if count:
                    self._hits += 1
                return True
        return False

    def expected_capture_scope(self, last_action: dict) -> str:
        fingerprint = action_fingerprint(last_action)
        for entry in [*self._entries, *self._terminal_entries]:
            if entry.action_fingerprint == fingerprint:
                return entry.capture_scope
        return "window"

    def suggest(self, last_action: dict) -> Optional[dict]:
        """Return a recorded next action without claiming a visual cache hit."""
        if last_action.get("tool") == "__start__":
            return dict(self._start_actions[0]) if self._start_actions else None
        fingerprint = action_fingerprint(last_action)
        for entry in self._entries:
            if entry.action_fingerprint == fingerprint and entry.next_action is not None:
                return dict(entry.next_action)
        return None

    def lookup(
        self, last_action: dict, screenshot: Image.Image, capture_scope: str = "window"
    ) -> Optional[dict]:
        if last_action.get("tool") == "__start__":
            return dict(self._start_actions[0]) if self._start_actions else None

        fingerprint = action_fingerprint(last_action)
        current = self._image_for(last_action, screenshot)
        for entry in self._entries:
            if entry.action_fingerprint != fingerprint:
                continue
            if self._similar(current, entry, capture_scope=capture_scope):
                self._hits += 1
                return dict(entry.next_action) if entry.next_action is not None else None
        self._misses += 1
        return None

    def store(
        self,
        last_action: dict,
        screenshot: Image.Image,
        next_action: dict,
        capture_scope: str = "window",
    ) -> None:
        next_action = normalize_replay_action(next_action, source="cache action")
        fingerprint = action_fingerprint(last_action)
        prepared = self._image_for(last_action, screenshot).copy()
        screenshot_hash = self._hash_for(last_action, screenshot)
        entry = CacheEntry(
            fingerprint,
            screenshot_hash,
            prepared,
            dict(next_action),
            capture_scope,
        )
        self._entries.append(entry)
        self._cache[CacheKey(fingerprint, screenshot_hash, capture_scope)] = dict(next_action)

    def store_terminal(
        self,
        last_action: dict,
        screenshot: Image.Image,
        capture_scope: str = "window",
    ) -> None:
        fingerprint = action_fingerprint(last_action)
        prepared = self._image_for(last_action, screenshot).copy()
        screenshot_hash = self._hash_for(last_action, screenshot)
        entry = CacheEntry(fingerprint, screenshot_hash, prepared, None, capture_scope)
        self._terminal_entries.append(entry)
        self._cache[CacheKey(fingerprint, screenshot_hash, capture_scope)] = None

    def load_from_dir(self, trajectory_dir: Path) -> None:
        turns = sorted(
            (
                path
                for path in trajectory_dir.iterdir()
                if path.is_dir() and path.name.startswith("turn-")
            ),
            key=lambda path: path.name,
        )
        valid_actions = []
        for turn in turns:
            action = json.loads((turn / "action.json").read_text(encoding="utf-8"))
            if action.get("tool") in NON_REPLAYABLE_TOOLS:
                continue
            if action.get("tool") not in SUPPORTED_ACTIONS:
                raise ValueError(
                    f"Invalid trajectory at {turn / 'action.json'}: tool "
                    f"'{action.get('tool')}' is not supported for replay"
                )
            capture_scope = recorded_capture_scope(action)
            action = normalize_replay_action(action, source=str(turn / "action.json"))
            valid_actions.append((action, turn, capture_scope))
        if not valid_actions:
            return

        actions = [item[0] for item in valid_actions]
        self._start_actions.append(actions[0])
        for index in range(len(valid_actions) - 1):
            current_action, current_turn, current_scope = valid_actions[index]
            next_action, _, _ = valid_actions[index + 1]
            screenshot_path = current_turn / "screenshot.png"
            if not screenshot_path.is_file():
                warnings.warn(
                    f"Skipping cache transition after {current_turn.name}: missing post-action screenshot.png",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            with Image.open(screenshot_path) as screenshot:
                self.store(
                    current_action,
                    screenshot,
                    next_action,
                    capture_scope=current_scope,
                )

        final_action, final_turn, final_scope = valid_actions[-1]
        final_screenshot = final_turn / "screenshot.png"
        if final_screenshot.is_file():
            with Image.open(final_screenshot) as screenshot:
                self.store_terminal(
                    final_action,
                    screenshot,
                    capture_scope=final_scope,
                )
        else:
            # Final actions often close the target window, making capture impossible.
            self._terminal_actions_without_screenshot.add(action_fingerprint(final_action))

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._entries) + len(self._terminal_entries) + len(self._start_actions),
        }

    def hit(self) -> None:
        self._hits += 1

    def miss(self) -> None:
        self._misses += 1
