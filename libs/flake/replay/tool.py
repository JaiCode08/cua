from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Tuple

import click

from replay.cache import Cache, action_fingerprint
from replay.executor import (
    ActionResult,
    ContinueReplayStep,
    Observation,
    ObservationCaptureError,
    ReplayExecutor,
)


def _observation_parts(observation) -> tuple[object, str, list[dict]]:
    if isinstance(observation, Observation):
        return observation.screenshot, observation.capture_scope, list(observation.warnings)
    if isinstance(observation, dict):
        return (
            observation.get("screenshot"),
            observation.get("capture_scope", "window"),
            list(observation.get("warnings", [])),
        )
    return None, "unknown", []


def _observation_error(step: int, exc: Exception) -> tuple[ActionResult, ContinueReplayStep]:
    warning = (
        dict(exc.warning)
        if isinstance(exc, ObservationCaptureError)
        else {"code": "screenshot_capture_failed", "message": str(exc), "artifact": "screenshot"}
    )
    result = ActionResult(step, False, "observation", 0, str(exc), warnings=[warning])
    replay_step = ContinueReplayStep(step, False, None, None, 0, "", warnings=[warning])
    return result, replay_step


async def continue_replay(
    executor: ReplayExecutor,
    cache: Cache,
    run_dir: Path,
    start_step: int,
    last_action: dict,
) -> Tuple[str, int, dict, list[ActionResult], list[ContinueReplayStep]]:
    """Replay cached actions until completion, a cache miss, or an action error."""
    step = start_step
    current_last_action = last_action
    action_results: list[ActionResult] = []
    steps: list[ContinueReplayStep] = []
    seen_transitions: set[tuple[str, str]] = set()

    while True:
        if cache.is_terminal(current_last_action):
            return "replay_complete", step, current_last_action, action_results, steps

        is_start_action = current_last_action.get("tool") == "__start__"
        observation_warnings: list[dict] = []
        if is_start_action:
            # The recording format has no pre-action screenshot. Bootstrap the
            # first action without claiming either a visual hit or miss.
            next_action = cache.suggest(current_last_action)
        else:
            try:
                expected_scope = cache.expected_capture_scope(current_last_action)
                observation = await executor.capture_observation(expected_scope)
                image, capture_scope, observation_warnings = _observation_parts(observation)
                if image is None:
                    raise ObservationCaptureError("Observation contained no screenshot")
            except Exception as exc:
                result, replay_step = _observation_error(step, exc)
                action_results.append(result)
                steps.append(replay_step)
                return "error", step, current_last_action, action_results, steps

            if cache.is_terminal(
                current_last_action, image, capture_scope=capture_scope, count=True
            ):
                return "replay_complete", step, current_last_action, action_results, steps
            next_action = cache.lookup(current_last_action, image, capture_scope=capture_scope)

        if next_action is None:
            if is_start_action:
                try:
                    observation = await executor.capture_observation("window")
                    image, _, observation_warnings = _observation_parts(observation)
                    if image is None:
                        raise ObservationCaptureError("Observation contained no screenshot")
                except Exception as exc:
                    result, replay_step = _observation_error(step, exc)
                    action_results.append(result)
                    steps.append(replay_step)
                    return "error", step, current_last_action, action_results, steps
                turn_dir = run_dir / f"turn-{step:05d}"
                turn_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = turn_dir / "screenshot_miss.png"
                image.save(screenshot_path)
                steps.append(
                    ContinueReplayStep(
                        step,
                        False,
                        None,
                        None,
                        0,
                        str(screenshot_path),
                        warnings=observation_warnings,
                    )
                )
                return "cache_miss", step, current_last_action, action_results, steps
            turn_dir = run_dir / f"turn-{step:05d}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = turn_dir / "screenshot_miss.png"
            image.save(screenshot_path)
            steps.append(
                ContinueReplayStep(
                    step,
                    False,
                    None,
                    None,
                    0,
                    str(screenshot_path),
                    warnings=observation_warnings,
                    capture_scope=capture_scope,
                )
            )
            return "cache_miss", step, current_last_action, action_results, steps

        transition = (action_fingerprint(current_last_action), action_fingerprint(next_action))
        if transition in seen_transitions:
            loop_error = ActionResult(
                step,
                False,
                next_action.get("tool", ""),
                0,
                "cache loop detected: repeated state transition",
            )
            action_results.append(loop_error)
            return "error", step, current_last_action, action_results, steps
        seen_transitions.add(transition)

        result = await executor.execute_action(next_action, step, run_dir)
        action_results.append(result)
        steps.append(
            ContinueReplayStep(
                step=step,
                cache_hit=not is_start_action,
                tool=next_action.get("tool"),
                args=next_action.get("arguments"),
                duration_ms=result.duration_ms,
                screenshot_path=result.screenshot_path or "",
                warnings=observation_warnings + list(result.warnings),
                capture_scope="unverified" if is_start_action else capture_scope,
            )
        )
        if not result.ok:
            cache.miss()
            steps[-1].cache_hit = False
            # Preserve the miss observation before the forgiving runner writes
            # the intervention artifacts into this same turn directory.
            if result.screenshot_path:
                source = Path(result.screenshot_path)
                if source.is_file():
                    miss_path = source.with_name("screenshot_miss.png")
                    shutil.copy2(source, miss_path)
                    steps[-1].screenshot_path = str(miss_path)
            return "cache_miss", step, current_last_action, action_results, steps
        current_last_action = next_action
        step += 1


@click.command()
@click.option("--trajectory", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--run-dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--mode", default="perceptual", type=click.Choice(["perceptual", "crop", "downsample"])
)
@click.option("--action-timeout", default=15.0, type=click.FloatRange(min=0.1))
@click.option("--launch-timeout", default=30.0, type=click.FloatRange(min=0.1))
@click.option("--replay-timeout", default=600.0, type=click.FloatRange(min=1.0))
def cli(
    trajectory: Path,
    run_dir: Path,
    mode: str,
    action_timeout: float,
    launch_timeout: float,
    replay_timeout: float,
) -> None:
    """Continue a trajectory replay until a cache miss or completion."""

    async def run() -> dict:
        cache = Cache(match_mode=mode)
        executor = ReplayExecutor(trajectory, action_timeout, launch_timeout, replay_timeout)
        await executor.setup()
        cache.load_from_dir(trajectory)
        cleanup_warnings: list[dict] = []
        try:
            status, step, last_action, action_results, steps = await asyncio.wait_for(
                continue_replay(executor, cache, run_dir, 1, {"tool": "__start__"}),
                timeout=replay_timeout,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "log.jsonl"
            with log_path.open("w", encoding="utf-8") as log:
                for item in steps:
                    log.write(
                        json.dumps({"event": "replay_step", **dataclasses.asdict(item)}) + "\n"
                    )
                for item in action_results:
                    log.write(
                        json.dumps({"event": "action_result", **dataclasses.asdict(item)}) + "\n"
                    )
            payload = {
                "status": status,
                "step": step,
                "last_action": last_action,
                "actions_executed": len(action_results),
                "cache_hits": cache.stats()["hits"],
                "cache_misses": cache.stats()["misses"],
                "steps": [dataclasses.asdict(item) for item in steps],
                "action_results": [dataclasses.asdict(item) for item in action_results],
                "cleanup_warnings": cleanup_warnings,
            }
            return payload
        finally:
            cleanup_warnings.extend(await executor.teardown())
            if cleanup_warnings:
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / "log.jsonl").open("a", encoding="utf-8") as log:
                    for warning in cleanup_warnings:
                        log.write(json.dumps({"event": "cleanup_warning", **warning}) + "\n")

    print(json.dumps(asyncio.run(run())))


if __name__ == "__main__":
    cli()
