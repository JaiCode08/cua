import pytest

from flakiness.classifier import classify_failure, FailureInfo
from replay.executor import ActionResult
from verifiers.base import VerificationResult


def test_classify_reset_failed():
    res = classify_failure(context="reset", error=Exception("crash"))
    assert res.category == "reset_failed"


def test_classify_agent_timeout():
    res = classify_failure(context="agent", error=TimeoutError("took too long"))
    assert res.category == "agent_timeout"


def test_classify_agent_cli_crashed():
    res = classify_failure(context="agent", error=RuntimeError("exit code 1"))
    assert res.category == "agent_cli_crashed"


def test_classify_cache_miss_unrecovered():
    ar = ActionResult(
        step=1,
        ok=False,
        action_type="cache_miss",
        duration_ms=0,
        error="cache_miss: max interventions",
    )
    res = classify_failure(action_result=ar)
    assert res.category == "cache_miss_unrecovered"


def test_classify_action_dispatch_failed():
    ar = ActionResult(
        step=1, ok=False, action_type="click", duration_ms=0, error="element not found"
    )
    res = classify_failure(action_result=ar)
    assert res.category == "action_dispatch_failed"


def test_classify_verifier_failed():
    vr = VerificationResult.fail(reason="missing file")
    res = classify_failure(verifier_result=vr)
    assert res.category == "verifier_failed"


def test_classify_verifier_exception():
    res = classify_failure(context="verify", error=TimeoutError("window missing"))
    assert res.category == "verifier_failed"


def test_classify_environment_drift():
    res = classify_failure(error=Exception("window drift observed"))
    assert res.category == "environment_drift"


def test_classify_unknown():
    res = classify_failure(error=Exception("mysterious issue"))
    assert res.category == "unknown"
