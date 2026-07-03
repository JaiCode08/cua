from dataclasses import dataclass
from typing import Optional

from replay.executor import ActionResult
from verifiers.base import VerificationResult


@dataclass
class FailureInfo:
    category: str
    reason: str
    detail: dict


def classify_failure(
    error: Optional[Exception] = None,
    action_result: Optional[ActionResult] = None,
    verifier_result: Optional[VerificationResult] = None,
    context: str = "",
) -> FailureInfo:
    # Rule 1: context == "reset" and error -> "reset_failed"
    if context == "reset" and error is not None:
        return FailureInfo(
            category="reset_failed", reason=str(error), detail={"error_type": type(error).__name__}
        )

    # Rule 2: context == "agent" and isinstance(error, TimeoutError) -> "agent_timeout"
    if context == "agent" and isinstance(error, TimeoutError):
        return FailureInfo(category="agent_timeout", reason=str(error), detail={})

    # Rule 3: context == "agent" and error -> "agent_cli_crashed"
    if context == "agent" and error is not None:
        return FailureInfo(
            category="agent_cli_crashed",
            reason=str(error),
            detail={"error_type": type(error).__name__},
        )

    if context == "verify" and error is not None:
        return FailureInfo(
            category="verifier_failed",
            reason=str(error),
            detail={"error_type": type(error).__name__},
        )

    # Rule 4 & 5: action_result dispatch failures
    if action_result and not action_result.ok:
        if action_result.error and "cache_miss" in action_result.error:
            return FailureInfo(
                category="cache_miss_unrecovered",
                reason=action_result.error,
                detail={"step": action_result.step, "action_type": action_result.action_type},
            )
        return FailureInfo(
            category="action_dispatch_failed",
            reason=action_result.error or "Unknown action error",
            detail={"step": action_result.step, "action_type": action_result.action_type},
        )

    # Rule 6: verifier_result and not verifier_result.passed -> "verifier_failed"
    if verifier_result and not verifier_result.passed:
        return FailureInfo(
            category="verifier_failed",
            reason=verifier_result.reason or "Verifier failed",
            detail=verifier_result.details,
        )

    # Rule 7: error and "drift" in str(error).lower() -> "environment_drift"
    if error and "drift" in str(error).lower():
        return FailureInfo(
            category="environment_drift",
            reason=str(error),
            detail={"error_type": type(error).__name__},
        )

    # Rule 8: default -> "unknown"
    # Ensure there's a fallback string if no obvious reason exists
    reason_str = str(error) if error else "Unknown failure"
    return FailureInfo(category="unknown", reason=reason_str, detail={})
