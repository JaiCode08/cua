"""Public verifier interface for the optional flakiness debugger package."""

try:
    from verifiers.base import VerificationResult, Verifier
except ImportError as exc:  # pragma: no cover - exercised only without optional package
    raise ImportError("cua_driver.flakiness requires the cua-internship flakiness package") from exc

__all__ = ["Verifier", "VerificationResult"]
