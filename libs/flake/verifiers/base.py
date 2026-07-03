from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class VerificationResult:
    passed: bool
    reason: Optional[str] = None
    details: dict = field(default_factory=dict)

    @classmethod
    def pass_(cls, details: Optional[dict] = None) -> VerificationResult:
        return cls(passed=True, details=details or {})

    @classmethod
    def fail(cls, reason: str, details: Optional[dict] = None) -> VerificationResult:
        return cls(passed=False, reason=reason, details=details or {})


class Verifier(ABC):
    task_description: str = "Complete the recorded GUI task."

    @abstractmethod
    async def reset(self, driver: Any) -> None:
        """Puts state back to a clean starting point before the run."""
        pass

    @abstractmethod
    async def verify(self, driver: Any) -> VerificationResult:
        """Checks whether the task actually succeeded after the run."""
        pass

    async def teardown(self, driver: Any) -> None:
        """Cleans up the environment after all runs are complete."""
        pass

    def prepare_action(self, action: dict) -> dict:
        """Optionally redirect fixture paths without modifying the source trajectory."""
        return action
