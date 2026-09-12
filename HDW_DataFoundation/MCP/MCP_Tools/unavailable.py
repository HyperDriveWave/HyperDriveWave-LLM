from __future__ import annotations

from typing import Any


class UnavailableService:
    def __init__(self, feature: str, reason: str) -> None:
        self.feature = feature
        self.reason = reason

    def __getattr__(self, name: str):
        def unavailable(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "status": "unavailable",
                "feature": self.feature,
                "tool": name,
                "message": self.reason,
            }

        return unavailable
