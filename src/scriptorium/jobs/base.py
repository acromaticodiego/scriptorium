"""Job abstraction between the API and the heavy pipeline.

The API must answer in milliseconds; OCR takes hundreds of them per page. So a
request enqueues work and returns a job id, and the client polls or streams.

The point of the protocol is that `InMemoryQueue` (good for a laptop demo and for
CI) and a Redis/RQ queue across several worker machines are swappable without the
API changing: the API only ever submits and reads status.
"""

from __future__ import annotations

import enum
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


class JobState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: JobState = JobState.QUEUED
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    result: Any = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "created_at": self.created_at,
            "result": self.result,
            "error": self.error,
        }


class JobQueue(Protocol):
    def submit(self, fn: Callable[[], Any]) -> Job:
        ...

    def get(self, job_id: str) -> Job | None:
        ...

    def shutdown(self) -> None:
        ...
