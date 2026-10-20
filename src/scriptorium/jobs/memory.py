"""In-process job queue backed by a thread pool.

Good enough for a single-machine deployment and for tests, and it keeps a fresh
clone free of infrastructure. OCR work releases the GIL inside OpenCV and the
Tesseract C library, so threads do give real parallelism here.

Jobs are kept in a bounded history so a long-lived server does not grow without
limit -- the kind of thing that only shows up after a demo has been running for
an hour.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .base import Job, JobState

log = logging.getLogger(__name__)


class InMemoryQueue:
    def __init__(self, workers: int = 2, history: int = 500) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scriptorium")
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._history = history

    def submit(self, fn: Callable[[], Any]) -> Job:
        job = Job()
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._history:
                self._jobs.popitem(last=False)
        self._pool.submit(self._run, job, fn)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.state.value] = counts.get(job.state.value, 0) + 1
            return counts

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: Job, fn: Callable[[], Any]) -> None:
        job.state = JobState.RUNNING
        try:
            job.result = fn()
            job.state = JobState.DONE
        except Exception as exc:
            log.exception("job %s failed", job.id)
            job.error = f"{type(exc).__name__}: {exc}"
            job.state = JobState.FAILED
