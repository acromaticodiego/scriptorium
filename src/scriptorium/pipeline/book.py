"""Session orchestration: a stream of frames in, an ordered book out.

Owns the two things neither loop can decide alone:

* **page numbering** -- a capture can yield two pages (a spread), so numbering is
  assigned here, after the worker reports how many pages came back;
* **the second dedup gate** -- the perceptual hash runs pre-OCR and is tuned
  tight, because rejecting a page there loses it with nothing downstream able to
  recover it. It therefore only catches near-identical frames; recognising the
  same *content* in two captures is this layer's job, once there is text to
  compare.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..capture.source import Frame, FrameSource
from ..config import Config
from ..export import speech
from ..ocr.base import OcrEngine
from ..ocr.registry import get_engine
from ..types import PageResult
from .edge import EdgeLoop
from .worker import PageWorker

log = logging.getLogger(__name__)


@dataclass
class BookSession:
    """Accumulates recognised pages for one scanning session."""

    cfg: Config = field(default_factory=Config)
    engine: OcrEngine = field(default=None)  # type: ignore[assignment]
    edge: EdgeLoop = field(default=None)  # type: ignore[assignment]
    worker: PageWorker = field(default=None)  # type: ignore[assignment]
    pages: list[PageResult] = field(default_factory=list)
    images: dict[int, np.ndarray] = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = get_engine(self.cfg.ocr_engine, self.cfg.ocr_lang)
        if self.edge is None:
            self.edge = EdgeLoop(cfg=self.cfg)
        if self.worker is None:
            self.worker = PageWorker(engine=self.engine, cfg=self.cfg)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    def feed(self, frame: Frame) -> list[PageResult]:
        """Push one frame through both loops. Returns pages finished by this frame."""
        event = self.edge.process(frame.image, frame.index, frame.timestamp)
        if event is None:
            return []

        output = self.worker.process(
            event.candidate, event.image, start_page_number=len(self.pages) + 1
        )
        if output.spread.is_spread:
            self.stats["spreads_split"] += 1

        accepted: list[PageResult] = []
        for result, image in zip(output.results, output.images, strict=True):
            if self.edge.deduper.is_duplicate_text(result.text):
                self.stats["rejected_duplicate_text"] += 1
                continue
            result.page_number = len(self.pages) + 1
            self.pages.append(result)
            self.images[result.page_number] = image
            accepted.append(result)
            self.stats["pages"] += 1
            if result.dewarped:
                self.stats["pages_dewarped"] += 1

        # Register the capture once, whatever the text gate decided, so the same
        # frames do not get re-processed on the next pass.
        self.edge.deduper.accept(
            event.candidate.phash, accepted[0].text if accepted else ""
        )
        if accepted:
            self._stitch_tail()
        return accepted

    def _stitch_tail(self) -> None:
        """Complete a word the previous page cut in half.

        A page's text is only final once the *next* page exists: a word broken at
        the bottom edge continues there, and until then there is nothing to join
        it to. So the repair runs here, on the last two pages, each time a page
        is added -- rather than at export time, so that a reader following along
        live gets the corrected text too.
        """
        if len(self.pages) < 2:
            return
        previous, current = self.pages[-2], self.pages[-1]
        repaired = speech.stitch_page_texts([previous.text, current.text])
        if repaired[0] != previous.text:
            previous.reading_text = repaired[0]
            current.reading_text = repaired[1]
            self.stats["pages_stitched"] += 1

    def run(self, source: FrameSource, max_pages: int | None = None) -> list[PageResult]:
        """Consume a whole source. Returns every page recognised."""
        # Still images never produce motion, so the stability gate would starve.
        self.edge.require_stability = not getattr(source, "is_still", False)

        for frame in source.frames():
            self.feed(frame)
            if max_pages is not None and len(self.pages) >= max_pages:
                log.info("stopping at max_pages=%d", max_pages)
                break
        return self.pages

    def summary(self) -> dict[str, object]:
        """Session telemetry: what was captured, rejected, and how fast."""
        merged = Counter(self.edge.stats) + Counter(self.stats)
        ocr_times = [p.timings_ms.get("ocr_ms", 0.0) for p in self.pages]
        dewarp_times = [p.timings_ms.get("dewarp_ms", 0.0) for p in self.pages]
        confidences = [p.ocr.confidence for p in self.pages]
        return {
            "pages": len(self.pages),
            "engine": getattr(self.engine, "name", "unknown"),
            "counters": dict(merged),
            "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "mean_ocr_ms": round(float(np.mean(ocr_times)), 1) if ocr_times else 0.0,
            "mean_dewarp_ms": round(float(np.mean(dewarp_times)), 1) if dewarp_times else 0.0,
        }

    def save_images(self, directory: str | Path) -> list[Path]:
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        for number, image in sorted(self.images.items()):
            path = folder / f"page_{number:04d}.png"
            cv2.imwrite(str(path), image)
            written.append(path)
        return written
