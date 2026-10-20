"""The slow loop: everything that is too expensive to run per frame.

Order of operations matters and is deliberate:

    split spread  ->  dewarp each page  ->  OCR

Splitting comes first because the two halves of an open book curve in *opposite*
directions: fitting one displacement field across the gutter averages the two and
flattens neither. Dewarping comes before OCR because line segmentation is what
breaks on curved text.

This stage is a pure function of (image, config), which is what lets it run in a
thread, in a separate process, or on another machine behind a queue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..export import speech
from ..ocr.base import OcrEngine
from ..types import OcrPage, PageCandidate, PageResult
from ..vision import dewarp as dewarp_mod
from ..vision import spread as spread_mod


@dataclass
class WorkerOutput:
    """One or two finished pages, plus the diagnostics behind them.

    `images[i]` is the image that `results[i]` was recognised from -- after
    splitting and dewarping. Exporters must use exactly this image as the page
    background: the OCR boxes are in its coordinate space, so pairing the text
    layer with the pre-dewarp image would put every word in the wrong place.
    """

    results: list[PageResult]
    images: list[np.ndarray]
    spread: spread_mod.SpreadSplit
    dewarp_reports: list[dewarp_mod.DewarpReport] = field(default_factory=list)


@dataclass
class PageWorker:
    engine: OcrEngine
    cfg: Config = field(default_factory=Config)

    def process(
        self, candidate: PageCandidate, image: np.ndarray, start_page_number: int = 1
    ) -> WorkerOutput:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        if self.cfg.split_spreads:
            pages, spread_result = spread_mod.split_if_spread(
                image,
                self.cfg.spread_min_aspect,
                self.cfg.spread_min_confidence,
                self.cfg.spread_min_gutter_span,
            )
        else:
            pages, spread_result = [image], spread_mod.SpreadSplit(False, None, 0.0, "disabled")
        timings["split_ms"] = (time.perf_counter() - t0) * 1000.0

        sides = ["single"] if len(pages) == 1 else ["left", "right"]
        results: list[PageResult] = []
        images: list[np.ndarray] = []
        reports: list[dewarp_mod.DewarpReport] = []

        for offset, (page_image, side) in enumerate(zip(pages, sides, strict=True)):
            page_timings = dict(timings)

            t1 = time.perf_counter()
            flat, report = dewarp_mod.dewarp(page_image, self.cfg.dewarp)
            page_timings["dewarp_ms"] = (time.perf_counter() - t1) * 1000.0
            reports.append(report)
            images.append(flat)

            t2 = time.perf_counter()
            ocr: OcrPage = self.engine.run(flat)
            page_timings["ocr_ms"] = (time.perf_counter() - t2) * 1000.0

            page_candidate = PageCandidate(
                frame_index=candidate.frame_index,
                timestamp=candidate.timestamp,
                quad=candidate.quad,
                quality=candidate.quality,
                phash=candidate.phash,
                side=side,
            )
            results.append(
                PageResult(
                    page_number=start_page_number + offset,
                    ocr=ocr,
                    candidate=page_candidate,
                    dewarped=report.applied,
                    timings_ms=page_timings,
                    reading_text=speech.page_text(ocr),
                )
            )

        return WorkerOutput(
            results=results, images=images, spread=spread_result, dewarp_reports=reports
        )
