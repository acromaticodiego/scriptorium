"""The fast loop: runs on every frame, must stay inside the frame budget.

Budget at 30 fps is 33 ms per frame, and we want most of it left for the UI, so
everything here works on downscaled gray copies. The loop never calls OCR; its
only output is "here is a new page, worth the expensive pipeline".

Rejections are counted, not silently dropped: `stats` is what the CLI prints and
the API exposes, and it is how you debug a session that captured nothing.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..types import PageCandidate, TurnState
from ..vision import dedup as dedup_mod
from ..vision import detect, quality
from ..vision.stability import StabilityTracker


@dataclass
class CaptureEvent:
    """A page the edge loop accepted, with the rectified image attached."""

    candidate: PageCandidate
    image: np.ndarray


@dataclass
class EdgeLoop:
    cfg: Config = field(default_factory=Config)
    tracker: StabilityTracker = field(default=None)  # type: ignore[assignment]
    deduper: dedup_mod.Deduplicator = field(default=None)  # type: ignore[assignment]
    stats: Counter = field(default_factory=Counter)
    require_stability: bool = True
    # Why the last capture attempt was dropped -- what a UI tells the reader
    # ("too dark", "hold still"), and the first thing to look at when a session
    # captures nothing.
    last_rejection: tuple[str, ...] = ()
    # The page corners found in the last frame, in that frame's pixel space, and
    # the size of that frame so a client can scale them to its own display.
    last_quad: list[tuple[int, int]] | None = None
    last_frame_size: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.tracker is None:
            self.tracker = StabilityTracker(self.cfg.stability)
        if self.deduper is None:
            self.deduper = dedup_mod.Deduplicator(self.cfg.dedup)

    @property
    def state(self) -> TurnState:
        return self.tracker.state

    def process(self, image: np.ndarray, index: int, timestamp: float) -> CaptureEvent | None:
        """Handle one frame. Returns a capture event at most once per page."""
        self.stats["frames"] += 1

        found = detect.locate(image, self.cfg.detect)
        quad = found.quad
        page_found = found.found
        # Published so a client can draw what the detector sees. Without it, "no
        # page detected" is an opaque refusal: a reader cannot tell whether the
        # page is too small in frame, too low-contrast against the surface, or
        # being found somewhere unexpected.
        self.last_quad = [(int(x), int(y)) for x, y in quad] if quad is not None else None
        self.last_frame_size = (int(image.shape[1]), int(image.shape[0]))
        if page_found:
            # Clear as soon as a page is seen again, not only on a capture: the
            # reason was being shown long after it stopped being true, so the
            # client told a reader "no page detected" while drawing the outline
            # of the page it had just found.
            self.last_rejection = ()
        else:
            self.stats["no_page"] += 1
            self.stats["detect_" + found.reason] += 1
            # The specific reason, not a generic refusal: a page too small and a
            # page running off the edges need opposite corrections from whoever
            # is holding the camera.
            self.last_rejection = (found.reason,)

        if self.require_stability:
            fired = self.tracker.update(image, page_found)
            if not fired:
                self.stats["state_" + self.tracker.state.value] += 1
                return None
        elif not page_found:
            return None

        assert quad is not None
        rectified = detect.rectify(image, quad, self.cfg.detect)

        # Quality is measured on the rectified page, not the whole frame, and only
        # for the one frame we are about to spend real money on. Measuring the
        # frame would average the page together with the desk behind it: a sharp
        # page on a featureless table scores as blurry, and the table's shadows
        # drag the exposure reading away from the paper we actually read.
        report = quality.assess(rectified, self.cfg.quality)
        if not report.ok:
            self.stats["rejected_quality"] += 1
            for reason in report.reasons:
                self.stats["quality_" + reason.split("(")[0]] += 1
            # Re-arm: this page still needs capturing, so do not sit in CAPTURED.
            self.tracker.state = TurnState.MOVING
            self.last_rejection = report.reasons
            return None

        phash = self.deduper.hash(rectified)
        if self.deduper.is_duplicate_hash(phash):
            self.stats["rejected_duplicate_hash"] += 1
            return None

        self.stats["captured"] += 1
        self.last_rejection = ()
        candidate = PageCandidate(
            frame_index=index,
            timestamp=timestamp or time.time(),
            quad=[(int(x), int(y)) for x, y in quad],
            quality=report,
            phash=phash,
        )
        return CaptureEvent(candidate=candidate, image=rectified)
