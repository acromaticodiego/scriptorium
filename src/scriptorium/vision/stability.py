"""Page-turn state machine.

The central design decision of this project: we do NOT OCR the video stream.
A book is a sequence of discrete, static pages that happen to be delivered over a
video feed, so the camera loop's only job is to answer "is there a new page
sitting still in front of me?" and fire exactly once per page.

At 30 fps, naive per-frame OCR would be ~30x the compute for the same text, with
the extra problem of having to reconcile 30 slightly different transcriptions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import StabilityConfig
from ..types import TurnState
from . import dedup


@dataclass
class StabilityTracker:
    """Feeds on consecutive frames and emits a capture decision.

    State transitions::

        SEARCHING  --page found-->      MOVING
        MOVING     --motion drops-->    SETTLING
        SETTLING   --held N frames-->   STABLE -> emit capture -> CAPTURED
        CAPTURED   --motion spikes-->   MOVING

    The CAPTURED state is what stops us from re-transcribing the page the reader
    is still holding: once a page is shipped, only a real disturbance re-arms us.
    """

    cfg: StabilityConfig = field(default_factory=StabilityConfig)
    state: TurnState = TurnState.SEARCHING
    last_motion: float = 0.0
    _prev: np.ndarray | None = field(default=None, repr=False)
    _still_streak: int = 0
    _captured_hash: str | None = field(default=None, repr=False)
    _recent: deque[float] = field(default_factory=lambda: deque(maxlen=30), repr=False)

    def reset(self) -> None:
        self.state = TurnState.SEARCHING
        self._prev = None
        self._still_streak = 0
        self._captured_hash = None
        self._recent.clear()
        self.last_motion = 0.0

    def still_bar(self) -> float:
        """How little movement counts as "still", for *this* camera right now.

        A fixed threshold silently assumes a noise floor. A camera on a stand
        reads about 0.5 on the motion metric; the same scene held in a hand never
        drops below roughly 3.5, because a hand is never still and the sensor is
        still adjusting exposure. A bar of 2.5 -- which is what the synthetic
        static captures suggested -- therefore means a handheld session is never
        still, never settles, and captures exactly once and then nothing, with
        the badge reporting "hold still" forever while the reader does.

        What matters is not absolute motion but whether the camera is as still as
        it is going to get, so the bar is drawn from the quietest moment in the
        recent past. On a stand the baseline is near zero and the configured
        floor applies unchanged; in a hand it rises to match.

        Capturing a slightly-moving frame is a safe failure here, because blur is
        measured separately on the rectified page and rejected there.
        """
        if not self._recent:
            return self.cfg.motion_threshold
        baseline = min(self._recent)
        return max(self.cfg.motion_threshold, baseline * self.cfg.motion_tolerance)

    def update(self, image: np.ndarray, page_found: bool) -> bool:
        """Advance the machine by one frame. Returns True on a capture edge."""
        motion = self._motion(image)
        self.last_motion = motion

        if not page_found:
            self.state = TurnState.SEARCHING
            self._still_streak = 0
            return False

        still_bar = self.still_bar()
        still = motion <= still_bar

        if self.state is TurnState.CAPTURED:
            turn_bar = max(self.cfg.turn_threshold, still_bar * 1.5)
            if motion >= turn_bar or self._content_changed(image):
                self.state = TurnState.MOVING
                self._still_streak = 0
                self._captured_hash = None
            return False

        if not still:
            self.state = TurnState.MOVING
            self._still_streak = 0
            return False

        self._still_streak += 1
        if self._still_streak < self.cfg.settle_frames:
            self.state = TurnState.SETTLING
            return False

        self.state = TurnState.CAPTURED
        self._still_streak = 0
        self._captured_hash = self._content_hash(image)
        return True

    def _content_changed(self, image: np.ndarray) -> bool:
        """Has the view changed since the capture, without much movement?

        Motion alone cannot see a page turn when the camera is on a stand and
        only the printed text changes. Measured on static captures: swapping one
        page for another moves the motion metric to 2.87, while the same page
        with nothing but sensor noise reads 2.41. The two populations overlap, so
        no motion threshold separates them -- a book on a stand would be captured
        once and never again.

        Comparing content does separate them, but only above a certain hash
        resolution, and the input resolution turns out not to matter. At 16x16
        the whole frame collapses to a thumbnail in which the text is gone;
        what is left is the gross layout, which is *identical* between two pages
        of the same book, so different pages hash closer (4 bits apart) than the
        same page does under noise (11). At 24x24 there is enough detail to see
        the text block, and the populations separate at every input size tested:
        same page at most 31 bits, different pages at least 38.

        Cheap either way, since it runs on the small frame the motion metric
        already produced.
        """
        if self._captured_hash is None:
            return False
        distance = dedup.hamming(self._content_hash(image), self._captured_hash)
        bits = self.cfg.content_hash_size**2
        return distance > bits * self.cfg.content_change_ratio

    def _content_hash(self, image: np.ndarray) -> str:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return dedup.dhash(gray, self.cfg.content_hash_size)

    def _motion(self, image: np.ndarray) -> float:
        """Mean absolute difference against the previous frame, on a small gray copy.

        Downscaling to 160x120 makes this robust to sensor noise and costs almost
        nothing -- the whole function runs well under a millisecond.
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        if self._prev is None:
            self._prev = small
            return float("inf")  # first frame: no reference, treat as motion
        diff = cv2.absdiff(small, self._prev)
        self._prev = small
        value = float(diff.mean())
        self._recent.append(value)
        return value
