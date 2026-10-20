"""Page-turn state machine.

The central design decision of this project: we do NOT OCR the video stream.
A book is a sequence of discrete, static pages that happen to be delivered over a
video feed, so the camera loop's only job is to answer "is there a new page
sitting still in front of me?" and fire exactly once per page.

At 30 fps, naive per-frame OCR would be ~30x the compute for the same text, with
the extra problem of having to reconcile 30 slightly different transcriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import StabilityConfig
from ..types import TurnState


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

    def reset(self) -> None:
        self.state = TurnState.SEARCHING
        self._prev = None
        self._still_streak = 0
        self.last_motion = 0.0

    def update(self, image: np.ndarray, page_found: bool) -> bool:
        """Advance the machine by one frame. Returns True on a capture edge."""
        motion = self._motion(image)
        self.last_motion = motion

        if not page_found:
            self.state = TurnState.SEARCHING
            self._still_streak = 0
            return False

        still = motion <= self.cfg.motion_threshold

        if self.state is TurnState.CAPTURED:
            if motion >= self.cfg.turn_threshold:
                self.state = TurnState.MOVING
                self._still_streak = 0
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
        return True

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
        return float(diff.mean())
