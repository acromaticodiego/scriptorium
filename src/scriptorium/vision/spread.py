"""Two-page spread handling.

A reader points the camera at an open book, which is two pages, not one. Feeding
the whole spread to OCR produces interleaved lines across the gutter -- the
classic failure where sentences from the left page continue into the right one.

The gutter is found as a dark vertical valley near the centre: the paper curves
away from the light there, so it is reliably darker than the text columns, and
the signal survives even when the shadow is faint.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SpreadSplit:
    """Result of spread analysis."""

    is_spread: bool
    gutter_x: int | None
    confidence: float
    reason: str = ""


def analyse(image: np.ndarray, min_aspect: float = 1.15) -> SpreadSplit:
    """Decide whether this rectified page is a two-page spread.

    `min_aspect` guards the cheap case first: a single book page is taller than it
    is wide, so anything portrait is never a spread and we skip the search.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    aspect = width / height
    if aspect < min_aspect:
        return SpreadSplit(False, None, 0.0, f"portrait(aspect={aspect:.2f})")

    # Column-wise mean brightness, smoothed to suppress the ink of individual letters.
    profile = gray.mean(axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 31), 0).ravel()

    # Only the central third can plausibly hold the gutter.
    lo, hi = int(width * 0.35), int(width * 0.65)
    central = profile[lo:hi]
    if central.size < 8:
        return SpreadSplit(False, None, 0.0, "too_narrow")

    gutter_offset = int(np.argmin(central))
    gutter_x = lo + gutter_offset
    valley = float(central[gutter_offset])

    # How pronounced is the valley compared to the page body either side of it?
    page_level = float(np.median(np.concatenate([profile[:lo], profile[hi:]])))
    spread = float(profile.max() - profile.min()) or 1.0
    confidence = (page_level - valley) / spread

    if confidence < 0.25:
        return SpreadSplit(False, None, confidence, f"no_gutter(conf={confidence:.2f})")
    return SpreadSplit(True, gutter_x, confidence)


def split(image: np.ndarray, gutter_x: int, margin: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Cut a spread into (left, right) pages, trimming the gutter shadow itself."""
    width = image.shape[1]
    left_end = max(1, min(width - 1, gutter_x - margin))
    right_start = max(1, min(width - 1, gutter_x + margin))
    return image[:, :left_end], image[:, right_start:]


def split_if_spread(image: np.ndarray) -> tuple[list[np.ndarray], SpreadSplit]:
    """Convenience wrapper: returns one page, or two in reading order."""
    result = analyse(image)
    if not result.is_spread or result.gutter_x is None:
        return [image], result
    left, right = split(image, result.gutter_x)
    return [left, right], result
