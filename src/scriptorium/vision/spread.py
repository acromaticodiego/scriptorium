"""Two-page spread handling.

A reader points the camera at an open book, which is two pages, not one. Feeding
the whole spread to OCR produces interleaved lines across the gutter -- the
classic failure where sentences from the left page continue into the right one.

The gutter is found as the column where the *ink* runs out: between the two
inner margins there is a band of blank paper that no body text crosses.

The first version looked for the spine's shadow instead -- a dark vertical
valley, on the reasoning that the paper curves away from the light there. It
worked about half the time. Measured over 11 real captures of one book on one
stand, the brightness drop at the gutter was 39-51 grey levels in five of them
and 1-7 in the other six: with flat or frontal lighting the spine simply casts
no shadow, so there was nothing to find and the spread went to OCR whole.

Ink does not depend on how the book is lit. Over those same captures the ink
measure scores 0.87-1.00 on spreads and 0.11-0.28 on single pages -- the two
populations are half the scale apart, and it finds all eleven.
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


def analyse(
    image: np.ndarray,
    min_aspect: float = 0.9,
    min_confidence: float = 0.7,
    min_gutter_span: float = 0.004,
) -> SpreadSplit:
    """Decide whether this rectified page is a two-page spread.

    `min_aspect` is a cheap pre-filter, not the decision: a single page is much
    taller than it is wide, so a portrait capture can be refused without looking
    for a gutter at all. The gutter is what actually decides.

    Getting that division of labour backwards is what made this stage a coin
    flip. The bar started at 1.15, which sounds safe -- "a spread must be wider
    than tall" -- but measured on real captures of an open book it sits in the
    middle of the spread population: they came out at 1.12 to 1.28, so some were
    split and others, of the same book on the same stand, were not. A single page
    in those captures measured 0.56 to 0.62, so the two populations are nowhere
    near each other and any bar in that gap separates them.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    aspect = width / height
    if aspect < min_aspect:
        return SpreadSplit(False, None, 0.0, f"portrait(aspect={aspect:.2f})")

    profile = _ink_profile(gray)
    body = float(np.percentile(profile, 75))
    if body < height * 0.01:
        # Too little text to locate anything by its absence.
        return SpreadSplit(False, None, 0.0, "no_text")

    # Only the central band can plausibly hold the gutter.
    low, high = int(width * 0.35), int(width * 0.65)
    central = profile[low:high]
    if central.size < 8:
        return SpreadSplit(False, None, 0.0, "too_narrow")

    gutter_x = low + int(np.argmin(central))
    valley = float(central.min())
    # 1.0 means the gutter column is completely free of ink; 0.0 means it is as
    # inked as the body text, i.e. the centre falls inside a text column.
    confidence = 1.0 - valley / body

    if confidence < min_confidence:
        return SpreadSplit(False, None, confidence, f"no_gutter(conf={confidence:.2f})")

    # Depth is not enough on its own. A wide single page whose text runs edge to
    # edge also has a emptiest-central-column, formed by the gaps between words
    # and the ragged line ends, and measured across widths and texts it reaches
    # 0.74 against 0.87 for the thinnest real gutter -- too close, and it moves
    # with the wording.
    #
    # What a gutter has and a chance gap does not is *width*: two inner margins
    # meeting make a band of blank paper several columns across, while a gap
    # between words is one narrow dip. Measured, that separates completely:
    # every wide continuous page gave 0.0% of the page width, every real spread
    # between 0.9% and 8.7%.
    span = _valley_width(profile, gutter_x, body) / width
    if span < min_gutter_span:
        return SpreadSplit(False, None, confidence, f"gutter_too_narrow(span={span:.1%})")
    return SpreadSplit(True, gutter_x, confidence)


def _valley_width(profile: np.ndarray, centre: int, body: float) -> int:
    """How many contiguous columns around `centre` stay nearly free of ink."""
    threshold = body * 0.25
    left = centre
    while left > 0 and profile[left] < threshold:
        left -= 1
    right = centre
    while right < len(profile) - 1 and profile[right] < threshold:
        right += 1
    return right - left


def _ink_profile(gray: np.ndarray) -> np.ndarray:
    """Ink per column, smoothed enough to ignore the gaps between words."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    counts = (binary > 0).sum(axis=0).astype(np.float32)
    return cv2.GaussianBlur(counts.reshape(-1, 1), (1, 41), 0).ravel()


def split(image: np.ndarray, gutter_x: int, margin: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Cut a spread into (left, right) pages, trimming the gutter shadow itself."""
    width = image.shape[1]
    left_end = max(1, min(width - 1, gutter_x - margin))
    right_start = max(1, min(width - 1, gutter_x + margin))
    return image[:, :left_end], image[:, right_start:]


def split_if_spread(
    image: np.ndarray,
    min_aspect: float = 0.9,
    min_confidence: float = 0.7,
    min_gutter_span: float = 0.004,
) -> tuple[list[np.ndarray], SpreadSplit]:
    """Convenience wrapper: returns one page, or two in reading order."""
    result = analyse(image, min_aspect, min_confidence, min_gutter_span)
    if not result.is_spread or result.gutter_x is None:
        return [image], result
    left, right = split(image, result.gutter_x)
    return [left, right], result
