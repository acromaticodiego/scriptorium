"""A dependency-free engine, so the pipeline is testable and demoable anywhere.

It does real line segmentation (ink projection + non-maximum suppression) and
emits one placeholder word per detected line. That is enough to exercise every
downstream stage -- PDF text layers, dedup, ordering, the API -- in CI, with no
OCR binary installed and no model download.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import OcrLine, OcrPage, OcrWord


class StubEngine:
    """Returns fixed text, or placeholder text positioned on the real text lines."""

    name = "stub"

    def __init__(self, text: str | None = None) -> None:
        self._text = text

    def run(self, image: np.ndarray) -> OcrPage:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        if self._text is not None:
            source_lines = [line for line in self._text.splitlines() if line.strip()]
            boxes = _even_boxes(len(source_lines), width, height)
        else:
            boxes = _detected_line_boxes(gray)
            # Tag the placeholder text with a fingerprint of this page.
            #
            # Without it two different pages that happen to have the same number
            # of lines produce identical text, and the text dedup gate correctly
            # -- but uselessly -- drops the second one. A real engine returns
            # different text for different pages, so the stub has to as well, or
            # it tests the pipeline against behaviour no engine has.
            tag = _page_tag(gray)
            source_lines = [f"{tag} linea {i + 1}" for i in range(len(boxes))]

        lines = [
            OcrLine(words=[OcrWord(text=text, bbox=box, confidence=1.0)])
            for text, box in zip(source_lines, boxes, strict=True)
        ]
        return OcrPage(lines=lines, width=width, height=height, engine=self.name)


def _page_tag(gray: np.ndarray) -> str:
    """Short, stable fingerprint of a page's ink layout."""
    thumb = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (thumb > thumb.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"pagina-{value:016x}"[:14]


def _even_boxes(count: int, width: int, height: int) -> list[tuple[int, int, int, int]]:
    if count == 0:
        return []
    step = height / (count + 1)
    box_height = max(1, int(step * 0.6))
    margin = int(width * 0.08)
    return [
        (margin, int(step * (i + 1) - box_height / 2), width - 2 * margin, box_height)
        for i in range(count)
    ]


def _detected_line_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape[:2]
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    profile = binary.sum(axis=1).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).ravel()
    if profile.max() <= 0:
        return []

    inside = profile > profile.max() * 0.2
    boxes: list[tuple[int, int, int, int]] = []
    start: int | None = None
    for y, flag in enumerate(inside):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            if y - start >= 3:
                boxes.append((0, start, width, y - start))
            start = None
    if start is not None and height - start >= 3:
        boxes.append((0, start, width, height - start))
    return boxes
