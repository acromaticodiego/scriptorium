"""The OCR seam.

Recognition itself is a solved, commoditised problem -- the engineering value of
this project lives in the stages around it. So the engine sits behind a one-method
interface: Tesseract today, PaddleOCR or a fine-tuned TrOCR for handwriting
tomorrow, without the pipeline, the exporters or the eval harness noticing.

The interface also makes the ablation harness possible: it can hold the engine
fixed while toggling dewarp or splitting, so the measured delta belongs to our
code and not to a version bump in somebody's model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import OcrPage


@runtime_checkable
class OcrEngine(Protocol):
    """Anything that can turn a rectified page image into positioned text."""

    name: str

    def run(self, image: np.ndarray) -> OcrPage:
        """Recognise a single, already rectified and dewarped page."""
        ...


class OcrUnavailable(RuntimeError):
    """Raised when a requested engine is not installed on this machine."""
