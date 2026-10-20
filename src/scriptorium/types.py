"""Core data contracts passed between pipeline stages.

Keeping these as plain dataclasses (no numpy in the serialisable ones) means the
edge loop and the worker can live in different processes -- or different machines --
without the stages needing to know about each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TurnState(str, Enum):
    """Where the page-turn state machine currently is."""

    SEARCHING = "searching"   # no page detected
    MOVING = "moving"         # a hand or the page itself is in motion
    SETTLING = "settling"     # motion stopped, waiting for it to hold
    STABLE = "stable"         # ready to capture
    CAPTURED = "captured"     # already shipped to the worker, waiting for a turn


@dataclass(frozen=True)
class QualityReport:
    """Why a frame is (or is not) worth OCRing."""

    sharpness: float          # variance of Laplacian, higher is sharper
    exposure: float           # mean luminance, 0..255
    clipped_ratio: float      # fraction of pixels blown out or crushed
    glare_ratio: float        # fraction of pixels in a specular hot spot
    ok: bool
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sharpness": round(self.sharpness, 2),
            "exposure": round(self.exposure, 2),
            "clipped_ratio": round(self.clipped_ratio, 4),
            "glare_ratio": round(self.glare_ratio, 4),
            "ok": self.ok,
            "reasons": list(self.reasons),
        }


@dataclass
class PageCandidate:
    """A rectified page image the edge loop decided is worth the heavy pipeline."""

    frame_index: int
    timestamp: float
    quad: list[tuple[int, int]]      # page corners in the original frame
    quality: QualityReport
    phash: str                        # perceptual hash, for dedup
    side: str = "single"              # single | left | right
    image_path: str | None = None     # set when handed across a process boundary


@dataclass
class OcrWord:
    text: str
    # bbox in rectified-page pixel space: x, y, w, h
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class OcrLine:
    words: list[OcrWord]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)


@dataclass
class OcrPage:
    lines: list[OcrLine]
    width: int
    height: int
    engine: str = "unknown"

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def confidence(self) -> float:
        lines = [line for line in self.lines if line.words]
        if not lines:
            return 0.0
        return sum(line.confidence for line in lines) / len(lines)


@dataclass
class PageResult:
    """One finished page: the OCR plus the provenance of how we got there."""

    page_number: int
    ocr: OcrPage
    candidate: PageCandidate
    dewarped: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
    image_path: str | None = None
    # Filled by the worker with words split across line breaks put back
    # together (see `export.speech.page_text`). Kept as a field rather than
    # computed here so this module stays a plain data contract with no imports
    # from the stages -- and so the repair happens once per page, not on every
    # access.
    reading_text: str = ""

    @property
    def text(self) -> str:
        """The page's text, hyphenated line breaks repaired.

        Every consumer -- exports, the eval harness, the API -- wants the text a
        correct reading produces, not the literal line-by-line transcription.
        `self.ocr.text` is still there for anyone who needs the raw layout.
        """
        return self.reading_text or self.ocr.text
