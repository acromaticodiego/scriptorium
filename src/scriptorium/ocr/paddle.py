"""PaddleOCR backend.

Kept behind the same interface as Tesseract so the eval harness can compare the
two on identical inputs. Paddle is usually stronger on low-contrast and
non-Latin print but pulls in a heavyweight dependency, hence the extra.
"""

from __future__ import annotations

import numpy as np

from ..types import OcrLine, OcrPage, OcrWord
from .base import OcrUnavailable

# PaddleOCR language codes do not match Tesseract's.
_LANG_MAP = {"spa": "es", "eng": "en", "spa+eng": "es", "eng+spa": "en"}


class PaddleEngine:
    name = "paddle"

    def __init__(self, lang: str = "spa+eng") -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OcrUnavailable(
                "paddleocr is not installed (pip install 'scriptorium[paddle]')"
            ) from exc
        self.lang = _LANG_MAP.get(lang, "es")
        self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    def run(self, image: np.ndarray) -> OcrPage:
        height, width = image.shape[:2]
        raw = self._ocr.ocr(image, cls=True)
        lines: list[OcrLine] = []
        for page in raw or []:
            for entry in page or []:
                polygon, (text, confidence) = entry
                if not text.strip():
                    continue
                xs = [point[0] for point in polygon]
                ys = [point[1] for point in polygon]
                bbox = (
                    int(min(xs)),
                    int(min(ys)),
                    int(max(xs) - min(xs)),
                    int(max(ys) - min(ys)),
                )
                # Paddle returns one box per text line, not per word.
                lines.append(
                    OcrLine(words=[OcrWord(text=text, bbox=bbox, confidence=float(confidence))])
                )
        lines.sort(key=lambda line: line.words[0].bbox[1] if line.words else 0)
        return OcrPage(lines=lines, width=width, height=height, engine=self.name)
