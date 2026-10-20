"""Engine selection.

`auto` degrades gracefully: the best engine installed on this machine, falling
back to the stub so that a fresh clone runs end to end with zero system
dependencies. Every result records which engine produced it, so a demo can never
silently pass off stub output as real recognition.
"""

from __future__ import annotations

import logging

from .base import OcrEngine, OcrUnavailable
from .stub import StubEngine

log = logging.getLogger(__name__)

_PREFERENCE = ("tesseract", "paddle", "stub")


def get_engine(name: str = "auto", lang: str = "spa+eng") -> OcrEngine:
    if name == "auto":
        for candidate in _PREFERENCE:
            try:
                engine = _build(candidate, lang)
            except OcrUnavailable as exc:
                log.info("ocr engine %s unavailable: %s", candidate, exc)
                continue
            log.info("ocr engine selected: %s", engine.name)
            return engine
        raise OcrUnavailable("no OCR engine available")
    return _build(name, lang)


def _build(name: str, lang: str) -> OcrEngine:
    if name == "tesseract":
        from .tesseract import TesseractEngine

        return TesseractEngine(lang=lang)
    if name == "paddle":
        from .paddle import PaddleEngine

        return PaddleEngine(lang=lang)
    if name == "stub":
        return StubEngine()
    raise OcrUnavailable(f"unknown OCR engine: {name}")


def available_engines(lang: str = "spa+eng") -> dict[str, bool]:
    """Which engines this machine can actually run -- surfaced by /health."""
    status = {}
    for candidate in _PREFERENCE:
        try:
            _build(candidate, lang)
            status[candidate] = True
        except OcrUnavailable:
            status[candidate] = False
    return status


def diagnose(lang: str = "spa+eng") -> dict[str, str]:
    """Per-engine detail for `scriptorium doctor`.

    Reports *why* an engine is unusable, not just that it is. "available" and
    "not installed" are not enough here: the common real state is an engine that
    is installed but missing the language data for the text you are scanning,
    and that needs a different fix than installing the engine.
    """
    detail: dict[str, str] = {}
    for candidate in _PREFERENCE:
        try:
            engine = _build(candidate, lang)
        except OcrUnavailable as exc:
            detail[candidate] = str(exc).splitlines()[0]
            continue
        languages = getattr(engine, "available_languages", None)
        if callable(languages):
            codes = sorted(languages())
            detail[candidate] = "ready (languages: {})".format(", ".join(codes) or "none")
        else:
            detail[candidate] = "ready"
    return detail
