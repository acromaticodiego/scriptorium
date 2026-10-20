"""Page deduplication.

A reader holding a book in front of a camera produces the same page over and over.
We use two independent signals, because each one fails on its own:

* perceptual hash -- cheap, runs before OCR, but two different pages of the same
  book share layout and can hash close together;
* text similarity -- accurate, but only available after OCR has run.

So we gate with the hash and confirm with the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np

from ..config import DedupConfig


def dhash(image: np.ndarray, size: int = 16) -> str:
    """Difference hash: compares each pixel to its right neighbour.

    Robust to exposure shifts and mild blur -- exactly what varies between
    consecutive frames of the same page -- unlike a plain average hash.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    width = size * size // 4
    return format(value, f"0{width}x")


def hamming(a: str, b: str) -> int:
    """Number of differing bits between two hex-encoded hashes."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def normalise_text(text: str) -> str:
    """Collapse whitespace and case so similarity reflects content, not formatting."""
    return " ".join(text.lower().split())


def text_similarity(a: str, b: str) -> float:
    na, nb = normalise_text(a), normalise_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


@dataclass
class Deduplicator:
    """Remembers accepted pages and rejects repeats."""

    cfg: DedupConfig = field(default_factory=DedupConfig)
    _hashes: list[str] = field(default_factory=list)
    _texts: list[str] = field(default_factory=list)

    def hash(self, image: np.ndarray) -> str:
        """Hash at the configured size, so callers cannot disagree about it."""
        return dhash(image, self.cfg.hash_size)

    def is_duplicate_hash(self, phash: str) -> bool:
        """Pre-OCR gate: cheap enough to run on every candidate."""
        return any(hamming(phash, seen) <= self.cfg.max_hash_distance for seen in self._hashes)

    def is_duplicate_text(self, text: str) -> bool:
        """Post-OCR confirmation.

        Short or empty pages are never treated as duplicates: two blank pages in a
        book are two real pages, and we must not collapse them.
        """
        if len(normalise_text(text)) < 40:
            return False
        return any(
            text_similarity(text, seen) >= self.cfg.min_text_similarity for seen in self._texts
        )

    def seed(self, hashes: list[str]) -> None:
        """Prime the gate with pages already stored.

        Reopening a book to add more pages starts a fresh session, and the page
        still lying under the camera is one it has never seen. Without the
        stored hashes it would be captured again as a duplicate of nothing.
        """
        self._hashes.extend(hashes)

    def accept(self, phash: str, text: str = "") -> None:
        self._hashes.append(phash)
        if text:
            self._texts.append(text)

    def __len__(self) -> int:
        return len(self._hashes)
