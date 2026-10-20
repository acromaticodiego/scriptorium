"""Transcription accuracy metrics.

CER (character error rate) and WER (word error rate) are edit distance normalised
by reference length: `(substitutions + insertions + deletions) / len(reference)`.
CER is the headline number for OCR because a single missed accent should not count
as badly as a destroyed word, and it keeps the comparison meaningful when the
engine mangles word boundaries.

Implemented here rather than pulled from a library so the normalisation is
explicit and auditable -- a harness that quietly lowercases and strips
punctuation reports a flattering number that does not survive contact with a
reader.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")
# Characters OCR engines legitimately render differently from the source text.
# ruff flags these as visually ambiguous characters; here they are the point --
# the whole table exists to map real typography onto its ASCII equivalent.
_EQUIVALENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',  # noqa: RUF001
    "–": "-", "—": "-", "…": "...", "«": '"', "»": '"',  # noqa: RUF001
}


def normalise(text: str, fold_case: bool = False, strip_accents: bool = False) -> str:
    """Canonicalise whitespace and typography. Accents are kept by default.

    Spanish text without accents is a different text, so folding them would hide
    real errors; the flag exists only to quantify how much of the error rate is
    accents alone.
    """
    text = unicodedata.normalize("NFC", text)
    for source, target in _EQUIVALENTS.items():
        text = text.replace(source, target)
    text = _WHITESPACE.sub(" ", text).strip()
    if fold_case:
        text = text.lower()
    if strip_accents:
        decomposed = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return text


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    """Levenshtein distance in O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


@dataclass
class ErrorRates:
    cer: float
    wer: float
    ref_chars: int
    ref_words: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "ref_chars": self.ref_chars,
            "ref_words": self.ref_words,
        }


def error_rates(reference: str, hypothesis: str, **kwargs) -> ErrorRates:
    ref = normalise(reference, **kwargs)
    hyp = normalise(hypothesis, **kwargs)

    ref_chars = len(ref)
    ref_words = ref.split()
    cer = edit_distance(ref, hyp) / ref_chars if ref_chars else (0.0 if not hyp else 1.0)
    wer = (
        edit_distance(ref_words, hyp.split()) / len(ref_words)
        if ref_words
        else (0.0 if not hyp.split() else 1.0)
    )
    # An engine can emit more text than the reference, so rates are not capped at 1
    # by construction; we clamp for reporting only.
    return ErrorRates(
        cer=min(cer, 1.0), wer=min(wer, 1.0), ref_chars=ref_chars, ref_words=len(ref_words)
    )
