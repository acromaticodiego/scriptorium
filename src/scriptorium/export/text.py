"""Plain-text, Markdown and JSON export.

The text exports exist for two unglamorous but important reasons: they are what
the eval harness diffs against ground truth, and they are what a text-to-speech
reader consumes. JSON keeps the per-page provenance (confidence, timings, whether
dewarp fired) so a low-quality page can be found and rescanned.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import PageResult

# Marker written into a draft ground-truth file, and the reason it exists.
#
# Measuring accuracy on real photographs needs a human transcription to compare
# against, and typing a page from scratch is slow enough that it does not get
# done. Seeding the file with the pipeline's own output and correcting it is far
# faster -- and quietly circular if anyone forgets to correct it, because the
# hypothesis would then be scored against itself and return a perfect result.
#
# So a draft is marked, and the eval harness refuses to score a file still
# carrying the marker. Deleting the line is the act of claiming the file was
# reviewed, which is a thing a person has to do on purpose.
UNREVIEWED_MARKER = "# UNREVIEWED: correct the text below, then delete this line"


def write_text(pages: list[PageResult], path: str | Path, separator: str = "\n\n") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(pages, key=lambda p: p.page_number)
    out.write_text(separator.join(page.text for page in ordered), encoding="utf-8")
    return out


def write_markdown(pages: list[PageResult], path: str | Path, title: str = "Capture") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for page in sorted(pages, key=lambda p: p.page_number):
        lines.append(f"## Page {page.page_number}")
        lines.append("")
        lines.append(page.text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_json(pages: list[PageResult], path: str | Path, summary: dict | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary or {},
        "pages": [
            {
                "page_number": page.page_number,
                "side": page.candidate.side,
                "frame_index": page.candidate.frame_index,
                "dewarped": page.dewarped,
                "confidence": round(page.ocr.confidence, 4),
                "engine": page.ocr.engine,
                "quality": page.candidate.quality.as_dict(),
                "timings_ms": {k: round(v, 2) for k, v in page.timings_ms.items()},
                "text": page.text,
            }
            for page in sorted(pages, key=lambda p: p.page_number)
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_draft_truth(text: str, path: str | Path) -> Path:
    """Write a ground-truth draft seeded with recognised text, marked unreviewed.

    Never overwrites: a file already on disk may be a transcription someone
    corrected by hand, and regenerating drafts must not be able to destroy that
    work.
    """
    out = Path(path)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{UNREVIEWED_MARKER}\n{text}\n", encoding="utf-8")
    return out


def read_reviewed_truth(path: str | Path) -> str | None:
    """Return a reviewed transcription, or None if it still carries the marker."""
    content = Path(path).read_text(encoding="utf-8")
    if UNREVIEWED_MARKER in content:
        return None
    return content.strip()
