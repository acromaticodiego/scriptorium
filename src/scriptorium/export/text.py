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
