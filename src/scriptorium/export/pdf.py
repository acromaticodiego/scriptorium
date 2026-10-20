"""Searchable PDF export.

The output is the page image with an *invisible* text layer sitting exactly on top
of the printed words (PDF text render mode 3: draw nothing, but stay selectable).
That is what makes the result useful rather than decorative:

* Ctrl+F works, and selecting a sentence highlights the real printed sentence;
* a screen reader reads the page, which is the whole point of the accessibility
  mode -- a plain image PDF is invisible to assistive technology;
* the page still looks like the book, so a reader can check the transcription
  against the original instead of trusting it blindly.

Word boxes come from OCR in the dewarped page's pixel space, so the background
must be that same dewarped image (see `pipeline.worker.WorkerOutput`).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from ..types import PageResult

_FONT = "Helvetica"
_DEFAULT_DPI = 200


def write_pdf(
    pages: list[PageResult],
    images: dict[int, np.ndarray],
    path: str | Path,
    dpi: int = _DEFAULT_DPI,
    title: str = "Scriptorium capture",
) -> Path:
    """Write one searchable PDF. Page order follows `PageResult.page_number`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(out))
    pdf.setTitle(title)
    pdf.setCreator("Scriptorium")

    for page in sorted(pages, key=lambda p: p.page_number):
        image = images.get(page.page_number)
        if image is None:
            continue
        _draw_page(pdf, page, image, dpi)
        pdf.showPage()

    pdf.save()
    return out


def _draw_page(pdf: canvas.Canvas, page: PageResult, image: np.ndarray, dpi: int) -> None:
    height_px, width_px = image.shape[:2]
    scale = 72.0 / dpi  # pixels -> PDF points
    page_w, page_h = width_px * scale, height_px * scale
    pdf.setPageSize((page_w, page_h))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    pdf.drawImage(ImageReader(Image.fromarray(rgb)), 0, 0, width=page_w, height=page_h)

    # OCR coordinates can come from a differently sized image than the one we
    # draw (e.g. an engine that downscales internally); rescale rather than assume.
    sx = width_px / page.ocr.width if page.ocr.width else 1.0
    sy = height_px / page.ocr.height if page.ocr.height else 1.0

    for line in page.ocr.lines:
        for word in line.words:
            x, y, w, h = word.bbox
            if not word.text.strip() or w <= 0 or h <= 0:
                continue
            box_w = w * sx * scale
            box_h = h * sy * scale
            # Baseline sits slightly above the box bottom; 0.18 of the box height
            # is the usual descender allowance and keeps selection aligned.
            baseline = page_h - (y * sy + h * sy) * scale + box_h * 0.18
            font_size = max(1.0, box_h * 0.82)

            text = pdf.beginText(x * sx * scale, baseline)
            text.setTextRenderMode(3)  # invisible, but selectable and searchable
            text.setFont(_FONT, font_size)
            natural = stringWidth(word.text, _FONT, font_size)
            if natural > 0 and box_w > 0:
                # Stretch the glyphs to the printed word's real width so that a
                # text selection traces the ink instead of drifting off it.
                text.setHorizScale(max(10.0, min(400.0, 100.0 * box_w / natural)))
            text.textOut(word.text)
            pdf.drawText(text)
