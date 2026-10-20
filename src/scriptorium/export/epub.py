"""EPUB export: reflowable text, which is what assistive technology wants.

The searchable PDF preserves the page -- useful for checking a transcription
against the original, and the wrong shape for actually *reading* one. A screen
reader has to infer order from coordinates, a phone cannot reflow a fixed page,
and nobody can change the type size. EPUB is the opposite trade: the layout is
gone and the text is a document, so it reflows, scales, and reads in order.

Written with the standard library's `zipfile` alone. The format is a zip with a
prescribed layout, and adding a dependency to write one would be the wrong
trade for a project whose point is the vision pipeline.

Paragraphs are recovered from the word boxes rather than guessed from
punctuation -- see `_paragraphs`.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from html import escape
from pathlib import Path
from uuid import uuid4

from ..types import OcrPage, PageResult

# A paragraph's last line is short: in justified text every other line reaches
# the measure. A line ending before this fraction of the widest line on the page
# is therefore taken to end a paragraph.
_SHORT_LINE_RATIO = 0.88

_CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

_CSS = """body { margin: 1em; line-height: 1.5; }
h1, h2 { font-weight: 600; }
p { margin: 0 0 0.8em; text-align: left; }
.page-number { color: #777; font-size: 0.8em; margin-bottom: 1.2em; }
"""


@dataclass
class EpubMetadata:
    title: str = "Scriptorium capture"
    language: str = "es"
    author: str = "Scriptorium"
    identifier: str = ""

    def __post_init__(self) -> None:
        if not self.identifier:
            self.identifier = f"urn:uuid:{uuid4()}"


def write_epub(
    pages: list[PageResult],
    path: str | Path,
    metadata: EpubMetadata | None = None,
) -> Path:
    """Write one EPUB 3 document, one section per page."""
    meta = metadata or EpubMetadata()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(pages, key=lambda page: page.page_number)
    documents = [
        (f"page_{page.page_number:04d}.xhtml", _page_document(page, meta))
        for page in ordered
    ]

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        # The mimetype entry must come first and be stored uncompressed. It is
        # how a reader identifies the file before unpacking it, and a compressed
        # or misplaced entry makes an archive that validators reject even though
        # every other byte is correct.
        archive.writestr(
            zipfile.ZipInfo("mimetype"), "application/epub+zip", zipfile.ZIP_STORED
        )
        archive.writestr("META-INF/container.xml", _CONTAINER)
        archive.writestr("OEBPS/style.css", _CSS)
        archive.writestr("OEBPS/content.opf", _package(documents, meta))
        archive.writestr("OEBPS/nav.xhtml", _navigation(documents, ordered, meta))
        for name, body in documents:
            archive.writestr(f"OEBPS/{name}", body)

    return out


def _paragraphs(page: OcrPage) -> list[str]:
    """Group the page's lines into paragraphs using line geometry.

    One `<p>` per OCR line would destroy reflow -- every line break becomes a
    hard paragraph break, so the text can never re-wrap to the reader's screen or
    type size. One `<p>` for the whole page loses the structure instead.

    The signal that separates them is in the word boxes: justified body text
    reaches the measure on every line *except* the last one of a paragraph. So a
    line whose right edge falls well short of the widest line on the page ends a
    paragraph.

    This is a heuristic, and it is wrong on text that is not justified, on
    headings, and on the last line of a page. Proper layout analysis is the real
    answer and is on the roadmap; this is the version that costs nothing and
    already makes the output reflow.
    """
    lines = [line for line in page.lines if line.words]
    if not lines:
        return []

    right_edges = [max(word.bbox[0] + word.bbox[2] for word in line.words) for line in lines]
    measure = max(right_edges)
    if measure <= 0:
        return [" ".join(line.text for line in lines)]

    paragraphs: list[str] = []
    current: list[str] = []
    for line, right in zip(lines, right_edges, strict=True):
        current.append(line.text)
        if right < measure * _SHORT_LINE_RATIO:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return [text for text in (p.strip() for p in paragraphs) if text]


def _page_document(page: PageResult, meta: EpubMetadata) -> str:
    paragraphs = _paragraphs(page.ocr) or [page.text]
    body = "\n".join(f"    <p>{escape(text)}</p>" for text in paragraphs)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"
      lang="{escape(meta.language)}" xml:lang="{escape(meta.language)}">
  <head>
    <title>{escape(meta.title)} - {page.page_number}</title>
    <link rel="stylesheet" type="text/css" href="style.css"/>
  </head>
  <body>
    <section epub:type="chapter">
    <p class="page-number" epub:type="pagebreak"
       id="page-{page.page_number}">{page.page_number}</p>
{body}
    </section>
  </body>
</html>
"""


def _package(documents: list[tuple[str, str]], meta: EpubMetadata) -> str:
    manifest = "\n".join(
        f'    <item id="p{index}" href="{name}" media-type="application/xhtml+xml"/>'
        for index, (name, _) in enumerate(documents, start=1)
    )
    spine = "\n".join(
        f'    <itemref idref="p{index}"/>' for index in range(1, len(documents) + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">{escape(meta.identifier)}</dc:identifier>
    <dc:title>{escape(meta.title)}</dc:title>
    <dc:language>{escape(meta.language)}</dc:language>
    <dc:creator>{escape(meta.author)}</dc:creator>
    <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
    <meta property="schema:accessMode">textual</meta>
    <meta property="schema:accessibilityFeature">readingOrder</meta>
    <meta property="schema:accessibilitySummary">Text recognised from a photographed
      book; reflowable and readable by assistive technology.</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="css" href="style.css" media-type="text/css"/>
{manifest}
  </manifest>
  <spine>
{spine}
  </spine>
</package>
"""


def _navigation(
    documents: list[tuple[str, str]], pages: list[PageResult], meta: EpubMetadata
) -> str:
    items = "\n".join(
        f'        <li><a href="{name}">Page {page.page_number}</a></li>'
        for (name, _), page in zip(documents, pages, strict=True)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"
      lang="{escape(meta.language)}" xml:lang="{escape(meta.language)}">
  <head>
    <title>{escape(meta.title)}</title>
  </head>
  <body>
    <nav epub:type="toc" id="toc">
      <h1>Pages</h1>
      <ol>
{items}
      </ol>
    </nav>
  </body>
</html>
"""
