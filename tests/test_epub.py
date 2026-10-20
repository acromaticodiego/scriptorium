"""EPUB export tests.

EPUB is a format with rules a reader enforces silently: get the mimetype entry
wrong and the file is rejected with no indication which byte was at fault. So
these tests check the structure a validator checks, not just that a zip appeared.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

import pytest

from scriptorium.export.epub import EpubMetadata, _paragraphs, write_epub
from scriptorium.types import OcrLine, OcrPage, OcrWord, PageCandidate, PageResult, QualityReport


def word(text: str, x: int, y: int, width: int) -> OcrWord:
    return OcrWord(text=text, bbox=(x, y, width, 40), confidence=0.9)


def make_page(number: int, lines: list[OcrLine], text: str = "") -> PageResult:
    ocr = OcrPage(lines=lines, width=1080, height=1600, engine="stub")
    return PageResult(
        page_number=number,
        ocr=ocr,
        candidate=PageCandidate(0, 0.0, [], QualityReport(0, 0, 0, 0, True), "0" * 64),
        dewarped=True,
        reading_text=text or ocr.text,
    )


@pytest.fixture()
def book() -> list[PageResult]:
    # Two full-measure lines then a short one: one paragraph of three lines.
    full_a = OcrLine([word("En un lugar de la Mancha de cuyo nombre", 60, 100, 960)])
    full_b = OcrLine([word("no quiero acordarme vivia un hidalgo", 60, 160, 960)])
    short = OcrLine([word("de lanza en astillero.", 60, 220, 300)])
    second = OcrLine([word("Segunda pagina del libro.", 60, 100, 400)])
    return [make_page(1, [full_a, full_b, short]), make_page(2, [second])]


class TestStructure:
    def test_mimetype_is_first_and_uncompressed(self, book, tmp_path):
        """The rule a reader checks before unpacking anything else."""
        out = write_epub(book, tmp_path / "book.epub")
        with zipfile.ZipFile(out) as archive:
            first = archive.infolist()[0]
            assert first.filename == "mimetype"
            assert first.compress_type == zipfile.ZIP_STORED
            assert archive.read("mimetype") == b"application/epub+zip"

    def test_required_files_are_present(self, book, tmp_path):
        out = write_epub(book, tmp_path / "book.epub")
        with zipfile.ZipFile(out) as archive:
            names = set(archive.namelist())
        assert {"META-INF/container.xml", "OEBPS/content.opf", "OEBPS/nav.xhtml"} <= names
        assert "OEBPS/page_0001.xhtml" in names
        assert "OEBPS/page_0002.xhtml" in names

    def test_every_xml_document_parses(self, book, tmp_path):
        """Unescaped OCR text is the obvious way to produce a broken EPUB."""
        out = write_epub(book, tmp_path / "book.epub")
        with zipfile.ZipFile(out) as archive:
            for name in archive.namelist():
                if name.endswith((".xml", ".opf", ".xhtml")):
                    ET.fromstring(archive.read(name))

    def test_text_with_xml_characters_is_escaped(self, tmp_path):
        page = make_page(1, [OcrLine([word("a < b & c > d", 60, 100, 900)])])
        out = write_epub([page], tmp_path / "book.epub")
        with zipfile.ZipFile(out) as archive:
            body = archive.read("OEBPS/page_0001.xhtml").decode("utf-8")
            ET.fromstring(body)
        assert "&amp;" in body
        assert "a < b" not in body

    def test_spine_lists_every_page_in_order(self, book, tmp_path):
        out = write_epub(book, tmp_path / "book.epub")
        with zipfile.ZipFile(out) as archive:
            opf = ET.fromstring(archive.read("OEBPS/content.opf"))
        ns = {"opf": "http://www.idpf.org/2007/opf"}
        refs = [item.get("idref") for item in opf.findall(".//opf:spine/opf:itemref", ns)]
        assert refs == ["p1", "p2"]

    def test_metadata_is_carried(self, book, tmp_path):
        meta = EpubMetadata(title="Don Quijote", language="es", author="Cervantes")
        out = write_epub(book, tmp_path / "book.epub", meta)
        with zipfile.ZipFile(out) as archive:
            opf = archive.read("OEBPS/content.opf").decode("utf-8")
        assert "Don Quijote" in opf
        assert "<dc:language>es</dc:language>" in opf
        assert "Cervantes" in opf

    def test_identifier_is_unique_per_document(self, book, tmp_path):
        first = EpubMetadata().identifier
        second = EpubMetadata().identifier
        assert first != second and first.startswith("urn:uuid:")

    def test_text_survives_into_the_document(self, book, tmp_path):
        out = write_epub(book, tmp_path / "book.epub")
        with zipfile.ZipFile(out) as archive:
            body = archive.read("OEBPS/page_0001.xhtml").decode("utf-8")
        assert "En un lugar de la Mancha" in body
        assert 'lang="es"' in body


class TestParagraphs:
    def test_a_short_last_line_closes_a_paragraph(self):
        """Justified text reaches the measure on every line but the last."""
        page = OcrPage(
            lines=[
                OcrLine([word("linea completa que llega al margen", 60, 100, 960)]),
                OcrLine([word("otra linea completa hasta el margen", 60, 160, 960)]),
                OcrLine([word("cierre corto.", 60, 220, 250)]),
                OcrLine([word("nueva linea completa hasta el margen", 60, 280, 960)]),
                OcrLine([word("y su cierre.", 60, 340, 220)]),
            ],
            width=1080,
            height=1600,
        )
        paragraphs = _paragraphs(page)
        assert len(paragraphs) == 2
        assert paragraphs[0].startswith("linea completa")
        assert paragraphs[0].endswith("cierre corto.")

    def test_lines_are_joined_not_kept_as_separate_paragraphs(self):
        """One <p> per OCR line would destroy reflow, which is the whole point."""
        page = OcrPage(
            lines=[
                OcrLine([word("primera mitad de la frase", 60, 100, 960)]),
                OcrLine([word("y la segunda mitad.", 60, 160, 300)]),
            ],
            width=1080,
            height=1600,
        )
        assert _paragraphs(page) == ["primera mitad de la frase y la segunda mitad."]

    def test_empty_page_has_no_paragraphs(self):
        assert _paragraphs(OcrPage(lines=[], width=100, height=100)) == []

    def test_page_with_a_single_line_is_one_paragraph(self):
        page = OcrPage(lines=[OcrLine([word("solo.", 60, 100, 200)])], width=1080, height=1600)
        assert _paragraphs(page) == ["solo."]
