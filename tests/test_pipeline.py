"""End-to-end tests with the stub engine.

The stub means CI exercises the whole path -- detection, rectification, dewarping,
splitting, dedup, page numbering, PDF and text export -- with no OCR binary
installed. Accuracy is measured separately by `eval/run.py`, which needs a real
engine; these tests are about the plumbing being correct.
"""

from __future__ import annotations

import numpy as np
from pypdf import PdfReader

from eval import synth
from scriptorium.capture.source import Frame
from scriptorium.config import Config
from scriptorium.export import pdf as pdf_export
from scriptorium.export import text as text_export
from scriptorium.ocr.stub import StubEngine
from scriptorium.pipeline.book import BookSession


class _Stills:
    """A still FrameSource over pre-made frames."""

    is_still = True

    def __init__(self, images: list[np.ndarray]) -> None:
        self._images = images

    def frames(self):
        for index, image in enumerate(self._images):
            yield Frame(index=index, timestamp=float(index), image=image)


def test_single_page_end_to_end(scene):
    frame, _ = scene
    session = BookSession(engine=StubEngine())
    pages = session.run(_Stills([frame]))

    assert len(pages) == 1
    page = pages[0]
    assert page.page_number == 1
    assert page.ocr.lines, "stub engine should find text lines on a rendered page"
    assert page.timings_ms["ocr_ms"] >= 0
    assert session.summary()["pages"] == 1


def test_repeated_frames_yield_one_page(scene):
    """The same page held in front of the camera must not become ten pages."""
    frame, _ = scene
    session = BookSession(engine=StubEngine())
    pages = session.run(_Stills([frame] * 10))
    assert len(pages) == 1
    assert session.edge.stats["rejected_duplicate_hash"] == 9


def test_distinct_pages_are_kept_and_numbered(scene):
    text_a = "Primera pagina del libro. " * 20
    text_b = "Contenido completamente distinto en la segunda pagina. " * 15
    frame_a, truth_a = synth.make_scene(text_a, seed=1)
    frame_b, truth_b = synth.make_scene(text_b, seed=2)

    # Stub returns the ground truth so the text dedup gate sees real differences.
    session = BookSession(engine=StubEngine(text=truth_a))
    session.edge.require_stability = False
    session.feed(Frame(0, 0.0, frame_a))
    session.engine = StubEngine(text=truth_b)
    session.worker.engine = session.engine
    session.feed(Frame(1, 1.0, frame_b))

    assert [page.page_number for page in session.pages] == [1, 2]


def test_stability_gate_requires_a_still_page(scene):
    """With the gate on, a single frame is not enough -- it has to settle."""
    frame, _ = scene
    session = BookSession(engine=StubEngine())
    session.edge.require_stability = True
    assert session.feed(Frame(0, 0.0, frame)) == []
    for index in range(1, 6):
        session.feed(Frame(index, float(index), frame))
    assert len(session.pages) == 1


def test_pdf_has_a_searchable_text_layer(scene, tmp_path):
    frame, _ = scene
    sentence = "texto buscable dentro del pdf"
    session = BookSession(engine=StubEngine(text=sentence))
    session.run(_Stills([frame]))

    out = pdf_export.write_pdf(session.pages, session.images, tmp_path / "book.pdf")
    assert out.exists() and out.stat().st_size > 1000

    extracted = PdfReader(str(out)).pages[0].extract_text() or ""
    # Invisible render mode must still leave the words extractable.
    assert "buscable" in extracted.lower()


def test_text_exports(scene, tmp_path):
    frame, _ = scene
    session = BookSession(engine=StubEngine(text="linea uno\nlinea dos"))
    session.run(_Stills([frame]))

    txt = text_export.write_text(session.pages, tmp_path / "book.txt")
    md = text_export.write_markdown(session.pages, tmp_path / "book.md")
    js = text_export.write_json(session.pages, tmp_path / "book.json", session.summary())

    assert "linea uno" in txt.read_text(encoding="utf-8")
    assert "## Page 1" in md.read_text(encoding="utf-8")
    assert '"pages"' in js.read_text(encoding="utf-8")


def test_spread_becomes_two_pages(spread_page):
    """A capture of an open book must yield two numbered pages, not one."""
    session = BookSession(cfg=Config(), engine=StubEngine())
    session.edge.require_stability = False
    output = session.worker.process(_fake_candidate(), spread_page)
    assert len(output.results) == 2
    assert [r.candidate.side for r in output.results] == ["left", "right"]
    assert output.spread.is_spread


def test_split_can_be_disabled(spread_page):
    cfg = Config()
    cfg.split_spreads = False
    session = BookSession(cfg=cfg, engine=StubEngine())
    output = session.worker.process(_fake_candidate(), spread_page)
    assert len(output.results) == 1


def _fake_candidate():
    from scriptorium.types import PageCandidate, QualityReport

    return PageCandidate(
        frame_index=0,
        timestamp=0.0,
        quad=[],
        quality=QualityReport(500.0, 200.0, 0.0, 0.0, True),
        phash="0" * 16,
    )
