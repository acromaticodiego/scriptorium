"""HTTP API, organised around books.

The unit is the **book**, not the request and not the session. A reader creates
one, adds pages to it over however many sittings, and comes back to it later;
the library on disk is what makes that possible (see `library.store`).

Live capture state -- the stability tracker, the dedup gate -- lives in memory
per open book, because it is about the last few frames and nothing else. It is
rebuilt on demand and seeded with the hashes of pages already stored, so
reopening a book does not re-capture the page still lying under the camera.

There is also a stateless `POST /v1/pages` for one-off recognition: a mobile
app, a batch script, anything wanting text out of a photograph without keeping
a book.

Sessions were the earlier model and were the wrong unit: a book died with the
server, nothing accumulated, and scanning one page and scanning three hundred
were the same kind of thing.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..capture.source import Frame
from ..config import Config
from ..export import epub as epub_export
from ..export import pdf as pdf_export
from ..export import speech
from ..jobs.memory import InMemoryQueue
from ..library import BookNotFound, Library, default_library
from ..ocr.registry import available_engines, get_engine
from ..pipeline.book import BookSession
from ..types import PageResult

log = logging.getLogger(__name__)

app = FastAPI(
    title="Scriptorium",
    version="0.2.0",
    description="Contactless book digitisation: a camera, a library of books, read aloud.",
)

_queue = InMemoryQueue(workers=2)
_captures: dict[str, BookSession] = {}
_engine = None
_library: Library | None = None


def library() -> Library:
    global _library
    if _library is None:
        _library = default_library()
    return _library


def _get_engine():
    """Lazy, cached engine. Model load is slow; do it once, not per request."""
    global _engine
    if _engine is None:
        cfg = Config()
        _engine = get_engine(cfg.ocr_engine, cfg.ocr_lang)
    return _engine


def _find_web_dir() -> Path | None:
    """Locate the browser client's static files.

    Resolving this relative to `__file__` was wrong in the way that only shows up
    once the package is no longer run from a checkout: in the container the code
    lives in site-packages while `web/` sits in the working directory, so the
    client silently was not served and `/` fell through to the API docs.
    """
    override = os.environ.get("SCR_WEB_DIR")
    candidates = [
        Path(override) if override else None,
        Path.cwd() / "web",
        Path(__file__).resolve().parents[3] / "web",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "index.html").is_file():
            return candidate
    return None


_WEB_DIR = _find_web_dir()
if _WEB_DIR is not None:
    # Served from the same origin as the API so the browser client needs no CORS
    # setup, and `getUserMedia` works because localhost counts as a secure
    # context -- a camera page served from any other host would need HTTPS.
    app.mount("/app", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
else:  # pragma: no cover - depends on deployment layout
    log.warning("browser client not found; / will redirect to the API docs")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/app/" if _WEB_DIR is not None else "/docs")


@app.get("/health")
def health() -> dict:
    engine = _get_engine()
    return {
        "status": "ok",
        "engine": getattr(engine, "name", "unknown"),
        "engines_available": available_engines(),
        "books": len(library().list()),
        "open_captures": len(_captures),
        "jobs": _queue.stats(),
    }


# ---------------------------------------------------------------------- books


@app.post("/v1/books", status_code=201)
def create_book(payload: dict = Body(default_factory=dict)) -> dict:
    title = str(payload.get("title") or "").strip()
    language = str(payload.get("language") or Config().ocr_lang.split("+")[0])
    return library().create(title=title, language=language).as_dict()


@app.get("/v1/books")
def list_books() -> dict:
    return {"books": [record.as_dict() for record in library().list()]}


@app.get("/v1/books/{book_id}")
def get_book(book_id: str) -> dict:
    record = _require_book(book_id)
    return {
        "book": record.as_dict(),
        "pages": [_page_payload(page) for page in library().pages(book_id)],
        "capturing": book_id in _captures,
    }


@app.patch("/v1/books/{book_id}")
def rename_book(book_id: str, payload: dict = Body(...)) -> dict:
    _require_book(book_id)
    title = str(payload.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title must not be empty")
    return library().rename(book_id, title).as_dict()


@app.delete("/v1/books/{book_id}")
def delete_book(book_id: str) -> dict:
    _require_book(book_id)
    library().delete(book_id)
    _captures.pop(book_id, None)
    return {"deleted": book_id}


# -------------------------------------------------------------------- capture


@app.post("/v1/books/{book_id}/frames")
async def push_frame(book_id: str, file: UploadFile = File(...)) -> dict:
    """Feed one frame into a book. The response tells the client what to show."""
    record = _require_book(book_id)
    image = _decode(await file.read())
    session = _capture_session(book_id)

    index = int(session.edge.stats["frames"])
    captured = session.feed(Frame(index=index, timestamp=0.0, image=image))

    stored = []
    for page in captured:
        number = library().add_page(book_id, page, session.images[page.page_number])
        # The library owns page numbering: a session counts from 1 every time it
        # starts, while a book continued tomorrow carries on from where it left.
        page.page_number = number
        stored.append(_page_payload(page))

    return {
        "state": session.edge.state.value,
        "motion": round(session.edge.tracker.last_motion, 2),
        # The detected page outline in the pixel space of the frame just sent, so
        # the client can draw what the detector sees rather than guess.
        "quad": session.edge.last_quad,
        "frame_size": session.edge.last_frame_size,
        "rejection": list(session.edge.last_rejection),
        "captured_now": stored,
        "total_pages": library().get(book_id).page_count if stored else record.page_count,
    }


@app.post("/v1/books/{book_id}/close")
def close_capture(book_id: str) -> dict:
    """Drop the in-memory capture state. The book and its pages stay."""
    _require_book(book_id)
    _captures.pop(book_id, None)
    return {"closed": book_id}


# ---------------------------------------------------------------------- pages


@app.get("/v1/books/{book_id}/pages/{page_number}/script")
def page_script(book_id: str, page_number: int) -> dict:
    """The read-aloud script: text to speak plus word geometry.

    The client must speak exactly the `text` returned here and resolve the
    engine's character offsets against these words -- see `export.speech` for
    why rebuilding the utterance client-side desynchronises the highlight.
    """
    return speech.build_script(_require_page(book_id, page_number).ocr).as_dict()


@app.get("/v1/books/{book_id}/pages/{page_number}/image")
def page_image(book_id: str, page_number: int) -> Response:
    """The rectified, dewarped page the word boxes are measured against."""
    _require_book(book_id)
    image = library().page_image(book_id, page_number)
    if image is None:
        raise HTTPException(status_code=404, detail=f"no page {page_number}")

    ok, buffer = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - encoder failure
        raise HTTPException(status_code=500, detail="could not encode page image")
    return Response(
        content=buffer.tobytes(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# -------------------------------------------------------------------- exports


@app.get("/v1/books/{book_id}/pdf")
def book_pdf(book_id: str) -> StreamingResponse:
    record = _require_book(book_id)
    pages = _require_pages(book_id)

    with tempfile.TemporaryDirectory() as tmp:
        path = pdf_export.write_pdf(
            pages, library().page_images(book_id), Path(tmp) / "book.pdf", title=record.title
        )
        payload = path.read_bytes()

    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_filename(record.title)}.pdf"'},
    )


@app.get("/v1/books/{book_id}/epub")
def book_epub(book_id: str) -> Response:
    """Reflowable text, for reading rather than for checking against the original."""
    record = _require_book(book_id)
    pages = _require_pages(book_id)

    with tempfile.TemporaryDirectory() as tmp:
        path = epub_export.write_epub(
            pages,
            Path(tmp) / "book.epub",
            epub_export.EpubMetadata(title=record.title, language=record.language),
        )
        payload = path.read_bytes()

    return Response(
        content=payload,
        media_type="application/epub+zip",
        headers={"Content-Disposition": f'attachment; filename="{_filename(record.title)}.epub"'},
    )


# ------------------------------------------------------- stateless recognition


@app.post("/v1/pages")
async def recognise_page(file: UploadFile = File(...)) -> JSONResponse:
    """One photo in, text out, without keeping a book. Queued so this returns at once."""
    image = _decode(await file.read())
    session = BookSession(engine=_get_engine())

    def work() -> dict:
        pages = session.run(_OneFrame(image))
        return {
            "pages": [_page_payload(page) for page in pages],
            "summary": session.summary(),
        }

    job = _queue.submit(work)
    return JSONResponse(status_code=202, content=job.as_dict())


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.as_dict()


# -------------------------------------------------------------------- helpers


class _OneFrame:
    """Minimal FrameSource wrapper for a single uploaded still."""

    is_still = True

    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def frames(self):
        yield Frame(index=0, timestamp=0.0, image=self._image)


def _capture_session(book_id: str) -> BookSession:
    """Live capture state for a book, created on the first frame.

    Seeded with the hashes of pages already stored: without them, reopening a
    book to add more pages would capture the page still under the camera as if
    it had never been seen.
    """
    session = _captures.get(book_id)
    if session is None:
        session = BookSession(engine=_get_engine())
        session.edge.deduper.seed(library().page_hashes(book_id))
        _captures[book_id] = session
    return session


def _require_book(book_id: str):
    try:
        return library().get(book_id)
    except BookNotFound:
        raise HTTPException(status_code=404, detail="unknown book") from None


def _require_pages(book_id: str) -> list[PageResult]:
    pages = library().pages(book_id)
    if not pages:
        raise HTTPException(status_code=409, detail="book has no pages yet")
    return pages


def _require_page(book_id: str, page_number: int) -> PageResult:
    _require_book(book_id)
    page = library().page(book_id, page_number)
    if page is None:
        raise HTTPException(status_code=404, detail=f"no page {page_number}")
    return page


def _decode(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return image


def _filename(title: str) -> str:
    """A title is free text; a Content-Disposition filename is not."""
    safe = "".join(char if char.isalnum() or char in " -_" else "-" for char in title).strip()
    return (safe or "book").replace(" ", "_")[:60] or uuid.uuid4().hex[:8]


def _page_payload(page: PageResult) -> dict:
    return {
        "page_number": page.page_number,
        "side": page.candidate.side,
        "dewarped": page.dewarped,
        "confidence": round(page.ocr.confidence, 4),
        "engine": page.ocr.engine,
        "timings_ms": {k: round(v, 1) for k, v in page.timings_ms.items()},
        "text": page.text,
    }
