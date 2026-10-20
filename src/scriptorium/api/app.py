"""HTTP API.

Two shapes of client, one pipeline:

* **stateless** -- `POST /v1/pages` with a single photo, get the text back. This is
  the integration surface: a mobile app, a batch digitisation script, a cron job.
* **session** -- `POST /v1/sessions`, then stream frames to
  `POST /v1/sessions/{id}/frames`. The server runs the edge loop and answers with
  the capture state, so a browser client can show "hold still / captured /
  turn the page" without shipping the vision code to the client.

A session is the *book*, not the request: pages accumulate, dedup state persists,
and the PDF is assembled at the end.

Sessions live in process memory here. That is the one piece which has to move to
shared storage before running more than one replica -- see README, Scaling.
"""

from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..capture.source import Frame
from ..config import Config
from ..export import epub as epub_export
from ..export import pdf as pdf_export
from ..export import speech
from ..jobs.memory import InMemoryQueue
from ..ocr.registry import available_engines, get_engine
from ..pipeline.book import BookSession
from ..types import PageResult

log = logging.getLogger(__name__)

app = FastAPI(
    title="Scriptorium",
    version="0.1.0",
    description="Contactless book digitisation: camera frames in, searchable PDF out.",
)

_queue = InMemoryQueue(workers=2)
_sessions: dict[str, BookSession] = {}
_engine = None


def _get_engine():
    """Lazy, cached engine. Model load is slow; do it once, not per request."""
    global _engine
    if _engine is None:
        cfg = Config()
        _engine = get_engine(cfg.ocr_engine, cfg.ocr_lang)
    return _engine


_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
if _WEB_DIR.is_dir():
    # Served from the same origin as the API so the browser client needs no CORS
    # setup and getUserMedia works without a separate dev server.
    app.mount("/app", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/app/" if _WEB_DIR.is_dir() else "/docs")


@app.get("/health")
def health() -> dict:
    engine = _get_engine()
    return {
        "status": "ok",
        "engine": getattr(engine, "name", "unknown"),
        "engines_available": available_engines(),
        "sessions": len(_sessions),
        "jobs": _queue.stats(),
    }


@app.post("/v1/pages")
async def recognise_page(file: UploadFile = File(...)) -> JSONResponse:
    """Stateless single-photo recognition, queued so the request returns at once."""
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


@app.post("/v1/sessions")
def create_session() -> dict:
    session = BookSession(engine=_get_engine())
    session_id = "s_" + uuid.uuid4().hex[:12]
    _sessions[session_id] = session
    return {"session_id": session_id, "engine": getattr(session.engine, "name", "unknown")}


@app.post("/v1/sessions/{session_id}/frames")
async def push_frame(session_id: str, file: UploadFile = File(...)) -> dict:
    """Feed one frame. The response tells the client what to show the reader."""
    session = _require_session(session_id)
    image = _decode(await file.read())

    index = int(session.edge.stats["frames"])
    pages = session.feed(Frame(index=index, timestamp=0.0, image=image))

    return {
        "state": session.edge.state.value,
        "motion": round(session.edge.tracker.last_motion, 2),
        "captured_now": [_page_payload(page) for page in pages],
        "total_pages": len(session.pages),
    }


@app.get("/v1/sessions/{session_id}")
def session_status(session_id: str) -> dict:
    session = _require_session(session_id)
    return {
        "session_id": session_id,
        "summary": session.summary(),
        "pages": [_page_payload(page) for page in session.pages],
    }


@app.get("/v1/sessions/{session_id}/pages/{page_number}/script")
def page_script(session_id: str, page_number: int) -> dict:
    """The read-aloud script for one page: text to speak + word geometry.

    The client must speak exactly the `text` returned here and resolve the
    engine's character offsets against these words -- see `export.speech` for
    why rebuilding the utterance client-side desynchronises the highlight.
    """
    page = _require_page(_require_session(session_id), page_number)
    return speech.build_script(page.ocr).as_dict()


@app.get("/v1/sessions/{session_id}/pages/{page_number}/image")
def page_image(session_id: str, page_number: int) -> Response:
    """The rectified, dewarped page the word boxes are measured against.

    Not the original frame: the boxes live in this image's coordinate space, so
    any client drawing a highlight has to draw it over exactly these pixels.
    """
    session = _require_session(session_id)
    _require_page(session, page_number)
    image = session.images.get(page_number)
    if image is None:
        raise HTTPException(status_code=404, detail="page image not retained")

    ok, buffer = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - encoder failure
        raise HTTPException(status_code=500, detail="could not encode page image")
    return Response(
        content=buffer.tobytes(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/v1/sessions/{session_id}/pdf")
def session_pdf(session_id: str) -> StreamingResponse:
    session = _require_session(session_id)
    if not session.pages:
        raise HTTPException(status_code=409, detail="session has no pages yet")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pdf_export.write_pdf(
            session.pages, session.images, Path(tmp) / "book.pdf", title=session_id
        )
        payload = path.read_bytes()

    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.pdf"'},
    )


@app.get("/v1/sessions/{session_id}/epub")
def session_epub(session_id: str) -> Response:
    """Reflowable text, for reading rather than for checking against the original."""
    session = _require_session(session_id)
    if not session.pages:
        raise HTTPException(status_code=409, detail="session has no pages yet")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = epub_export.write_epub(
            session.pages,
            Path(tmp) / "book.epub",
            epub_export.EpubMetadata(title=session_id, language=Config().ocr_lang.split("+")[0]),
        )
        payload = path.read_bytes()

    return Response(
        content=payload,
        media_type="application/epub+zip",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.epub"'},
    )


@app.delete("/v1/sessions/{session_id}")
def close_session(session_id: str) -> dict:
    _require_session(session_id)
    del _sessions[session_id]
    return {"closed": session_id}


class _OneFrame:
    """Minimal FrameSource wrapper for a single uploaded still."""

    is_still = True

    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def frames(self):
        yield Frame(index=0, timestamp=0.0, image=self._image)


def _require_session(session_id: str) -> BookSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return session


def _require_page(session: BookSession, page_number: int) -> PageResult:
    for page in session.pages:
        if page.page_number == page_number:
            return page
    raise HTTPException(status_code=404, detail=f"session has no page {page_number}")


def _decode(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return image


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
