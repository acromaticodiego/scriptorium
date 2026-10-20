"""API contract tests.

Deliberately thin: they check the shape of the contract a client depends on
(status codes, session lifecycle, the PDF download), not recognition quality.
"""

from __future__ import annotations

import time

import cv2
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from scriptorium.api import app as app_module
from scriptorium.ocr.stub import StubEngine


@pytest.fixture()
def client(monkeypatch):
    # Pin the engine so the API tests never depend on what is installed locally.
    monkeypatch.setattr(app_module, "_engine", StubEngine(text="pagina de prueba"))
    app_module._sessions.clear()
    return TestClient(app_module.app)


@pytest.fixture()
def frame_bytes(scene):
    frame, _ = scene
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer.tobytes()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "stub" in body["engines_available"]


def test_session_lifecycle(client, frame_bytes):
    session_id = client.post("/v1/sessions").json()["session_id"]

    states = []
    for _ in range(8):
        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        states.append(response.json()["state"])

    assert "captured" in states, states
    status = client.get(f"/v1/sessions/{session_id}").json()
    assert status["summary"]["pages"] == 1

    pdf = client.get(f"/v1/sessions/{session_id}/pdf")
    assert pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"

    assert client.delete(f"/v1/sessions/{session_id}").status_code == 200
    assert client.get(f"/v1/sessions/{session_id}").status_code == 404


def test_reading_endpoints(client, frame_bytes):
    """The read-aloud contract: a speakable string plus boxes in page space."""
    session_id = client.post("/v1/sessions").json()["session_id"]
    for _ in range(8):
        client.post(
            f"/v1/sessions/{session_id}/frames",
            files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
        )
    assert client.get(f"/v1/sessions/{session_id}").json()["summary"]["pages"] == 1

    script = client.get(f"/v1/sessions/{session_id}/pages/1/script").json()
    assert script["text"]
    assert script["words"]
    assert script["page_width"] > 0 and script["page_height"] > 0

    # Offsets must index the string the client will hand to the speech engine.
    for word in script["words"]:
        assert script["text"][word["char_start"] : word["char_end"]] == word["text"]

    # Boxes must lie inside the page they are drawn over, or the highlight
    # lands off-image.
    for word in script["words"]:
        for x, y, w, h in word["boxes"]:
            assert 0 <= x <= script["page_width"]
            assert 0 <= y <= script["page_height"]
            assert w > 0 and h > 0

    image = client.get(f"/v1/sessions/{session_id}/pages/1/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_reading_a_missing_page_is_404(client, frame_bytes):
    session_id = client.post("/v1/sessions").json()["session_id"]
    assert client.get(f"/v1/sessions/{session_id}/pages/7/script").status_code == 404
    assert client.get(f"/v1/sessions/{session_id}/pages/7/image").status_code == 404


def test_epub_download(client, frame_bytes):
    session_id = client.post("/v1/sessions").json()["session_id"]
    for _ in range(8):
        client.post(
            f"/v1/sessions/{session_id}/frames",
            files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
        )

    response = client.get(f"/v1/sessions/{session_id}/epub")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/epub+zip"
    assert response.content[:2] == b"PK"  # a zip, as EPUB is


def test_epub_before_any_page_is_a_conflict(client):
    session_id = client.post("/v1/sessions").json()["session_id"]
    assert client.get(f"/v1/sessions/{session_id}/epub").status_code == 409


def test_pdf_before_any_page_is_a_conflict(client):
    session_id = client.post("/v1/sessions").json()["session_id"]
    assert client.get(f"/v1/sessions/{session_id}/pdf").status_code == 409


def test_stateless_page_endpoint_queues_a_job(client, frame_bytes):
    response = client.post(
        "/v1/pages", files={"file": ("page.jpg", frame_bytes, "image/jpeg")}
    )
    assert response.status_code == 202
    job_id = response.json()["id"]

    # Wait on a deadline, not on an iteration count. A fixed number of polls is a
    # race: the requests are far quicker than the ~100 ms the worker needs for a
    # page, so a loaded CI runner can exhaust the loop before the job finishes --
    # which is exactly how this test failed intermittently on one Python version
    # and not the others.
    job = _await_job(client, job_id)
    assert job["state"] == "done", job.get("error")
    assert job["result"]["summary"]["pages"] == 1


def test_bad_image_is_rejected(client):
    response = client.post(
        "/v1/pages", files={"file": ("x.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 400


def test_unknown_session(client):
    assert client.get("/v1/sessions/nope").status_code == 404


def _await_job(client, job_id: str, timeout: float = 30.0) -> dict:
    """Poll a queued job until it settles, or give up after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["state"] in {"done", "failed"} or time.monotonic() > deadline:
            return job
        time.sleep(0.05)
