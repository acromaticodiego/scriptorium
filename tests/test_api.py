"""API contract tests.

Deliberately thin on recognition quality -- that is the eval harness's job --
and thorough on the contract a client depends on: status codes, the book
lifecycle, and the one behaviour that only shows up across restarts, which is
that a book outlives the process that captured it.
"""

from __future__ import annotations

import time

import cv2
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from scriptorium.api import app as app_module
from scriptorium.library import Library
from scriptorium.ocr.stub import StubEngine


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Pin the engine so the API tests never depend on what is installed locally,
    # and give each test its own library so none can see another's books.
    monkeypatch.setattr(app_module, "_engine", StubEngine())
    monkeypatch.setattr(app_module, "_library", Library(tmp_path / "books"))
    app_module._captures.clear()
    return TestClient(app_module.app)


@pytest.fixture()
def frame_bytes(scene):
    frame, _ = scene
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer.tobytes()


def _capture(client, book_id: str, payload: bytes, frames: int = 8) -> dict:
    """Push a page at the camera and return the last response.

    `captured_now` is aggregated over the whole run rather than taken from the
    final frame: a page is captured once it settles, part-way through, and every
    frame after that correctly reports nothing new.
    """
    response: dict = {"captured_now": [], "total_pages": 0}
    captured: list = []
    for _ in range(frames):
        response = client.post(
            f"/v1/books/{book_id}/frames",
            files={"file": ("frame.jpg", payload, "image/jpeg")},
        ).json()
        captured.extend(response["captured_now"])
    response["captured_now"] = captured
    return response


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "stub" in body["engines_available"]


class TestBookLifecycle:
    def test_create_list_rename_delete(self, client):
        created = client.post("/v1/books", json={"title": "Don Quijote"})
        assert created.status_code == 201
        book_id = created.json()["id"]
        assert created.json()["page_count"] == 0

        assert [b["title"] for b in client.get("/v1/books").json()["books"]] == ["Don Quijote"]

        renamed = client.patch(f"/v1/books/{book_id}", json={"title": "El Quijote"})
        assert renamed.json()["title"] == "El Quijote"

        assert client.delete(f"/v1/books/{book_id}").status_code == 200
        assert client.get("/v1/books").json()["books"] == []
        assert client.get(f"/v1/books/{book_id}").status_code == 404

    def test_an_untitled_book_still_gets_a_name(self, client):
        assert client.post("/v1/books", json={}).json()["title"] == "Untitled"

    def test_renaming_to_nothing_is_rejected(self, client):
        book_id = client.post("/v1/books", json={"title": "A"}).json()["id"]
        assert client.patch(f"/v1/books/{book_id}", json={"title": "  "}).status_code == 400

    def test_unknown_book(self, client):
        assert client.get("/v1/books/ffffffffffff").status_code == 404

    def test_a_malformed_id_is_not_found_rather_than_an_error(self, client):
        """Ids are generated, so anything else is malformed -- or a path attack."""
        assert client.get("/v1/books/..%2F..%2Fetc").status_code in (404, 400)


class TestCapture:
    def test_frames_accumulate_into_the_book(self, client, frame_bytes):
        book_id = client.post("/v1/books", json={"title": "Libro"}).json()["id"]
        final = _capture(client, book_id, frame_bytes)

        assert final["total_pages"] == 1
        assert final["captured_now"]
        assert client.get(f"/v1/books/{book_id}").json()["book"]["page_count"] == 1

    def test_the_detected_outline_is_reported_for_the_client_to_draw(self, client, frame_bytes):
        book_id = client.post("/v1/books", json={"title": "Libro"}).json()["id"]
        final = _capture(client, book_id, frame_bytes, frames=3)

        assert final["quad"] is not None
        assert len(final["quad"]) == 4
        width, height = final["frame_size"]
        for x, y in final["quad"]:
            assert -width <= x <= 2 * width
            assert -height <= y <= 2 * height

    def test_reopening_a_book_does_not_recapture_its_last_page(self, client, frame_bytes):
        """The page still under the camera has already been stored.

        Capture state is per-process and rebuilt on demand, so without seeding it
        from the stored pages a reader who reopened a book would get a duplicate
        of wherever they left the book lying.
        """
        book_id = client.post("/v1/books", json={"title": "Libro"}).json()["id"]
        _capture(client, book_id, frame_bytes)
        assert client.get(f"/v1/books/{book_id}").json()["book"]["page_count"] == 1

        client.post(f"/v1/books/{book_id}/close")
        _capture(client, book_id, frame_bytes)
        assert client.get(f"/v1/books/{book_id}").json()["book"]["page_count"] == 1


class TestPersistence:
    def test_a_book_outlives_the_process(self, client, frame_bytes, tmp_path, monkeypatch):
        """The whole point of the library: pages are on disk, not in memory."""
        book_id = client.post("/v1/books", json={"title": "Persistente"}).json()["id"]
        _capture(client, book_id, frame_bytes)

        # A fresh library over the same directory stands in for a restart.
        monkeypatch.setattr(app_module, "_library", Library(tmp_path / "books"))
        app_module._captures.clear()
        fresh = TestClient(app_module.app)

        body = fresh.get(f"/v1/books/{book_id}").json()
        assert body["book"]["title"] == "Persistente"
        assert body["book"]["page_count"] == 1
        assert body["pages"][0]["text"]

    def test_word_boxes_survive_storage(self, client, frame_bytes):
        """Stored books have to be readable aloud, which needs the geometry."""
        book_id = client.post("/v1/books", json={"title": "Libro"}).json()["id"]
        _capture(client, book_id, frame_bytes)

        script = client.get(f"/v1/books/{book_id}/pages/1/script").json()
        assert script["words"]
        for word in script["words"]:
            assert script["text"][word["char_start"] : word["char_end"]] == word["text"]
            for _x, _y, width, height in word["boxes"]:
                assert width > 0 and height > 0


class TestExports:
    def test_pdf_and_epub(self, client, frame_bytes):
        book_id = client.post("/v1/books", json={"title": "Mi Libro"}).json()["id"]
        _capture(client, book_id, frame_bytes)

        pdf = client.get(f"/v1/books/{book_id}/pdf")
        assert pdf.status_code == 200
        assert pdf.content[:4] == b"%PDF"
        assert "Mi_Libro.pdf" in pdf.headers["content-disposition"]

        epub = client.get(f"/v1/books/{book_id}/epub")
        assert epub.status_code == 200
        assert epub.content[:2] == b"PK"

    def test_a_title_cannot_escape_the_download_filename(self, client, frame_bytes):
        """Titles are free text; a Content-Disposition filename is not."""
        book_id = client.post("/v1/books", json={"title": '../../x"; evil'}).json()["id"]
        _capture(client, book_id, frame_bytes)
        disposition = client.get(f"/v1/books/{book_id}/pdf").headers["content-disposition"]
        assert ".." not in disposition
        assert disposition.count('"') == 2

    def test_exporting_an_empty_book_is_a_conflict(self, client):
        book_id = client.post("/v1/books", json={"title": "Vacio"}).json()["id"]
        assert client.get(f"/v1/books/{book_id}/pdf").status_code == 409
        assert client.get(f"/v1/books/{book_id}/epub").status_code == 409

    def test_page_image_is_the_dewarped_page(self, client, frame_bytes):
        book_id = client.post("/v1/books", json={"title": "Libro"}).json()["id"]
        _capture(client, book_id, frame_bytes)

        image = client.get(f"/v1/books/{book_id}/pages/1/image")
        assert image.status_code == 200
        assert image.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert client.get(f"/v1/books/{book_id}/pages/9/image").status_code == 404


class TestStatelessRecognition:
    def test_a_single_photo_needs_no_book(self, client, frame_bytes):
        response = client.post(
            "/v1/pages", files={"file": ("page.jpg", frame_bytes, "image/jpeg")}
        )
        assert response.status_code == 202

        job = _await_job(client, response.json()["id"])
        assert job["state"] == "done", job.get("error")
        assert job["result"]["summary"]["pages"] == 1
        assert client.get("/v1/books").json()["books"] == []

    def test_bad_image_is_rejected(self, client):
        response = client.post(
            "/v1/pages", files={"file": ("x.jpg", b"not an image", "image/jpeg")}
        )
        assert response.status_code == 400

    def test_unknown_job(self, client):
        assert client.get("/v1/jobs/nope").status_code == 404


def _await_job(client, job_id: str, timeout: float = 30.0) -> dict:
    """Poll a queued job until it settles, or give up after `timeout` seconds.

    Waits on a deadline, not an iteration count: the requests are far quicker
    than the ~100 ms the worker needs, so a fixed number of polls is a race that
    a loaded CI runner loses.
    """
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["state"] in {"done", "failed"} or time.monotonic() > deadline:
            return job
        time.sleep(0.05)
