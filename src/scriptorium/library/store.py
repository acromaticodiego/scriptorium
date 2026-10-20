"""The library: books that outlive the process that captured them.

A session is the wrong unit for a book. Sessions live in memory, so a book died
with the server, a reader could not put one down and come back to page 94
tomorrow, and nothing accumulated: a scan of one page and a scan of three
hundred were the same kind of thing. The useful unit is the **book** -- created
once, added to over many sittings, kept.

Stored as plain files, one directory per book:

    <root>/<book_id>/book.json          title, language, page count, timestamps
    <root>/<book_id>/pages/0001.json    the full recognition result
    <root>/<book_id>/pages/0001.png     the dewarped page those boxes refer to

Files rather than a database because of what the data is. The page images are
the bulk of it and belong on a filesystem or in object storage either way; the
records are a few kilobytes each and are only ever read by book. A directory is
also inspectable, backs up by copying, and needs nothing installed -- and the
day this needs to serve several replicas, the same layout maps onto object
storage without changing the shape of the data.

The full OCR result is kept, not just the text, because the word boxes are what
the read-aloud highlight needs. A library that stored only transcriptions could
show a book again but never read it back the same way.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..types import PageResult, page_as_dict, page_from_dict

# Book ids end up in URLs and in filesystem paths, so they are generated, never
# taken from user input. A title like "../../etc" must not be able to decide
# where files are written.
_ID_PATTERN = re.compile(r"^[a-z0-9]{12}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class BookRecord:
    """What the library knows about a book without opening its pages."""

    id: str
    title: str
    language: str = "spa"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    page_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "language": self.language,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "page_count": self.page_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BookRecord:
        return cls(
            id=data["id"],
            title=data["title"],
            language=data.get("language", "spa"),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            page_count=data.get("page_count", 0),
        )


class BookNotFound(KeyError):
    """No book with that id."""


@dataclass
class Library:
    """A directory of books."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ books

    def create(self, title: str, language: str = "spa") -> BookRecord:
        book_id = uuid.uuid4().hex[:12]
        record = BookRecord(id=book_id, title=title.strip() or "Untitled", language=language)
        (self._dir(book_id) / "pages").mkdir(parents=True, exist_ok=True)
        self._save_record(record)
        return record

    def list(self) -> list[BookRecord]:
        """Every book, most recently touched first."""
        records = []
        for entry in self.root.iterdir():
            if not entry.is_dir() or not _ID_PATTERN.match(entry.name):
                continue
            try:
                records.append(self.get(entry.name))
            except (BookNotFound, json.JSONDecodeError, OSError):
                # One unreadable book must not take the whole library with it.
                continue
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def get(self, book_id: str) -> BookRecord:
        path = self._dir(book_id) / "book.json"
        if not path.is_file():
            raise BookNotFound(book_id)
        return BookRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def rename(self, book_id: str, title: str) -> BookRecord:
        record = self.get(book_id)
        record.title = title.strip() or record.title
        record.updated_at = _now()
        self._save_record(record)
        return record

    def delete(self, book_id: str) -> None:
        directory = self._dir(book_id)
        if not directory.is_dir():
            raise BookNotFound(book_id)
        shutil.rmtree(directory)

    # ------------------------------------------------------------------ pages

    def add_page(self, book_id: str, page: PageResult, image: np.ndarray) -> int:
        """Append a page and return the number it was given.

        The number comes from the book, not from the caller: a capture session
        counts from 1 every time it starts, and a book continued tomorrow has to
        carry on from where it stopped.
        """
        record = self.get(book_id)
        number = record.page_count + 1

        pages = self._dir(book_id) / "pages"
        pages.mkdir(parents=True, exist_ok=True)

        stored = page_as_dict(page)
        stored["page_number"] = number
        # Image first: a record pointing at a missing image is a broken page,
        # while an image with no record is invisible and harmless.
        cv2.imwrite(str(pages / f"{number:04d}.png"), image)
        self._write_json(pages / f"{number:04d}.json", stored)

        record.page_count = number
        record.updated_at = _now()
        self._save_record(record)
        return number

    def pages(self, book_id: str) -> list[PageResult]:
        """Every stored page, in order."""
        self.get(book_id)  # raises if the book is unknown
        results = []
        for path in sorted((self._dir(book_id) / "pages").glob("*.json")):
            try:
                results.append(page_from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return results

    def page(self, book_id: str, number: int) -> PageResult | None:
        path = self._dir(book_id) / "pages" / f"{number:04d}.json"
        if not path.is_file():
            return None
        return page_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def page_image(self, book_id: str, number: int) -> np.ndarray | None:
        path = self._dir(book_id) / "pages" / f"{number:04d}.png"
        if not path.is_file():
            return None
        return cv2.imread(str(path))

    def page_images(self, book_id: str) -> dict[int, np.ndarray]:
        """All page images, keyed by number -- what the exporters expect."""
        images: dict[int, np.ndarray] = {}
        for path in sorted((self._dir(book_id) / "pages").glob("*.png")):
            image = cv2.imread(str(path))
            if image is not None:
                images[int(path.stem)] = image
        return images

    def page_hashes(self, book_id: str) -> list[str]:
        """Perceptual hashes of the stored pages.

        Handed to a resuming capture session so reopening a book does not
        re-capture the page still sitting under the camera.
        """
        hashes = []
        for path in sorted((self._dir(book_id) / "pages").glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                hashes.append(data["candidate"]["phash"])
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return hashes

    # ---------------------------------------------------------------- private

    def _dir(self, book_id: str) -> Path:
        if not _ID_PATTERN.match(book_id):
            # Ids are generated, so anything else is a malformed or hostile path.
            raise BookNotFound(book_id)
        return self.root / book_id

    def _save_record(self, record: BookRecord) -> None:
        self._write_json(self._dir(record.id) / "book.json", record.as_dict())

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        """Write via a temporary file and replace, so a crash cannot truncate it.

        A half-written book.json would make the book unreadable; os.replace is
        atomic on both POSIX and Windows.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)


def default_library() -> Library:
    """The library the API and CLI share, overridable for tests and deployment."""
    root = os.environ.get("SCR_LIBRARY_DIR") or (Path.cwd() / "data" / "books")
    return Library(Path(root))
