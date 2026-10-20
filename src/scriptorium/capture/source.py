"""Frame sources.

A live camera, a recorded video and a folder of stills all look the same to the
pipeline. That is what makes the system testable: the demo video in `samples/`
replays a real capture session deterministically in CI, and a reviewer with no
webcam can still run the whole thing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray


class FrameSource(Protocol):
    # Still sources (a folder of photos) carry no motion, so the stability gate
    # would starve waiting for a page to settle. Sources declare which they are
    # instead of the pipeline guessing from the class name.
    is_still: bool

    def frames(self) -> Iterator[Frame]:
        ...


@dataclass
class CameraSource:
    """Live webcam."""

    is_still = False
    device: int = 0
    width: int = 1920
    height: int = 1080

    def frames(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not capture.isOpened():
            raise RuntimeError(f"could not open camera device {self.device}")
        index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                yield Frame(index=index, timestamp=time.time(), image=image)
                index += 1
        finally:
            capture.release()


@dataclass
class VideoSource:
    """Recorded capture session; `stride` subsamples long videos."""

    is_still = False
    path: str | Path
    stride: int = 1

    def frames(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video {self.path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if index % self.stride == 0:
                    yield Frame(index=index, timestamp=index / fps, image=image)
                index += 1
        finally:
            capture.release()


@dataclass
class ImageDirSource:
    """A folder of stills, sorted by name. Each image is treated as a settled page.

    Still images carry no motion, so a tracker fed from here would never leave the
    MOVING state; `pipeline.book.BookSession` recognises this source and bypasses
    the stability gate.
    """

    path: str | Path

    def frames(self) -> Iterator[Frame]:
        folder = Path(self.path)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        if not files:
            raise RuntimeError(f"no images found in {folder}")
        for index, file in enumerate(files):
            image = cv2.imread(str(file))
            if image is None:
                continue
            yield Frame(index=index, timestamp=float(index), image=image)


def open_source(target: str | int, stride: int = 1) -> FrameSource:
    """Resolve a CLI argument into a source: a device index, a video or a folder."""
    if isinstance(target, int) or (isinstance(target, str) and target.isdigit()):
        return CameraSource(device=int(target))
    path = Path(target)
    if path.is_dir():
        return ImageDirSource(path)
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        return _SingleImageSource(path)
    return VideoSource(path, stride=stride)


@dataclass
class _SingleImageSource:
    is_still = True
    path: Path

    def frames(self) -> Iterator[Frame]:
        image = cv2.imread(str(self.path))
        if image is None:
            raise RuntimeError(f"could not read image {self.path}")
        yield Frame(index=0, timestamp=0.0, image=image)
