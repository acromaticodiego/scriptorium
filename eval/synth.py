"""Synthetic capture generator: ground truth without a scanner.

The expensive part of evaluating a scanning pipeline is labelled data -- somebody
has to photograph pages and type out what they say. We sidestep that: start from
a public-domain text, *render* it as a page, then simulate the camera (page curl,
tilt, blur, glare, uneven lighting, sensor noise) to produce a frame. The source
text is exact ground truth, for free, in unlimited quantity.

This also makes the degradations controllable, which is what turns the harness
from a single score into a diagnosis: we can sweep curl amplitude alone and watch
where dewarping stops rescuing the page.

Synthetic frames are not a substitute for real ones -- real paper has texture,
show-through and print variation. `eval/run.py` accepts a folder of real captures
for exactly that reason. The synthetic set is what makes the numbers *reproducible*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
)


@dataclass
class SceneConfig:
    """How hard to make one synthetic capture."""

    curl_px: float = 26.0        # peak vertical bow of the text lines
    tilt_deg: float = 6.0        # out-of-plane rotation of the page
    blur_sigma: float = 0.6
    glare: float = 0.0           # 0..1 intensity of a specular blob
    light_gradient: float = 0.25  # 0..1 strength of uneven illumination
    noise_sigma: float = 2.5
    page_fraction: float = 0.78  # how much of the frame the page fills

    @classmethod
    def easy(cls) -> SceneConfig:
        return cls(curl_px=6.0, tilt_deg=2.0, blur_sigma=0.3, light_gradient=0.1)

    @classmethod
    def hard(cls) -> SceneConfig:
        return cls(curl_px=44.0, tilt_deg=11.0, blur_sigma=1.1, glare=0.5, light_gradient=0.4)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    # Last resort: the bitmap default. Rendering is ugly and OCR scores will be
    # worse, so the harness reports which font it used.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def wrap(text: str, font, max_width: int, hyphenate: bool = True) -> list[tuple[str, bool]]:
    """Lay text out in lines. Returns (line, broken_by_hyphen) pairs.

    The flag is what lets the caller reconstruct the ground truth: a line ending
    in a hyphen because a word was split is *not* text containing a hyphen, and
    scoring against the hyphenated form would mark a correct transcription wrong.

    Hyphenation is on by default because typeset books do it on a large share of
    lines, and it is the artefact `export.speech` has to repair before reading a
    page aloud. A simulator that never produces it leaves that repair untested
    against a real OCR engine.
    """
    lines: list[tuple[str, bool]] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append(("", False))
            continue
        current = ""
        for word in paragraph.split():
            trial = word if not current else current + " " + word
            if _text_width(font, trial) <= max_width or not current:
                current = trial
                continue

            head, tail = ("", word)
            if hyphenate:
                head, tail = _split_word(word, font, max_width - _text_width(font, current + " "))
            if head:
                lines.append((current + " " + head + "-", True))
                current = tail
            else:
                lines.append((current, False))
                current = word
        if current:
            lines.append((current, False))
    return lines


def _split_word(word: str, font, available: float) -> tuple[str, str]:
    """Break a word so the head plus a hyphen fits in `available` pixels.

    Deliberately naive -- it splits on width, not on syllables, so the result is
    typographically wrong in the way a cheap word processor is wrong. That is
    fine, and arguably better for a test corpus: the pipeline must repair the
    break wherever it falls, and real OCR has no idea where a syllable boundary
    was supposed to be either.
    """
    # Too short to split, or no room: not worth breaking.
    if len(word) < 7 or available <= 0:
        return "", word
    for cut in range(len(word) - 3, 2, -1):
        head = word[:cut]
        if _text_width(font, head + "-") <= available:
            return head, word[cut:]
    return "", word


def render_page(
    text: str,
    width: int = 1240,
    height: int = 1754,
    font_size: int = 34,
    margin: int = 110,
    hyphenate: bool = True,
) -> tuple[np.ndarray, str]:
    """Render text as a clean page. Returns (BGR image, text actually rendered).

    The second value matters: only the lines that fit on the page are ground
    truth. Scoring against text that was never printed would inflate the error
    rate and make every later comparison meaningless.
    """
    font = load_font(font_size)
    # Paper at 222, not 255: photographed paper is never pure white, and a
    # renderer that starts at the clipping point makes every later lighting
    # effect saturate -- which would quietly turn the quality gate into a
    # test of the simulator instead of a test of the capture.
    image = Image.new("L", (width, height), color=222)
    draw = ImageDraw.Draw(image)

    line_height = int(font_size * 1.55)
    usable = width - 2 * margin
    lines = wrap(text, font, usable, hyphenate=hyphenate)

    y = margin
    truth_parts: list[str] = []
    pending_hyphen = False
    for line, broken in lines:
        if y + line_height > height - margin:
            break
        if line:
            draw.text((margin, y), line, font=font, fill=28)
            # Ground truth is the text a correct reading produces, which is not
            # what is printed: a word split across the break is one word, so the
            # hyphen is dropped and the halves are joined with no space.
            piece = line[:-1] if broken else line
            if pending_hyphen:
                truth_parts[-1] += piece
            else:
                truth_parts.append(piece)
            pending_hyphen = broken
        y += line_height

    page = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
    return page, "\n".join(truth_parts).strip()


def apply_curl(page: np.ndarray, amplitude: float) -> np.ndarray:
    """Bow the text lines the way an open book does near the spine.

    A single sine across the page width, strongest in the middle: the same shape
    `vision.dewarp` has to recover, but it only ever sees the result, never the
    parameter.
    """
    if amplitude <= 0:
        return page
    height, width = page.shape[:2]
    xs = np.arange(width, dtype=np.float32)
    shift = amplitude * np.sin(np.pi * xs / max(1, width - 1))
    # Slightly stronger towards the bottom, as a page lifted at the spine behaves.
    depth = np.linspace(0.85, 1.15, height, dtype=np.float32)[:, None]
    map_x = np.tile(xs, (height, 1))
    map_y = np.arange(height, dtype=np.float32)[:, None] + shift[None, :] * depth
    return cv2.remap(
        page, map_x, map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def place_on_desk(page: np.ndarray, tilt_deg: float, page_fraction: float) -> np.ndarray:
    """Put the page on a darker surface with a perspective tilt.

    Without this the page would fill the frame edge to edge and page detection
    would have nothing to find -- the harness would then be grading OCR alone
    instead of the pipeline.
    """
    page_h, page_w = page.shape[:2]
    frame_h = int(page_h / page_fraction)
    frame_w = int(page_w / page_fraction)

    rng = np.random.default_rng(7)
    grain = rng.integers(-8, 9, size=(frame_h, frame_w, 3), dtype=np.int16)
    desk = np.clip(70 + grain, 0, 255).astype(np.uint8)

    offset_x = (frame_w - page_w) // 2
    offset_y = (frame_h - page_h) // 2
    source = np.float32([[0, 0], [page_w, 0], [page_w, page_h], [0, page_h]])

    skew = np.tan(np.deg2rad(tilt_deg)) * page_h * 0.5
    destination = np.float32(
        [
            [offset_x + skew, offset_y],
            [offset_x + page_w, offset_y + skew * 0.4],
            [offset_x + page_w - skew * 0.6, offset_y + page_h],
            [offset_x, offset_y + page_h - skew * 0.2],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)

    warped = cv2.warpPerspective(page, matrix, (frame_w, frame_h), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(
        np.full((page_h, page_w), 255, np.uint8), matrix, (frame_w, frame_h)
    )
    # Soft edge so the page boundary is a realistic gradient, not a perfect step.
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (warped * alpha + desk * (1.0 - alpha)).astype(np.uint8)


def apply_lighting(frame: np.ndarray, gradient: float, glare: float) -> np.ndarray:
    if gradient <= 0 and glare <= 0:
        return frame
    height, width = frame.shape[:2]
    out = frame.astype(np.float32)

    if gradient > 0:
        # Centred on 1.0 so the lit side does not clip the paper to pure white.
        half = gradient * 0.45
        ramp_x = np.linspace(1.0 + half, 1.0 - half, width, dtype=np.float32)
        ramp_y = np.linspace(1.0 + half * 0.4, 1.0 - half * 0.4, height, dtype=np.float32)
        out *= (ramp_y[:, None] * ramp_x[None, :])[:, :, None]

    if glare > 0:
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = width * 0.68, height * 0.3
        sigma = min(width, height) * 0.13
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)))
        out += (blob * 255.0 * glare)[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def make_scene(text: str, cfg: SceneConfig | None = None, seed: int = 0) -> tuple[np.ndarray, str]:
    """Full synthetic capture. Returns (camera frame BGR, ground-truth text)."""
    cfg = cfg or SceneConfig()
    page, truth = render_page(text)
    page = apply_curl(page, cfg.curl_px)
    frame = place_on_desk(page, cfg.tilt_deg, cfg.page_fraction)
    frame = apply_lighting(frame, cfg.light_gradient, cfg.glare)

    if cfg.blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), cfg.blur_sigma)
    if cfg.noise_sigma > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, cfg.noise_sigma, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame, truth


def chunk_corpus(text: str, chunks: int, chars_per_chunk: int = 1400) -> list[str]:
    """Split a corpus into `chunks` page-sized pieces, on sentence boundaries.

    Returns the number of pieces asked for whenever the corpus has that many
    sentences, by dividing the available text evenly rather than cutting at a
    fixed size. The earlier version packed paragraphs up to `chars_per_chunk` and
    stopped when the text ran out, so `--pages 8` on this corpus quietly produced
    3 -- and a metrics table whose sample size silently disagreed with the flag
    that was supposed to set it.
    """
    sentences = _sentences(text)
    if not sentences or chunks <= 0:
        return []

    # Always fill to a full page rather than dividing the corpus evenly. Even
    # division would hand back the requested count by making every page sparse,
    # and a page holding four lines of text is not the thing being benchmarked:
    # it gives the dewarp fit almost no baselines and computes CER over so few
    # characters that one bad word swings the score. A short corpus should
    # produce fewer *real* pages and say so, which `eval/run.py` does.
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else current + " " + sentence
        if len(candidate) > chars_per_chunk and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
        if len(pieces) >= chunks:
            return pieces[:chunks]
    if current:
        pieces.append(current)
    return pieces[:chunks]


def _sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation, keeping the punctuation attached."""
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [part.strip() for part in parts if part.strip()]


def _text_width(font, text: str) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    return float(font.getsize(text)[0])  # pragma: no cover - very old Pillow
