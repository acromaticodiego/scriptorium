"""Page localisation and perspective rectification.

Finds the quadrilateral of the page inside the frame and maps it to a rectangle.
Two strategies, tried in order, because neither is reliable alone:

1. edge-based -- works when the page contrasts with the surface it rests on;
2. brightness-based -- fallback for a white page on a light table, where edges
   are faint but the page is still the brightest connected blob.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import DetectConfig

Quad = np.ndarray  # shape (4, 2), float32, ordered tl, tr, br, bl

_WORK_HEIGHT = 600

# Why a candidate was turned down. These exist because "no page detected" is not
# actionable: a page too small in frame and a page running off the edges need
# opposite corrections from the person holding the camera, and telling them to
# fill more of the frame when the page already overflows it is worse than saying
# nothing.
NO_CANDIDATE = "no_candidate"
TOO_SMALL = "page_too_small"
FILLS_FRAME = "page_fills_frame"
OVERFLOWS = "page_overflows_frame"


@dataclass
class Detection:
    """Where the page is, or why it was not accepted."""

    quad: Quad | None
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.quad is not None


def locate(image: np.ndarray, cfg: DetectConfig | None = None) -> Detection:
    """Find the page, reporting why when it is not found."""
    cfg = cfg or DetectConfig()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    scale = _WORK_HEIGHT / gray.shape[0]
    work = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    work = cv2.GaussianBlur(work, (5, 5), 0)
    frame_area = float(work.shape[0] * work.shape[1])

    quad, reason = _find_by_edges(work, frame_area, cfg)
    reason = reason or NO_CANDIDATE
    if quad is None:
        quad, fallback_reason = _find_by_brightness(work, frame_area, cfg)
        # Prefer whichever strategy got furthest: a specific complaint about a
        # candidate it did find beats the other one shrugging.
        if quad is None:
            reason = fallback_reason if reason == NO_CANDIDATE else reason

    if quad is None:
        return Detection(None, reason)

    if scale < 1.0:
        quad = quad / scale
    return Detection(order_corners(quad))


def find_page(image: np.ndarray, cfg: DetectConfig | None = None) -> Quad | None:
    """Return the page corners in original-frame coordinates, or None."""
    return locate(image, cfg).quad


def rectify(image: np.ndarray, quad: Quad, cfg: DetectConfig | None = None) -> np.ndarray:
    """Perspective-warp the quad to an upright rectangle.

    The output aspect ratio comes from the detected edge lengths rather than being
    fixed, so a landscape spread does not get squeezed into portrait.
    """
    cfg = cfg or DetectConfig()
    quad = order_corners(np.asarray(quad, dtype=np.float32))
    tl, tr, br, bl = quad

    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if height <= 1 or width <= 1:
        raise ValueError("degenerate page quad")

    out_h = cfg.output_height
    out_w = max(1, round(out_h * (width / height)))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC)


def order_corners(points: np.ndarray) -> Quad:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Sorting by angle around the centroid is stable for any rotation under 45 deg,
    unlike the common sum/difference trick which breaks on tilted pages.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    ordered = pts[np.argsort(angles)]
    # argsort over atan2 starts at -pi (left, slightly above centre); rotate so the
    # first point is the one closest to the top-left corner of the bounding box.
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def _judge(quad: np.ndarray, frame_area: float, width: int, height: int,
           cfg: DetectConfig) -> str:
    """Accept a candidate, or name what is wrong with it."""
    ratio = cv2.contourArea(quad.astype(np.float32)) / frame_area
    if ratio < cfg.min_area_ratio:
        return TOO_SMALL
    if ratio > cfg.max_area_ratio:
        return FILLS_FRAME
    if not _corners_inside(quad, width, height, cfg):
        return OVERFLOWS
    return ""


def _corners_inside(quad: np.ndarray, width: int, height: int, cfg: DetectConfig) -> bool:
    """Are the page's corners actually in the picture?

    This is a precondition for rectification, not a heuristic about what pages
    look like. The perspective transform maps four corners to a rectangle; if a
    corner sits on the frame boundary then the real corner is somewhere outside
    the image, the transform is fitted to a point that is not the corner, and the
    result is a warp of whatever happened to be at the edge.

    That failure is not subtle when you see it -- a diagonal smear of the book's
    edge with a black wedge where the transform sampled outside the frame -- but
    nothing upstream reports it, so the junk gets stored as a page.

    A page overflowing one edge is tolerated: a reader framing tightly will clip
    a margin, and the other three corners still pin the transform. Two or more is
    not recoverable.
    """
    margin_x = width * cfg.edge_margin_ratio
    margin_y = height * cfg.edge_margin_ratio
    xs, ys = quad[:, 0], quad[:, 1]
    touching = sum(
        [
            bool(xs.min() <= margin_x),
            bool(xs.max() >= width - margin_x),
            bool(ys.min() <= margin_y),
            bool(ys.max() >= height - margin_y),
        ]
    )
    return touching <= 1


def _find_by_edges(
    work: np.ndarray, frame_area: float, cfg: DetectConfig
) -> tuple[Quad | None, str]:
    median = float(np.median(work))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(work, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    reason = ""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        candidate = approx.reshape(4, 2).astype(np.float32)
        complaint = _judge(candidate, frame_area, work.shape[1], work.shape[0], cfg)
        if complaint:
            # Keep the complaint about the largest candidate seen, which is the
            # one most likely to be the page.
            reason = reason or complaint
            continue
        return candidate, ""
    return None, reason


def _find_by_brightness(
    work: np.ndarray, frame_area: float, cfg: DetectConfig
) -> tuple[Quad | None, str]:
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, NO_CANDIDATE
    largest = max(contours, key=cv2.contourArea)
    box = np.asarray(cv2.boxPoints(cv2.minAreaRect(largest)), dtype=np.float32)
    # The checks go on the *box*, not the contour: minAreaRect can inflate a
    # ragged contour considerably, and it is the box that gets rectified.
    complaint = _judge(box, frame_area, work.shape[1], work.shape[0], cfg)
    return (None, complaint) if complaint else (box, "")
