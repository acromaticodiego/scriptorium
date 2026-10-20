"""Page localisation and perspective rectification.

Finds the quadrilateral of the page inside the frame and maps it to a rectangle.
Two strategies, tried in order, because neither is reliable alone:

1. edge-based -- works when the page contrasts with the surface it rests on;
2. brightness-based -- fallback for a white page on a light table, where edges
   are faint but the page is still the brightest connected blob.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import DetectConfig

Quad = np.ndarray  # shape (4, 2), float32, ordered tl, tr, br, bl

_WORK_HEIGHT = 600


def find_page(image: np.ndarray, cfg: DetectConfig | None = None) -> Quad | None:
    """Return the page corners in original-frame coordinates, or None."""
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

    quad = _find_by_edges(work, frame_area, cfg)
    if quad is None:
        quad = _find_by_brightness(work, frame_area, cfg)
    if quad is None:
        return None

    if scale < 1.0:
        quad = quad / scale
    return order_corners(quad)


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


def _plausible_area(quad: np.ndarray, frame_area: float, cfg: DetectConfig) -> bool:
    """Is this quad the right size to be a page, rather than noise or the frame?"""
    ratio = cv2.contourArea(quad.astype(np.float32)) / frame_area
    return cfg.min_area_ratio <= ratio <= cfg.max_area_ratio


def _find_by_edges(work: np.ndarray, frame_area: float, cfg: DetectConfig) -> Quad | None:
    median = float(np.median(work))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(work, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        candidate = approx.reshape(4, 2).astype(np.float32)
        if not _plausible_area(candidate, frame_area, cfg):
            continue
        return candidate
    return None


def _find_by_brightness(work: np.ndarray, frame_area: float, cfg: DetectConfig) -> Quad | None:
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    box = np.asarray(cv2.boxPoints(cv2.minAreaRect(largest)), dtype=np.float32)
    # The area check goes on the *box*, not the contour: minAreaRect can inflate
    # a ragged contour considerably, and it is the box that gets rectified.
    if not _plausible_area(box, frame_area, cfg):
        return None
    return box
