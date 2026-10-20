"""Curvature correction for an open book page.

The problem perspective rectification cannot solve
-----------------------------------------------------
A page lying open on a table is not a plane: it curls up towards the spine. A
homography maps a *plane* to a rectangle, so after rectification the text lines
are still bowed. OCR engines assume horizontal baselines; a bowed line gets cut
across by the line-segmentation step and the text comes back shredded. In our own
ablation this single stage is the largest error reduction in the pipeline.

Approach: text-line based dewarping
-----------------------------------
We do not need to recover the 3D shape of the paper. We only need the lines flat.

1. Binarise the page so ink is white on black.
2. Cut the page into vertical slices; in each slice take a horizontal projection
   of the ink and pick its peaks -- each peak is where a text line crosses.
3. Link peaks across neighbouring slices into continuous baselines, starting at
   the page centre (most text, least distortion) and walking outwards.
4. Fit a low-degree polynomial to each baseline.
5. Build a vertical displacement field by interpolating between the fitted
   baselines, and resample the image through it.

Everything is a projection, a polyfit and one `cv2.remap`: no training data, no
model to ship, a few milliseconds per page.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from ..config import DewarpConfig


@dataclass
class DewarpReport:
    """What the stage actually did -- surfaced in the API and the eval table."""

    applied: bool
    baselines: int
    max_shift_px: float
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "baselines": self.baselines,
            "max_shift_px": round(self.max_shift_px, 2),
            "reason": self.reason,
        }


def dewarp(image: np.ndarray, cfg: DewarpConfig | None = None) -> tuple[np.ndarray, DewarpReport]:
    """Flatten curved text lines. Returns (image, report).

    Falls back to the input untouched whenever the evidence is too weak -- a
    half-estimated displacement field is worse than none.
    """
    cfg = cfg or DewarpConfig()
    if not cfg.enabled:
        return image, DewarpReport(False, 0, 0.0, "disabled")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    ink = _binarise(gray)

    slice_centres, peaks_per_slice, spacing = _slice_peaks(ink, cfg.slices)
    if spacing is None:
        return image, DewarpReport(False, 0, 0.0, "no_text_lines")

    baselines = _link_baselines(slice_centres, peaks_per_slice, spacing)
    min_span = max(2, int(cfg.min_baseline_span * len(slice_centres)))
    baselines = [b for b in baselines if len(b) >= min_span]
    if len(baselines) < 2:
        return image, DewarpReport(False, len(baselines), 0.0, "too_few_baselines")

    curves, references = _fit_curves(baselines, slice_centres, width, cfg.poly_degree)
    # Displacement that each baseline needs in order to become horizontal.
    shifts = curves - references[:, None]

    shifts, consensus = _enforce_consensus(shifts)
    if consensus < cfg.min_consensus:
        return image, DewarpReport(
            False,
            len(baselines),
            float(np.abs(shifts).max()),
            f"inconsistent_baselines(consensus={consensus:.2f})",
        )

    max_shift = float(np.abs(shifts).max())
    if max_shift < cfg.min_shift_px:
        return image, DewarpReport(False, len(baselines), max_shift, "already_flat")
    # A displacement larger than the page is a failed fit, not a curved page.
    if max_shift > 0.25 * height:
        return image, DewarpReport(False, len(baselines), max_shift, "implausible_fit")

    field = _displacement_field(shifts, references, height)
    map_x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    map_y = (np.arange(height, dtype=np.float32)[:, None] + field).astype(np.float32)
    flat = cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    # Verify instead of predict.
    #
    # Everything above is an estimate of how the page is bent, and the checks on
    # it are all guesses about whether the estimate is trustworthy. But once the
    # correction exists we can simply *measure* whether it helped, with a metric
    # that does not come from the fit. If it did not, we keep the original: a
    # wrong correction resamples a good page and costs ~9% of line separation.
    #
    # This is what makes the aggressive thresholds above safe. They only decide
    # whether a correction is worth computing; this decides whether to keep it.
    before = line_separation(image)
    after = line_separation(flat)
    if after <= before * (1.0 + cfg.min_improvement):
        return image, DewarpReport(
            False,
            len(baselines),
            max_shift,
            f"no_improvement({before:.3f}->{after:.3f})",
        )

    return flat, DewarpReport(True, len(baselines), max_shift)


def line_separation(image: np.ndarray) -> float:
    """How cleanly text lines stand out in the horizontal ink projection.

    Straight lines put all their ink in a few rows and leave the interline gaps
    empty, so the row-wise profile swings hard between the two and its relative
    standard deviation is high. A bowed line smears its ink across neighbouring
    rows and flattens the profile.

    Cheap (one threshold and one sum, ~2 ms on a 1600 px page) and, importantly,
    derived from the pixels rather than from the baseline fit -- so it can be used
    to audit the fit's own output.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    profile = binary.sum(axis=1).astype(np.float32)
    return float(profile.std() / (profile.mean() + 1e-6))


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Ink -> 1, paper -> 0, tolerant of uneven lighting across the page."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    # Bridge the gaps between glyphs so a text line reads as one horizontal run.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    return (cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) > 0).astype(np.float32)


def _slice_peaks(
    ink: np.ndarray, n_slices: int
) -> tuple[np.ndarray, list[np.ndarray], float | None]:
    """Horizontal ink projection per vertical slice, reduced to text-line peaks."""
    width = ink.shape[1]
    n_slices = max(3, min(n_slices, width // 8))
    bounds = np.linspace(0, width, n_slices + 1).astype(int)
    centres = ((bounds[:-1] + bounds[1:]) / 2.0).astype(np.float32)

    # Estimate line spacing once, from the whole page: more text, less noise.
    spacing = _estimate_line_spacing(_smooth(ink.sum(axis=1)))
    if spacing is None:
        return centres, [], None

    min_distance = max(3, int(spacing * 0.6))
    radius = max(2, int(spacing * 0.4))
    peaks_per_slice = []
    for lo, hi in pairwise(bounds):
        profile = _smooth(ink[:, lo:hi].sum(axis=1))
        peaks = _find_peaks(profile, min_distance=min_distance)
        peaks_per_slice.append(_refine_peaks(profile, peaks, radius))
    return centres, peaks_per_slice, spacing


def _refine_peaks(profile: np.ndarray, peaks: np.ndarray, radius: int) -> np.ndarray:
    """Replace each integer peak with the ink centroid around it.

    `argmax` answers "which single row has the most ink", which moves by a few
    pixels depending on whether this slice happens to contain capitals,
    ascenders or a short last word. The centroid of the whole line body averages
    that away and lands between pixels.

    This is the dominant error term in the whole stage: on flat pages it is what
    the fit mistakes for curvature, and the mistake is *consistent* across lines,
    so the consensus check cannot catch it.
    """
    refined = []
    for peak in peaks:
        lo = max(0, int(peak) - radius)
        hi = min(profile.size, int(peak) + radius + 1)
        window = profile[lo:hi]
        total = float(window.sum())
        if total <= 0:
            refined.append(float(peak))
            continue
        rows = np.arange(lo, hi, dtype=np.float32)
        refined.append(float((window * rows).sum() / total))
    return np.array(refined, dtype=np.float32)


def _estimate_line_spacing(profile: np.ndarray) -> float | None:
    """Distance between consecutive text lines, from contiguous runs of ink.

    Every later threshold is derived from this number, so it has to be right.
    Peak-picking with a guessed minimum distance is not good enough: with a
    minimum of a few pixels it reports the ascender band, the x-height body and
    the descender band of one line as three lines, which underestimates the
    spacing by 3x. Everything downstream then fragments -- a 30-line page yields
    85 "baselines" and the consensus check sees noise where there is a page.

    Thresholding the profile and measuring between the *centres of ink runs*
    needs no such guess: one run is one line, whatever the type size.
    """
    if profile.size == 0 or profile.max() <= 0:
        return None

    inside = profile > profile.max() * 0.2
    centres: list[float] = []
    start: int | None = None
    for row, flag in enumerate(inside):
        if flag and start is None:
            start = row
        elif not flag and start is not None:
            centres.append((start + row - 1) / 2.0)
            start = None
    if start is not None:
        centres.append((start + len(inside) - 1) / 2.0)

    if len(centres) < 2:
        return None
    spacing = float(np.median(np.diff(centres)))
    return spacing if np.isfinite(spacing) and spacing >= 4 else None


def _smooth(profile: np.ndarray) -> np.ndarray:
    column = profile.astype(np.float32).reshape(-1, 1)
    return cv2.GaussianBlur(column, (1, 5), 0).ravel()


def _find_peaks(profile: np.ndarray, min_distance: int) -> np.ndarray:
    """Greedy non-maximum suppression over a 1D profile.

    Avoids a scipy dependency and lets us enforce a minimum spacing directly,
    which is what keeps the ascenders and descenders of one line from being
    reported as three separate lines.
    """
    if profile.size == 0:
        return np.empty(0, dtype=np.int32)
    peak_floor = max(profile.max() * 0.22, profile.mean())
    candidates = np.flatnonzero(profile >= peak_floor)
    if candidates.size == 0:
        return np.empty(0, dtype=np.int32)

    accepted: list[int] = []
    for index in candidates[np.argsort(-profile[candidates])]:
        if all(abs(int(index) - other) >= min_distance for other in accepted):
            accepted.append(int(index))
    return np.array(sorted(accepted), dtype=np.int32)


def _link_baselines(
    centres: np.ndarray, peaks_per_slice: list[np.ndarray], spacing: float
) -> list[dict[int, float]]:
    """Chain peaks from neighbouring slices into per-line tracks.

    Seeded at the middle slice and grown outwards: distortion grows towards the
    page edges, so starting in the middle means every step is a short, confident
    one instead of one long guess from an edge.
    """
    if not peaks_per_slice:
        return []
    middle = len(peaks_per_slice) // 2
    tolerance = spacing * 0.5

    tracks: list[dict[int, float]] = [{middle: float(y)} for y in peaks_per_slice[middle]]
    if not tracks:
        return []

    for direction in (-1, 1):
        index = middle + direction
        while 0 <= index < len(peaks_per_slice):
            peaks = peaks_per_slice[index]
            previous = index - direction
            if peaks.size:
                taken: set[int] = set()
                for track in tracks:
                    if previous not in track:
                        continue
                    predicted = track[previous]
                    distances = np.abs(peaks - predicted)
                    order = np.argsort(distances)
                    for choice in order:
                        if int(choice) in taken:
                            continue
                        if distances[choice] <= tolerance:
                            track[index] = float(peaks[choice])
                            taken.add(int(choice))
                        break
            index += direction

    # Convert slice indices to x coordinates.
    return [{int(i): y for i, y in track.items()} for track in tracks]


def _fit_curves(
    baselines: list[dict[int, float]], centres: np.ndarray, width: int, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit each baseline and sample it over every column.

    Returns (curves of shape (K, width), reference y per baseline). The reference
    is the mean of the curve, so flattening does not drift the page up or down.
    """
    xs_all = np.arange(width, dtype=np.float32)
    curves = []
    for track in sorted(baselines, key=lambda t: float(np.mean(list(t.values())))):
        slice_indices = sorted(track.keys())
        xs = np.array([centres[i] for i in slice_indices], dtype=np.float32)
        ys = np.array([track[i] for i in slice_indices], dtype=np.float32)
        effective_degree = min(degree, len(xs) - 1)
        coefficients = np.polyfit(xs, ys, effective_degree)
        # Evaluate only inside the span the baseline actually covers, holding the
        # end values beyond it. A line that stops mid-page (the last line of a
        # paragraph) otherwise gets its parabola extrapolated into the empty
        # margin, where it diverges: that alone produced ~19 px of invented
        # displacement on a perfectly flat page.
        clamped = np.clip(xs_all, float(xs.min()), float(xs.max()))
        curves.append(np.polyval(coefficients, clamped))

    curve_array = np.asarray(curves, dtype=np.float32)
    references = curve_array.mean(axis=1)
    return curve_array, references


def _enforce_consensus(shifts: np.ndarray) -> tuple[np.ndarray, float]:
    """Keep only the one bowing shape all the baselines agree on.

    Physical insight: a page curls as a surface, so every text line on it is bent
    by the *same* shape, differing only in amplitude. Measurement noise, by
    contrast, is independent per line. A rank-1 truncated SVD separates the two:
    the first singular component is the shared curvature, everything after it is
    noise, and the share of variance it explains tells us whether there was a
    common shape to find at all.

    Returns (denoised shifts, explained variance ratio in 0..1).
    """
    if shifts.shape[0] < 3:
        # Too few lines to tell a shape from noise; trust them as they are.
        return shifts, 1.0

    u, s, vt = np.linalg.svd(shifts, full_matrices=False)
    total = float((s**2).sum())
    if total <= 1e-9:
        return shifts, 1.0
    explained = float(s[0] ** 2 / total)
    rank_one = (u[:, :1] * s[0]) @ vt[:1]
    return rank_one.astype(np.float32), explained


def _displacement_field(
    shifts: np.ndarray, references: np.ndarray, height: int
) -> np.ndarray:
    """Interpolate per-baseline displacement into a dense (height, width) field.

    Between baselines: linear in y. Above the first and below the last: held
    constant, which keeps margins and headers from being stretched by an
    extrapolated polynomial that has no text to support it.
    """
    rows = np.arange(height, dtype=np.float32)
    if shifts.shape[0] == 1:
        return np.tile(shifts[0], (height, 1))

    upper = np.searchsorted(references, rows).clip(1, len(references) - 1)
    lower = upper - 1
    span = (references[upper] - references[lower]).clip(1e-6, None)
    weight = ((rows - references[lower]) / span).clip(0.0, 1.0)[:, None]
    return shifts[lower] * (1.0 - weight) + shifts[upper] * weight
