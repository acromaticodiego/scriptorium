"""Frame triage: decide in ~1 ms whether a frame deserves the expensive pipeline.

This is the cheapest and highest-leverage stage. A blurred or glare-washed frame
costs the same to OCR as a good one but produces garbage that then has to be
detected and thrown away downstream, so we reject it here.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityConfig
from ..types import QualityReport


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: the standard no-reference focus measure.

    Scale-dependent, so we normalise the input to a fixed height first -- otherwise
    a 4K phone frame and a 720p webcam frame are not comparable.
    """
    norm = _normalise_height(gray, 720)
    return float(cv2.Laplacian(norm, cv2.CV_64F).var())


def exposure(gray: np.ndarray) -> tuple[float, float]:
    """Return (mean luminance, fraction of clipped pixels)."""
    mean = float(gray.mean())
    clipped = float(np.count_nonzero((gray <= 4) | (gray >= 251)) / gray.size)
    return mean, clipped


def glare_ratio(gray: np.ndarray) -> float:
    """Fraction of the image taken by specular hot spots.

    Three conditions have to hold at once, and each one rules out a different
    false positive:

    * **bright** -- a reflection is near the top of the range;
    * **locally flat** -- it has no print texture inside it;
    * **brighter than its own surroundings** -- this is the one that matters. A
      blank margin is bright and flat too, so brightness and flatness alone would
      flag every page with white space on it. Glare is *local*: it stands above
      the paper around it, while paper sits level with its own neighbourhood.
    """
    norm = _normalise_height(gray, 480)
    f = norm.astype(np.float32)

    bright = norm >= 246
    if not bright.any():
        return 0.0

    # Local variance via the identity Var(X) = E[X^2] - E[X]^2.
    mean = cv2.blur(f, (9, 9))
    mean_sq = cv2.blur(f * f, (9, 9))
    local_var = np.clip(mean_sq - mean * mean, 0, None)
    flat = local_var < 12.0

    # Large-scale background: a kernel a third of the image high, so a hot spot
    # cannot hide inside its own average.
    span = max(3, (norm.shape[0] // 3) | 1)
    background = cv2.blur(f, (span, span))
    stands_out = (f - background) > 10.0

    return float(np.count_nonzero(bright & flat & stands_out) / norm.size)


def assess(image: np.ndarray, cfg: QualityConfig | None = None) -> QualityReport:
    cfg = cfg or QualityConfig()
    gray = to_gray(image)

    sharp = sharpness(gray)
    mean, clipped = exposure(gray)
    glare = glare_ratio(gray)

    reasons: list[str] = []
    if sharp < cfg.min_sharpness:
        reasons.append(f"blurry(sharpness={sharp:.0f}<{cfg.min_sharpness:.0f})")
    if mean < cfg.min_exposure:
        reasons.append(f"too_dark(mean={mean:.0f})")
    if mean > cfg.max_exposure:
        reasons.append(f"too_bright(mean={mean:.0f})")
    if clipped > cfg.max_clipped_ratio:
        reasons.append(f"clipped({clipped:.1%})")
    if glare > cfg.max_glare_ratio:
        reasons.append(f"glare({glare:.1%})")

    return QualityReport(
        sharpness=sharp,
        exposure=mean,
        clipped_ratio=clipped,
        glare_ratio=glare,
        ok=not reasons,
        reasons=tuple(reasons),
    )


def _normalise_height(gray: np.ndarray, target: int) -> np.ndarray:
    h = gray.shape[0]
    if h == target:
        return gray
    scale = target / h
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
