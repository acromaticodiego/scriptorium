"""Tunables in one place, overridable by environment variable.

Each threshold carries the measurement that chose it. They were calibrated by
sweeping the parameter over the synthetic corpus (`eval/synth.py`) and reading
the resulting distributions, not by eyeballing a single photo -- so the comments
below cite numbers you can reproduce with `python -m eval.run`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass
class QualityConfig:
    """Thresholds for rejecting a frame before paying to recognise it.

    Calibrated against **downstream error, not appearance**. The first set of
    values came from looking at degraded pages and deciding they looked too bad
    to read, and they were far too strict: they rejected every page of the
    `hard` eval preset, including ones the pipeline went on to transcribe at
    CER 0.012. A gate that refuses pages the system can read is not caution, it
    is a reader stuck on "too much glare" with nothing captured.

    So each limit sits where measured CER actually starts to climb. Sweeping
    glare over the hard preset (Tesseract, Spanish, 3 pages per step):

        synth glare   measured glare   clipped    CER
             0.00            0.0344     0.022    0.000
             0.35            0.0450     0.118    0.001
             0.50            0.0661     0.163    0.012
             0.65            0.0720     0.201    0.111   <- degradation starts
             0.80            0.0762     0.230    0.170

    The limits below are drawn between the 0.50 and 0.65 rows.
    """

    min_sharpness: float = field(default_factory=lambda: _env_float("SCR_MIN_SHARPNESS", 90.0))
    min_exposure: float = field(default_factory=lambda: _env_float("SCR_MIN_EXPOSURE", 55.0))
    max_exposure: float = field(default_factory=lambda: _env_float("SCR_MAX_EXPOSURE", 225.0))
    max_clipped_ratio: float = field(default_factory=lambda: _env_float("SCR_MAX_CLIPPED", 0.18))
    max_glare_ratio: float = field(default_factory=lambda: _env_float("SCR_MAX_GLARE", 0.07))


@dataclass
class StabilityConfig:
    # Mean absolute frame difference below which we call the scene still.
    motion_threshold: float = field(default_factory=lambda: _env_float("SCR_MOTION_THRESH", 2.5))
    # Consecutive still frames required before we trust it (debounce).
    settle_frames: int = field(default_factory=lambda: _env_int("SCR_SETTLE_FRAMES", 4))
    # Motion above this after a capture means the page may have changed.
    #
    # Measured on rendered pages: sensor noise on a still page reads ~0.5, while
    # swapping one dense page for another reads ~6.4 -- so 9.0 (the first value we
    # tried) silently missed page turns that did not involve a hand sweep. The
    # bar belongs *low*, because re-arming is cheap (a rectify and a hash, no OCR)
    # and the perceptual-hash gate is what actually prevents a double capture,
    # whereas a missed turn loses a page with no way to notice.
    turn_threshold: float = field(default_factory=lambda: _env_float("SCR_TURN_THRESH", 4.0))


@dataclass
class DetectConfig:
    # A page quad must cover at least this fraction of the frame to count.
    #
    # 0.12 of the *area*, which is about 35-40% of the frame's width and height.
    # Below that there are too few pixels per character to recognise reliably,
    # and in practice it means filling a good part of the viewfinder with the
    # page rather than holding the book at arm's length from a wide-angle webcam.
    min_area_ratio: float = field(default_factory=lambda: _env_float("SCR_MIN_AREA", 0.12))
    # ...and no more than this, which catches a failure that otherwise passes
    # silently.
    #
    # When the paper and the surface under it are close in tone, the
    # brightness fallback thresholds both as one blob and returns the whole
    # frame as the "page". Measured on a light desk: 99.6% of the frame area,
    # reported as a successful detection. Rectifying that warps the desk along
    # with the page and hands the lot to OCR, so the run degrades instead of
    # failing -- and a degraded result with no error is far harder to notice
    # than a refusal. A genuine page photographed in full always leaves some
    # border; a quad reaching the frame edges means the boundary was not found.
    max_area_ratio: float = field(default_factory=lambda: _env_float("SCR_MAX_AREA", 0.92))
    # Rectified output height in pixels; width follows the detected aspect ratio.
    output_height: int = field(default_factory=lambda: _env_int("SCR_OUT_HEIGHT", 1600))


@dataclass
class DewarpConfig:
    enabled: bool = True
    # Number of vertical slices used to trace text baselines across the page.
    slices: int = field(default_factory=lambda: _env_int("SCR_DEWARP_SLICES", 32))
    # Polynomial degree fitted to each baseline.
    #
    # Degree 3 and 4 fit the bow better in isolation (recovery 39% vs 34% of the
    # line-straightness gap) but they also fit the *noise* better: the invented
    # displacement on flat pages then overlaps the real displacement on curved
    # ones, and the stage can no longer tell which is which. Degree 2 keeps those
    # two populations apart, which is what `min_shift_px` relies on.
    poly_degree: int = field(default_factory=lambda: _env_int("SCR_DEWARP_DEGREE", 2))
    # Baselines shorter than this fraction of page width are discarded as noise.
    min_baseline_span: float = field(default_factory=lambda: _env_float("SCR_DEWARP_SPAN", 0.45))
    # Fraction of the baselines' variance that must be explained by one shared
    # shape before we believe the page is curved rather than noisily measured.
    #
    # A weaker filter than it looks: flat pages also score high here (median
    # 0.86 against 0.94 for curved ones), because the measurement bias it cannot
    # see is itself shared between lines. It earns its place by catching the
    # genuinely broken fits -- pages where the baselines disagree outright.
    min_consensus: float = field(default_factory=lambda: _env_float("SCR_DEWARP_CONSENSUS", 0.70))
    # Minimum correction worth computing at all.
    #
    # Measured over 48 rendered pages (3 font sizes x 9 texts x 4 curvatures),
    # displacement magnitude does *not* separate flat pages from curved ones:
    # flat pages invent up to 14 px, while a real 15 px curl can produce as
    # little as 9 px. So this is not the safety check it first appears to be --
    # it only avoids paying for a resample that could not matter. The actual
    # safety check is `min_improvement`, which measures the finished correction
    # instead of guessing at it.
    min_shift_px: float = field(default_factory=lambda: _env_float("SCR_DEWARP_MIN_SHIFT", 3.0))
    # Relative gain in line separation required to keep a computed correction.
    #
    # Resampling a page that was already flat costs ~9% of its line separation,
    # so a correction that cannot demonstrate an improvement is discarded. 1% is
    # above the metric's run-to-run variation while still accepting any real gain
    # (a genuine 28 px curl improves it by 20-30%).
    min_improvement: float = field(
        default_factory=lambda: _env_float("SCR_DEWARP_MIN_GAIN", 0.01)
    )


@dataclass
class DedupConfig:
    # Perceptual hash side length; 16 -> 256 bits.
    #
    # 8x8 (64 bits) separates the two cases too, but with only 6 bits of margin:
    # at that resolution one flipped bit is 1.6% of the distance, so the decision
    # rests on a handful of pixels. 256 bits puts 22 bits between the
    # populations and makes the same decision a stable one.
    hash_size: int = field(default_factory=lambda: _env_int("SCR_DEDUP_HASH_SIZE", 16))
    # Pages closer than this fraction of the hash bits count as the same page.
    #
    # Deliberately tight, because the two gates have different jobs and the hash
    # gate's is the narrow one: reject *the same page in consecutive frames*,
    # which hashes within 3.1% of the bits (measured across varying tilt, focus
    # and sensor noise). Recognising the same *content* is the text gate's job.
    #
    # Setting it loosely looks safer and is not. Prose pages of one book sit
    # around 12-18% apart, but two pages sharing a layout pattern can come as
    # close as 6.6% -- and the failure modes are not symmetric:
    #
    #   * too tight -> the same page is OCRed twice, and the text gate catches
    #                  the repeat. Cost: one wasted OCR call.
    #   * too loose -> a real page is discarded before OCR ever sees it, so no
    #                  later stage can recover it. Cost: a page missing from the
    #                  book, with nothing in the output to say so.
    #
    # So the bar sits just above the same-page population and nowhere near the
    # different-page one.
    #
    # Only valid *after* rectification: on raw frames the same page drifts to
    # 14%, because the hash is then partly measuring camera pose.
    max_hash_distance_ratio: float = field(
        default_factory=lambda: _env_float("SCR_DEDUP_HASH_RATIO", 0.05)
    )
    # Text similarity above which two pages are "the same", even if hashes differ.
    min_text_similarity: float = field(default_factory=lambda: _env_float("SCR_DEDUP_TEXT", 0.88))

    @property
    def max_hash_distance(self) -> int:
        """Ratio expressed in bits, for the Hamming comparison."""
        return max(1, round(self.hash_size**2 * self.max_hash_distance_ratio))


@dataclass
class Config:
    quality: QualityConfig = field(default_factory=QualityConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    dewarp: DewarpConfig = field(default_factory=DewarpConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    split_spreads: bool = True
    ocr_engine: str = field(default_factory=lambda: os.environ.get("SCR_OCR_ENGINE", "auto"))
    ocr_lang: str = field(default_factory=lambda: os.environ.get("SCR_OCR_LANG", "spa+eng"))


DEFAULT = Config()
