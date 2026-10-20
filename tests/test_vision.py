"""Vision stage tests.

These assert *behaviour that matters to the output*, not implementation details:
a blurred frame must be rejected, a curled page must come out flatter, a spread
must be cut at the gutter. They are the regression net for the thresholds in
`config.py`, which are the easiest thing in the project to break by accident.
"""

from __future__ import annotations

import cv2
import numpy as np

from eval import synth
from scriptorium.config import DedupConfig, StabilityConfig
from scriptorium.vision import dedup, detect, dewarp, quality, spread

_PAGE_TEXT = "Texto de prueba para la pagina. " * 40


class TestQuality:
    def test_clean_page_passes(self, clean_page):
        report = quality.assess(clean_page)
        assert report.ok, report.reasons

    def test_blur_is_rejected(self, clean_page):
        blurred = cv2.GaussianBlur(clean_page, (0, 0), 4.0)
        report = quality.assess(blurred)
        assert not report.ok
        assert any("blurry" in reason for reason in report.reasons)

    def test_sharpness_orders_blur_levels(self, clean_page):
        gray = quality.to_gray(clean_page)
        mild = quality.sharpness(cv2.GaussianBlur(gray, (0, 0), 1.0))
        heavy = quality.sharpness(cv2.GaussianBlur(gray, (0, 0), 3.0))
        assert quality.sharpness(gray) > mild > heavy

    def test_glare_blob_is_detected(self, clean_page):
        glared = cv2.cvtColor(clean_page, cv2.COLOR_BGR2GRAY).copy()
        cv2.circle(glared, (400, 400), 220, 255, -1)
        assert quality.glare_ratio(glared) > quality.glare_ratio(
            cv2.cvtColor(clean_page, cv2.COLOR_BGR2GRAY)
        )

    def test_plain_white_paper_is_not_glare(self, clean_page):
        """Paper is bright but textured by print; only flat hot spots are glare."""
        assert quality.glare_ratio(quality.to_gray(clean_page)) < 0.03

    def test_gate_accepts_captures_the_pipeline_can_still_read(self):
        """The gate must not reject pages that OCR handles fine.

        Pins the calibration in `QualityConfig`: the `hard` preset (44 px curl,
        11 deg tilt, specular glare, defocus) measures CER 0.013 end to end, so
        it has to pass. An earlier, stricter set of thresholds rejected all of
        it, which left a live session stuck reporting glare and capturing
        nothing -- the failure this test exists to catch.
        """
        frame, _ = synth.make_scene(_PAGE_TEXT, synth.SceneConfig.hard())
        quad = detect.find_page(frame)
        assert quad is not None
        report = quality.assess(detect.rectify(frame, quad))
        assert report.ok, report.reasons

    def test_gate_still_rejects_what_breaks_ocr(self):
        """The other side of the same calibration: severe glare must be refused."""
        scene = synth.SceneConfig.hard()
        scene.glare = 0.9
        frame, _ = synth.make_scene(_PAGE_TEXT, scene)
        quad = detect.find_page(frame)
        assert quad is not None
        report = quality.assess(detect.rectify(frame, quad))
        assert not report.ok


class TestDetect:
    def test_finds_page_on_desk(self, scene):
        frame, _ = scene
        quad = detect.find_page(frame)
        assert quad is not None
        area = cv2.contourArea(quad.astype(np.float32))
        frame_area = frame.shape[0] * frame.shape[1]
        # The simulated page covers ~0.78^2 of the frame; allow generous slack.
        assert 0.3 < area / frame_area < 0.95

    def test_rectified_page_is_portrait(self, scene):
        frame, _ = scene
        quad = detect.find_page(frame)
        rectified = detect.rectify(frame, quad)
        assert rectified.shape[0] > rectified.shape[1]

    def test_a_quad_filling_the_frame_is_rejected(self):
        """Refuse rather than degrade when the page boundary was not found.

        On a surface close in tone to the paper, the brightness fallback
        thresholds both as one blob and returns the entire frame as the "page" --
        measured at 99.6% of the frame area, reported as success. Rectifying that
        warps the desk in with the page and quietly hands the lot to OCR. A
        degraded result with no error is much harder to notice than a refusal, so
        a quad reaching the frame edges is treated as no detection.
        """
        page, _ = synth.render_page(_PAGE_TEXT)
        # Paper on a desk only 17 grey levels away: almost no edge to find.
        height, width = page.shape[:2]
        frame = np.full((int(height / 0.7), int(width / 0.7), 3), 205, np.uint8)
        offset_y = (frame.shape[0] - height) // 2
        offset_x = (frame.shape[1] - width) // 2
        frame[offset_y : offset_y + height, offset_x : offset_x + width] = page

        quad = detect.find_page(frame)
        if quad is not None:
            area = cv2.contourArea(quad.astype(np.float32)) / (frame.shape[0] * frame.shape[1])
            assert area <= 0.92, f"returned {area:.1%} of the frame as a page"

    def test_a_page_too_small_in_frame_is_rejected(self):
        """Too few pixels per character to recognise; better to say so."""
        page, _ = synth.render_page(_PAGE_TEXT)
        height, width = page.shape[:2]
        frame = np.full((height * 4, width * 4, 3), 70, np.uint8)
        frame[:height, :width] = page
        assert detect.find_page(frame) is None

    def test_corner_ordering_is_rotation_stable(self):
        square = np.array([[10, 10], [110, 12], [112, 140], [8, 138]], dtype=np.float32)
        for shift in range(4):
            ordered = detect.order_corners(np.roll(square, shift, axis=0))
            assert ordered[0].tolist() == [10.0, 10.0]  # top-left stays first


class TestDewarp:
    def test_flattens_a_curled_page(self, dense_page):
        curled = synth.apply_curl(dense_page, 28.0)
        flat, report = dewarp.dewarp(curled)

        assert report.applied, report.reason
        assert report.baselines >= 3
        before, after = _baseline_bow(curled), _baseline_bow(flat)
        assert after < before * 0.5, f"bow went from {before:.1f}px to {after:.1f}px"

    def test_bow_is_reduced_across_curvatures(self, dense_page):
        """Guards against a regression that silently stops correcting anything."""
        for amplitude in (15.0, 28.0, 45.0):
            curled = synth.apply_curl(dense_page, amplitude)
            flat, report = dewarp.dewarp(curled)
            assert report.applied, f"{amplitude}px: {report.reason}"
            before, after = _baseline_bow(curled), _baseline_bow(flat)
            assert after < before, f"{amplitude}px: bow {before:.1f} -> {after:.1f}"

    def test_a_flat_page_is_left_untouched(self, dense_page):
        """A correction that cannot prove it helped must be discarded.

        Flat and curved pages cannot be told apart by displacement magnitude
        alone (see `DewarpConfig.min_shift_px`), so the stage computes the
        correction and then measures it. On an already-flat page there is nothing
        to gain and resampling costs ~9% of line separation, so the original
        pixels have to survive -- identity, not merely similarity.
        """
        flat, report = dewarp.dewarp(dense_page)
        assert not report.applied, report.as_dict()
        assert flat is dense_page

    def test_stronger_curl_needs_a_larger_correction(self, dense_page):
        shifts = []
        for amplitude in (15.0, 45.0):
            _, report = dewarp.dewarp(synth.apply_curl(dense_page, amplitude))
            assert report.applied, report.reason
            shifts.append(report.max_shift_px)
        assert shifts[1] > shifts[0], "a 45 px curl must need more correction than a 15 px one"

    def test_blank_page_is_safe(self):
        blank = np.full((800, 600, 3), 250, dtype=np.uint8)
        flat, report = dewarp.dewarp(blank)
        assert not report.applied
        assert flat.shape == blank.shape

    def test_disabled_is_a_no_op(self, dense_page):
        from scriptorium.config import DewarpConfig

        flat, report = dewarp.dewarp(dense_page, DewarpConfig(enabled=False))
        assert report.reason == "disabled"
        assert flat is dense_page


class TestSpread:
    def test_detects_and_splits_a_spread(self, spread_page):
        result = spread.analyse(spread_page)
        assert result.is_spread, result.reason
        middle = spread_page.shape[1] // 2
        assert abs(result.gutter_x - middle) < spread_page.shape[1] * 0.06

        left, right = spread.split(spread_page, result.gutter_x)
        assert left.shape[0] == right.shape[0] == spread_page.shape[0]
        assert left.shape[1] + right.shape[1] < spread_page.shape[1]

    def test_single_page_is_not_a_spread(self, clean_page):
        result = spread.analyse(clean_page)
        assert not result.is_spread
        assert "portrait" in result.reason


class TestDedup:
    def test_same_page_hashes_identically(self, clean_page):
        assert dedup.dhash(clean_page) == dedup.dhash(clean_page.copy())

    def test_hash_survives_exposure_change(self, clean_page):
        darker = np.clip(clean_page.astype(np.int16) - 25, 0, 255).astype(np.uint8)
        assert dedup.hamming(dedup.dhash(clean_page), dedup.dhash(darker)) <= 4

    def test_different_content_hashes_apart(self, clean_page):
        other = cv2.rotate(clean_page, cv2.ROTATE_180)
        assert dedup.hamming(dedup.dhash(clean_page), dedup.dhash(other)) > 8

    def test_deduplicator_gates_repeats(self, clean_page):
        deduper = dedup.Deduplicator(DedupConfig())
        phash = dedup.dhash(clean_page)
        assert not deduper.is_duplicate_hash(phash)
        deduper.accept(phash, "some page text that is long enough to compare properly")
        assert deduper.is_duplicate_hash(phash)

    def test_blank_pages_are_not_text_duplicates(self):
        deduper = dedup.Deduplicator(DedupConfig())
        deduper.accept("0" * 64, "")
        assert not deduper.is_duplicate_text("")

    def test_hash_threshold_sits_between_the_measured_populations(self):
        """Pins the calibration the dedup gate depends on.

        Hashes are compared on *rectified* pages, as the pipeline does, and the
        two scenes differ the way consecutive frames of a handheld capture do:
        different tilt, focus and sensor noise. The same page must land inside
        the threshold and a different page outside it, with margin. If a change
        to rectification or hashing narrows that gap, it fails here instead of
        silently dropping pages from a real scan.
        """
        cfg = DedupConfig()
        deduper = dedup.Deduplicator(cfg)

        text_a = "Primera pagina con su propio contenido impreso. " * 14
        text_b = "Segunda pagina, un texto distinto del anterior. " * 14

        def rectified(text: str, seed: int, tilt: float, blur: float) -> np.ndarray:
            frame, _ = synth.make_scene(
                text, synth.SceneConfig(tilt_deg=tilt, blur_sigma=blur), seed=seed
            )
            quad = detect.find_page(frame)
            assert quad is not None
            return detect.rectify(frame, quad)

        base = deduper.hash(rectified(text_a, 0, 6.0, 0.6))
        same = dedup.hamming(base, deduper.hash(rectified(text_a, 3, 6.8, 0.9)))
        different = dedup.hamming(base, deduper.hash(rectified(text_b, 0, 6.0, 0.6)))

        assert same <= cfg.max_hash_distance, f"same page read as different ({same} bits)"
        assert different > cfg.max_hash_distance, "different page read as same"
        assert different > same * 2, f"margin too thin: {same} vs {different} bits"

    def test_text_gate_catches_what_the_hash_misses(self):
        """The second gate exists because the first one is not sufficient."""
        deduper = dedup.Deduplicator(DedupConfig())
        page_text = "El ingenioso hidalgo don Quijote de la Mancha, primera parte, capitulo uno."
        deduper.accept("0" * 64, page_text)
        # Same page re-OCRed: a few characters differ, the content does not.
        assert deduper.is_duplicate_text(page_text.replace("capitulo uno", "capitulo vno"))
        assert not deduper.is_duplicate_text("Un texto completamente diferente del anterior aqui.")


class TestStability:
    def test_a_held_page_is_captured_exactly_once(self, clean_page):
        from scriptorium.vision.stability import StabilityTracker

        tracker = StabilityTracker(StabilityConfig(settle_frames=3))
        fires = [tracker.update(clean_page, page_found=True) for _ in range(12)]
        assert sum(fires) == 1, "a still page must be captured exactly once"

    def test_captures_again_after_a_page_turn(self, clean_page):
        """A real turn: the reader's hand crosses the frame, then a new page settles."""
        from scriptorium.vision.stability import StabilityTracker

        tracker = StabilityTracker(StabilityConfig(settle_frames=3))
        for _ in range(6):
            tracker.update(clean_page, page_found=True)

        hand = np.full_like(clean_page, 40)
        tracker.update(hand, page_found=False)

        next_page = cv2.rotate(clean_page, cv2.ROTATE_180)
        fires = [tracker.update(next_page, page_found=True) for _ in range(8)]
        assert sum(fires) == 1, "the next page must be captured after the turn"

    def test_rearms_when_the_page_content_changes(self, dense_page):
        """A swapped page with no hand sweep must still re-arm the tracker.

        This is why `turn_threshold` is low: a page-to-page change measures ~6 on
        the motion scale, so a higher bar would drop the page silently.
        """
        from scriptorium.vision.stability import StabilityTracker

        other = cv2.rotate(dense_page, cv2.ROTATE_180)
        tracker = StabilityTracker(StabilityConfig(settle_frames=3))

        fires = sum(tracker.update(dense_page, page_found=True) for _ in range(6))
        fires += sum(tracker.update(other, page_found=True) for _ in range(6))
        assert fires == 2

    def test_never_fires_without_a_page(self, clean_page):
        from scriptorium.vision.stability import StabilityTracker

        tracker = StabilityTracker(StabilityConfig(settle_frames=2))
        assert not any(tracker.update(clean_page, page_found=False) for _ in range(10))

    def test_motion_is_reported_for_the_ui(self, clean_page):
        from scriptorium.vision.stability import StabilityTracker

        tracker = StabilityTracker()
        tracker.update(clean_page, page_found=True)
        tracker.update(clean_page, page_found=True)
        assert tracker.last_motion < StabilityConfig().motion_threshold


def _baseline_bow(image: np.ndarray) -> float:
    """How far text lines sag in the middle of the page, in pixels.

    Locates each text line twice -- once near the left edge, once in the central
    band -- and returns the mean vertical disagreement. A straight line sits at
    the same height in both, a bowed one does not, so this reads out the physical
    quantity dewarping is supposed to remove.

    Deliberately *not* `dewarp.line_separation`: the stage now uses that metric
    internally to accept or reject its own correction, so asserting on it would
    only confirm that the gate did what the gate does. This measures the geometry
    instead, and from a different part of the image.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    width = binary.shape[1]
    edge = _line_centres(binary[:, : width // 4])
    centre = _line_centres(binary[:, 3 * width // 8 : 5 * width // 8])

    count = min(len(edge), len(centre))
    if count < 3:
        return 0.0
    return float(np.mean(np.abs(np.array(centre[:count]) - np.array(edge[:count]))))


def _line_centres(band: np.ndarray) -> list[float]:
    """Vertical centre of each contiguous run of ink in a vertical band."""
    profile = band.sum(axis=1).astype(np.float32)
    if profile.max() <= 0:
        return []
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
    return centres
