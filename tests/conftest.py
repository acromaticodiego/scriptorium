"""Shared fixtures.

The synthetic generator doubles as the test fixture factory: the same code that
produces evaluation data produces test data, so a change that breaks the
simulation breaks the tests too instead of quietly invalidating the metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval import synth

SAMPLE_TEXT = (
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho "
    "tiempo que vivia un hidalgo de los de lanza en astillero, adarga antigua, "
    "rocin flaco y galgo corredor. Una olla de algo mas vaca que carnero, "
    "salpicon las mas noches, duelos y quebrantos los sabados, lentejas los "
    "viernes, algun palomino de anadidura los domingos."
)


@pytest.fixture(scope="session")
def clean_page() -> np.ndarray:
    page, _ = synth.render_page(SAMPLE_TEXT)
    return page


@pytest.fixture(scope="session")
def dense_page() -> np.ndarray:
    """A fully typeset page.

    Several stages need enough text lines to measure anything: the dewarp fit
    needs baselines that span the page, and the motion metric needs ink spread
    over the frame. A sparse page is a different, and much weaker, test input.
    """
    page, _ = synth.render_page(DENSE_TEXT)
    return page


@pytest.fixture(scope="session")
def curled_page(dense_page: np.ndarray) -> np.ndarray:
    return synth.apply_curl(dense_page, amplitude=28.0)


DENSE_TEXT = (
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho "
    "tiempo que vivia un hidalgo de los de lanza en astillero. "
) * 12


@pytest.fixture(scope="session")
def scene() -> tuple[np.ndarray, str]:
    """A full simulated camera frame: curled page, tilted, on a desk."""
    return synth.make_scene(SAMPLE_TEXT, synth.SceneConfig())


@pytest.fixture(scope="session")
def spread_page(clean_page: np.ndarray) -> np.ndarray:
    """Two pages side by side with a darkened gutter between them."""
    left = clean_page.copy()
    right = clean_page.copy()
    gutter = np.full((clean_page.shape[0], 60, 3), 248, dtype=np.uint8)
    spread = np.hstack([left, gutter, right])
    # Darken a band at the gutter, as the paper curving away from the light does.
    start = clean_page.shape[1]
    band = spread[:, start : start + 60].astype(np.float32) * 0.45
    spread[:, start : start + 60] = band.astype(np.uint8)
    return spread
