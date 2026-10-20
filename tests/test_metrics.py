"""Metric tests.

A silently wrong metric is worse than no metric: every number in the README
depends on these three functions, so they are pinned against hand-computed
values.
"""

from __future__ import annotations

from eval.metrics import edit_distance, error_rates, normalise


def test_edit_distance_known_values():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


def test_perfect_transcription_scores_zero():
    rates = error_rates("hola mundo", "hola mundo")
    assert rates.cer == 0.0
    assert rates.wer == 0.0


def test_single_character_error():
    rates = error_rates("hola mundo", "hola munda")
    assert rates.cer == 0.1  # 1 of 10 characters
    assert rates.wer == 0.5  # 1 of 2 words


def test_whitespace_is_normalised_but_accents_are_not():
    assert error_rates("hola   mundo", "hola mundo").cer == 0.0
    assert error_rates("habia", "había").cer > 0.0


def test_accent_folding_is_opt_in():
    assert error_rates("habia", "había", strip_accents=True).cer == 0.0


def test_typographic_quotes_are_equivalent():
    assert error_rates('dijo "si"', "dijo “si”").cer == 0.0


def test_empty_hypothesis_is_total_failure():
    assert error_rates("texto de referencia", "").cer == 1.0


def test_rates_are_clamped_for_reporting():
    rates = error_rates("a", "x" * 50)
    assert rates.cer == 1.0


def test_normalise_collapses_newlines():
    assert normalise("linea\n\nuno") == "linea uno"
