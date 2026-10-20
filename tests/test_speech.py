"""Reading script tests.

The contract these pin down is narrow and unforgiving: a character offset the
speech engine reports must land on the word the engine is actually saying. If
the offsets drift, the highlight drifts, and a reader following the voice is
pointed at the wrong line -- a silent failure that no exception reports.
"""

from __future__ import annotations

import pytest

from scriptorium.export.speech import build_script
from scriptorium.types import OcrLine, OcrPage, OcrWord


def word(text: str, x: int = 0, y: int = 0, height: int = 40) -> OcrWord:
    return OcrWord(text=text, bbox=(x, y, max(1, len(text) * 18), height), confidence=0.9)


def page(*lines: list[OcrWord], width: int = 1080, height: int = 1600) -> OcrPage:
    return OcrPage(
        lines=[OcrLine(words=list(items)) for items in lines], width=width, height=height
    )


class TestOffsets:
    def test_every_offset_indexes_its_own_word(self):
        """The invariant the whole module exists to guarantee."""
        script = build_script(
            page(
                [word("En"), word("un"), word("lugar"), word("de")],
                [word("la"), word("Mancha,"), word("de"), word("cuyo")],
                [word("nombre"), word("no"), word("quiero")],
            )
        )
        assert script.words
        for spoken in script.words:
            assert script.text[spoken.char_start : spoken.char_end] == spoken.text

    def test_text_is_single_spaced(self):
        script = build_script(page([word("uno"), word("dos")], [word("tres")]))
        assert script.text == "uno dos tres"

    def test_lookup_finds_the_word_being_spoken(self):
        script = build_script(page([word("alfa"), word("beta"), word("gamma")]))
        assert script.word_at_char(0).text == "alfa"
        assert script.word_at_char(3).text == "alfa"
        assert script.word_at_char(5).text == "beta"
        assert script.word_at_char(10).text == "gamma"

    def test_offset_in_a_gap_holds_the_previous_word(self):
        """Stops the highlight blinking off during every inter-word pause."""
        script = build_script(page([word("alfa"), word("beta")]))
        gap = script.words[0].char_end  # the space itself
        assert script.word_at_char(gap).text == "alfa"

    def test_offset_past_the_end_holds_the_last_word(self):
        script = build_script(page([word("alfa"), word("beta")]))
        assert script.word_at_char(10_000).text == "beta"

    def test_negative_offset_is_not_a_word(self):
        script = build_script(page([word("alfa")]))
        assert script.word_at_char(-1) is None


class TestHyphenation:
    def test_word_split_across_lines_is_rejoined(self):
        """Reading "conti-" then "nuacion" is spelling, not reading."""
        script = build_script(
            page([word("En"), word("conti-")], [word("nuacion"), word("del"), word("texto")])
        )
        assert "continuacion" in script.text
        assert "conti-" not in script.text
        assert script.rejoined_words == 1

    def test_a_rejoined_word_keeps_both_boxes(self):
        """The reader has to see both halves of the word light up."""
        script = build_script(
            page([word("conti-", x=400, y=100)], [word("nuacion", x=60, y=160)])
        )
        joined = next(w for w in script.words if w.text == "continuacion")
        assert len(joined.boxes) == 2
        assert joined.boxes[0][1] != joined.boxes[1][1]  # different lines

    def test_a_compound_across_a_break_is_not_rejoined(self):
        """"Madrid-" then "Barcelona" is a compound, not a broken word."""
        script = build_script(page([word("Madrid-")], [word("Barcelona")]))
        assert script.rejoined_words == 0
        assert "Madrid-" in script.text

    def test_offsets_stay_correct_after_rejoining(self):
        """The regression this is really guarding: a join that shifts later offsets."""
        script = build_script(
            page(
                [word("primero"), word("hiper-")],
                [word("activo"), word("segundo")],
                [word("tercero")],
            )
        )
        for spoken in script.words:
            assert script.text[spoken.char_start : spoken.char_end] == spoken.text

    def test_page_ending_mid_word_is_flagged_not_silently_cut(self):
        """A word cut at the page edge continues on the *next* page.

        There is no next page here to join it to, so the fragment is spoken as
        it stands and the condition is reported. Stripping the hyphen would make
        the text claim a complete word that is not one, and the same text goes
        into the txt and PDF exports.
        """
        script = build_script(page([word("texto"), word("incom-")]))
        assert script.text == "texto incom-"
        assert script.ends_mid_word

    def test_a_normal_page_is_not_flagged(self):
        script = build_script(page([word("texto"), word("completo.")]))
        assert not script.ends_mid_word

    # The lookalike dashes are the point of the test, hence the noqa.
    @pytest.mark.parametrize("hyphen", ["-", "‐", "­"])  # noqa: RUF001
    def test_all_hyphen_lookalikes_are_handled(self, hyphen):
        """OCR returns whichever dash its model prefers; all of them must join."""
        script = build_script(page([word("conti" + hyphen)], [word("nuacion")]))
        assert script.text == "continuacion"


class TestGeometry:
    def test_page_dimensions_are_carried(self):
        script = build_script(page([word("x")], width=900, height=1400))
        assert (script.page_width, script.page_height) == (900, 1400)

    def test_empty_page_is_safe(self):
        script = build_script(page())
        assert script.text == ""
        assert script.words == []
        assert script.word_at_char(0) is None

    def test_blank_lines_are_skipped(self):
        script = build_script(page([], [word("solo")], []))
        assert script.text == "solo"

    def test_line_index_is_reported(self):
        script = build_script(page([word("a")], [word("b")]))
        assert [w.line_index for w in script.words] == [0, 1]

    def test_serialises_for_the_api(self):
        payload = build_script(page([word("hola")])).as_dict()
        assert payload["text"] == "hola"
        assert payload["words"][0]["boxes"][0][0] == 0
        assert "page_width" in payload
