"""Reading script: the bridge between spoken text and the pixels it came from.

A read-aloud mode has to answer one question continuously: *the engine is
currently at character 417 of the utterance — which word on the page is that?*
Speech engines report progress as a character offset into the string they were
given, so the mapping only works if the string handed to the engine and the
string those offsets are resolved against are the **same string**.

That is why this lives on the server rather than in the browser client. If the
client rebuilt the utterance by joining words itself, any disagreement about
joining -- one space or two, how a hyphenated word is put back together, whether
a line break becomes a space -- would shift every offset after it, and the
highlight would drift away from the voice. One builder, one mapping, both
clients.

Carries no audio and no engine: a browser using `SpeechSynthesis`, a CLI using
pyttsx3, and a cloud voice all consume the same script.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import OcrPage

Box = tuple[int, int, int, int]

# Characters a line can end with when a word is broken across the break. All
# three look alike on the page and OCR picks whichever its model prefers, so all
# three have to be recognised: hyphen-minus, Unicode hyphen, soft hyphen.
_HYPHENS = "-‐­"  # noqa: RUF001 - the lookalikes are the point


@dataclass
class SpokenWord:
    """One word as the engine will say it, with where it sits on the page.

    `boxes` is a list because a word broken across a line break is a single
    spoken word in two places, and a reader following the highlight needs to see
    both halves light up.
    """

    text: str
    char_start: int
    char_end: int
    boxes: list[Box]
    line_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "boxes": [list(box) for box in self.boxes],
            "line_index": self.line_index,
        }


@dataclass
class ReadingScript:
    """What to speak, and what to highlight while speaking it."""

    text: str
    words: list[SpokenWord] = field(default_factory=list)
    page_width: int = 0
    page_height: int = 0
    rejoined_words: int = 0
    # True when the last word on the page is cut by a hyphen, i.e. it continues
    # on the *next* page. Joining across a page boundary needs state this
    # function does not have -- the next page may not even be captured yet -- so
    # the fragment is spoken as it stands, hyphen included, and the flag is
    # raised rather than the problem hidden. A caller that holds both pages can
    # stitch them; see the roadmap.
    ends_mid_word: bool = False
    _starts: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._starts = [word.char_start for word in self.words]

    def word_at_char(self, index: int) -> SpokenWord | None:
        """The word containing a character offset, or None before the first one.

        Binary search, because a boundary event arrives several times a second
        for the whole length of a page.

        An offset landing in the space *between* two words resolves to the word
        just spoken rather than to nothing, which is what stops the highlight
        from blinking off during every pause.
        """
        if not self.words or index < 0:
            return None
        position = bisect_right(self._starts, index) - 1
        return self.words[position] if position >= 0 else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "rejoined_words": self.rejoined_words,
            "ends_mid_word": self.ends_mid_word,
            "words": [word.as_dict() for word in self.words],
        }


def build_script(page: OcrPage) -> ReadingScript:
    """Turn a recognised page into a speakable script with page geometry."""
    units, rejoined = _reading_units(page)

    words: list[SpokenWord] = []
    cursor = 0
    for text, boxes, line_index in units:
        if words:
            cursor += 1  # the single space that joins the previous word to this
        words.append(
            SpokenWord(
                text=text,
                char_start=cursor,
                char_end=cursor + len(text),
                boxes=boxes,
                line_index=line_index,
            )
        )
        cursor += len(text)

    return ReadingScript(
        text=" ".join(word.text for word in words),
        words=words,
        page_width=page.width,
        page_height=page.height,
        rejoined_words=rejoined,
        ends_mid_word=bool(words) and words[-1].text[-1:] in _HYPHENS,
    )


def stitch_page_texts(texts: Sequence[str]) -> list[str]:
    """Repair words cut by a *page* break, across a sequence of page texts.

    `build_script` and `page_text` repair a word split across a line break, but a
    word cut at the bottom of a page continues on the next one, and neither
    function can see that far: when a page is recognised the next one may not
    have been captured yet. So the repair happens here, once the pages exist
    together.

    The completed word is placed on the page where it *starts*, and removed from
    the beginning of the next, which keeps the result usable per page -- a
    Markdown section, an EPUB document, one page read aloud -- instead of only as
    one continuous blob. Returns the same number of texts it was given.
    """
    repaired = list(texts)
    for index in range(len(repaired) - 1):
        current = repaired[index].rstrip()
        if not current or current[-1] not in _HYPHENS or len(current) < 2:
            continue

        following = repaired[index + 1].lstrip()
        match = re.match(r"(\w+)([\s\S]*)", following, flags=re.UNICODE)
        if match is None:
            continue
        head, rest = match.group(1), match.group(2)
        # Same guard as within a page: a lowercase continuation is a broken word,
        # an uppercase one is a new sentence or a compound that merely happened to
        # land on the boundary.
        if not head[:1].islower():
            continue

        repaired[index] = current[:-1] + head
        repaired[index + 1] = rest.lstrip()
    return repaired


def page_text(page: OcrPage) -> str:
    """The page as text, with words split across line breaks put back together.

    Differs from `build_script().text` in one way only: line breaks are kept, so
    a text or Markdown export still looks like a page, while the utterance for a
    speech engine is one continuous string.

    Both come from the same `_reading_units`, which is the point -- a second
    implementation of "how are words joined" is how the exported text and the
    spoken text drift apart. Measured on hyphenated pages, this repair removes
    98% of the remaining character error (CER 0.0082 -> 0.0002): the raw
    line-joined text claims words like "emen-dado" that are not words.
    """
    units, _ = _reading_units(page)
    if not units:
        return ""

    lines: list[list[str]] = []
    current_line = -1
    for text, _boxes, line_index in units:
        if line_index != current_line:
            lines.append([])
            current_line = line_index
        lines[-1].append(text)
    return "\n".join(" ".join(line) for line in lines)


def _reading_units(page: OcrPage) -> tuple[list[tuple[str, list[Box], int]], int]:
    """Flatten the page into spoken units, repairing words split by line breaks.

    Books break words across lines with a hyphen. Read literally that is not
    reading, it is spelling: "conti-" then "nuación". So a line-final hyphen
    whose continuation starts lowercase is stitched back onto the next fragment,
    and the result keeps both bounding boxes.
    """
    lines = [line for line in page.lines if line.words]
    units: list[tuple[str, list[Box], int]] = []
    rejoined = 0
    carried: tuple[str, list[Box], int] | None = None

    for line_index, line in enumerate(lines):
        words = list(line.words)

        for position, word in enumerate(words):
            text = word.text

            if carried is not None:
                stem, boxes, origin = carried
                units.append((stem + text, [*boxes, word.bbox], origin))
                rejoined += 1
                carried = None
                continue

            last_in_line = position == len(words) - 1
            if (
                last_in_line
                and len(text) > 1
                and text[-1] in _HYPHENS
                and _continues_lowercase(lines, line_index)
            ):
                carried = (text[:-1], [word.bbox], line_index)
                continue

            units.append((text, [word.bbox], line_index))

    if carried is not None:
        # The page ended mid-word: speak the fragment rather than drop it.
        stem, boxes, origin = carried
        units.append((stem, boxes, origin))

    return units, rejoined


def _continues_lowercase(lines: list, line_index: int) -> bool:
    """Does the next line begin with a lowercase continuation?

    The guard against over-eager joining: "Madrid-" at the end of a line
    followed by "Barcelona" is a compound spanning the break, not a broken word.
    """
    if line_index + 1 >= len(lines):
        return False
    first = lines[line_index + 1].words[0].text
    return bool(first) and first[0].islower()
