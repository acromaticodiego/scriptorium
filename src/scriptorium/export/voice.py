"""Offline speech output for the headless reading aid.

The browser client speaks through the Web Speech API. This is the other half: a
capture station with no browser -- a Raspberry Pi with a camera arm and a
speaker, which is what an actual reading aid looks like -- speaking through the
operating system's own voices via pyttsx3.

Both consume `speech.ReadingScript`, so the text spoken and the word order are
identical in either mode. The difference is only what happens to the highlight:
here there are no pixels, so the current word is printed instead, which is also
what makes the sync verifiable from a terminal.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .speech import ReadingScript, SpokenWord

log = logging.getLogger(__name__)

OnWord = Callable[[SpokenWord], None]


class VoiceUnavailable(RuntimeError):
    """No usable system speech engine."""


@dataclass
class Narrator:
    """Speaks a reading script through the local system voice."""

    rate: float = 1.0
    lang_hint: str = "es"

    def __post_init__(self) -> None:
        try:
            import pyttsx3
        except ImportError as exc:  # pragma: no cover - optional extra
            raise VoiceUnavailable(
                "pyttsx3 is not installed (pip install 'scriptorium[tts]')"
            ) from exc
        try:
            self._engine = pyttsx3.init()
        except Exception as exc:  # pragma: no cover - platform dependent
            raise VoiceUnavailable(f"no system speech engine available: {exc}") from exc

        self._apply_rate()
        self._prefer_language()

    def speak(self, script: ReadingScript, on_word: OnWord | None = None) -> None:
        """Read a page aloud, reporting each word as it is spoken.

        pyttsx3 reports progress by (name, location, length) where `location` is
        a character offset into the utterance -- the same contract the browser's
        boundary event uses, which is why one `ReadingScript` serves both.
        """
        if not script.text.strip():
            return

        if on_word is not None:
            def relay(name: str, location: int, length: int) -> None:
                word = script.word_at_char(location)
                if word is not None:
                    on_word(word)

            token = self._engine.connect("started-word", relay)
        else:
            token = None

        try:
            self._engine.say(script.text)
            self._engine.runAndWait()
        finally:
            if token is not None:
                try:
                    self._engine.disconnect(token)
                except Exception:  # pragma: no cover - driver dependent
                    log.debug("could not disconnect word callback", exc_info=True)

    def _apply_rate(self) -> None:
        # pyttsx3 rate is words per minute, not a multiplier; ~200 is the usual
        # default, so scale from there to keep `--rate` meaning the same thing
        # as it does in the browser client.
        try:
            base = self._engine.getProperty("rate") or 200
            self._engine.setProperty("rate", int(base * self.rate))
        except Exception:  # pragma: no cover - driver dependent
            log.debug("could not set speech rate", exc_info=True)

    def _prefer_language(self) -> None:
        """Pick a voice matching the hint; an English voice reading Spanish is useless."""
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception:  # pragma: no cover - driver dependent
            return
        hint = self.lang_hint.lower()
        for voice in voices:
            languages = [
                item.decode(errors="ignore") if isinstance(item, bytes) else str(item)
                for item in (getattr(voice, "languages", None) or [])
            ]
            haystack = " ".join([*languages, str(getattr(voice, "id", "")),
                                 str(getattr(voice, "name", ""))]).lower()
            if hint in haystack:
                self._engine.setProperty("voice", voice.id)
                return
        log.info("no voice matching %r; using the system default", self.lang_hint)
