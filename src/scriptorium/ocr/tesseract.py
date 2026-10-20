"""Tesseract backend.

We ask for TSV (`image_to_data`) rather than plain text because the word boxes are
what make the exported PDF *searchable*: the invisible text layer has to sit on
top of the matching pixels, and the reading-aid mode needs a word box to
highlight while it speaks.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from ..types import OcrLine, OcrPage, OcrWord
from .base import OcrUnavailable

# Where the Windows installers put the binary. It is not added to PATH for
# already-running shells, so a user who just installed Tesseract and comes back
# to an open terminal gets "not installed" from a machine that has it.
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def find_binary() -> str | None:
    """Locate the Tesseract executable: explicit override, PATH, then install dirs."""
    override = os.environ.get("SCR_TESSERACT_CMD")
    if override and Path(override).exists():
        return override
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    return next((c for c in _WINDOWS_CANDIDATES if Path(c).exists()), None)


def find_tessdata() -> str | None:
    """Locate a language-data directory: `SCR_TESSDATA_DIR`, then `./tessdata`.

    The convention exists because of how the engine fails without it. Language
    models are a separate download from the binary -- the Windows installer ships
    English only -- and a missing model does not raise anything useful: the
    registry's `auto` mode falls through to the next engine and ends up on the
    stub. The run then *succeeds*, producing placeholder text, which is a far
    worse outcome than an error.

    Only the working directory is searched, deliberately. Resolving a path
    relative to this file would work from a source checkout and point at
    something arbitrary once the package is installed in site-packages, so the
    same code would behave differently depending on how it was installed. "The
    directory you run from" is a rule that holds either way -- and one a test can
    actually pin down.

    A directory with no `.traineddata` in it is not language data and must not
    shadow the system-wide install. Returning None is the normal case: it means
    the system directory already has what is needed.
    """
    override = os.environ.get("SCR_TESSDATA_DIR")
    if override and Path(override).is_dir():
        return override

    local = Path.cwd() / "tessdata"
    if local.is_dir() and any(local.glob("*.traineddata")):
        return str(local)
    return None


class TesseractEngine:
    name = "tesseract"

    def __init__(
        self, lang: str = "spa+eng", psm: int = 4, tessdata_dir: str | None = None
    ) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OcrUnavailable(
                "pytesseract is not installed (pip install 'scriptorium[tesseract]')"
            ) from exc

        binary = find_binary()
        if binary:
            pytesseract.pytesseract.tesseract_cmd = binary

        # The python binding and the engine are two separate installs, and having
        # the first without the second is the normal state of a fresh machine.
        # Checking only the import makes `doctor` claim the engine is available
        # and `auto` select it, and the failure then surfaces mid-scan instead of
        # at startup -- so probe the binary here, where the fallback still works.
        try:
            self.version = str(pytesseract.get_tesseract_version())
        except Exception as exc:  # pragma: no cover - environment dependent
            raise OcrUnavailable(
                f"the Tesseract binary was not found ({type(exc).__name__}). Install it:\n"
                "  Debian/Ubuntu : sudo apt install tesseract-ocr tesseract-ocr-spa\n"
                "  macOS         : brew install tesseract tesseract-lang\n"
                "  Windows       : winget install UB-Mannheim.TesseractOCR\n"
                "Already installed? Point SCR_TESSERACT_CMD at the executable."
            ) from exc

        # Language models are a separate download again -- the Windows installer
        # ships English only by default. Requesting a missing one fails the whole
        # call, so check up front and say which is missing.
        self.tessdata_dir = tessdata_dir or find_tessdata()
        if self.tessdata_dir:
            # Via the environment, not the `--tessdata-dir` flag. pytesseract
            # splits its config string with `shlex.split(posix=not_windows)`,
            # and with posix=False the quotes survive into the argument, so
            # Tesseract receives a path that still has quote characters in it and
            # cannot open the file. Leaving the path unquoted instead would break
            # on any directory containing a space. TESSDATA_PREFIX sidesteps the
            # quoting question entirely and is Tesseract's documented mechanism.
            os.environ["TESSDATA_PREFIX"] = str(Path(self.tessdata_dir).resolve())

        available = self.available_languages()
        missing = [code for code in lang.split("+") if code and code not in available]
        if missing:
            raise OcrUnavailable(
                "Tesseract has no language data for {} (installed: {}).\n"
                "Download the .traineddata files from "
                "https://github.com/tesseract-ocr/tessdata_fast into a folder and "
                "point SCR_TESSDATA_DIR at it.".format(
                    ", ".join(missing), ", ".join(sorted(available)) or "none"
                )
            )

        self.lang = lang
        # psm 4 = "a single column of text of variable sizes", the best fit for a
        # book page once it has been rectified and dewarped.
        self.psm = psm

    def available_languages(self) -> set[str]:
        """Language codes this installation can actually load."""
        import pytesseract

        try:
            if self.tessdata_dir:
                return {
                    path.stem
                    for path in Path(self.tessdata_dir).glob("*.traineddata")
                }
            return set(pytesseract.get_languages(config=""))
        except Exception:  # pragma: no cover - environment dependent
            return set()

    def _config(self) -> str:
        return f"--psm {self.psm}"

    def run(self, image: np.ndarray) -> OcrPage:
        import pytesseract
        from pytesseract import Output

        height, width = image.shape[:2]
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self._config(),
                output_type=Output.DICT,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise OcrUnavailable(f"tesseract binary not usable: {exc}") from exc

        grouped: dict[tuple[int, int, int], list[OcrWord]] = {}
        for i in range(len(data["text"])):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            confidence = float(data["conf"][i])
            if confidence < 0:  # -1 marks layout rows, not words
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            grouped.setdefault(key, []).append(
                OcrWord(
                    text=text,
                    bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                    confidence=confidence / 100.0,
                )
            )

        lines = []
        for key in sorted(grouped):
            words = sorted(grouped[key], key=lambda w: w.bbox[0])
            lines.append(OcrLine(words=words))
        return OcrPage(lines=lines, width=width, height=height, engine=self.name)
