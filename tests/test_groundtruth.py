"""Ground-truth draft workflow.

The point of these tests is one specific failure: measuring accuracy against a
transcription that is still the pipeline's own output. That comparison returns a
perfect score and is completely meaningless, and nothing about it looks wrong
from the outside -- which is exactly why it needs a test rather than a warning
in a docstring.
"""

from __future__ import annotations

import cv2
import pytest

from eval.run import load_real_samples
from scriptorium.export.text import (
    UNREVIEWED_MARKER,
    read_reviewed_truth,
    write_draft_truth,
)


class TestDrafts:
    def test_a_draft_is_marked_unreviewed(self, tmp_path):
        path = write_draft_truth("texto reconocido", tmp_path / "page.txt")
        assert UNREVIEWED_MARKER in path.read_text(encoding="utf-8")
        assert read_reviewed_truth(path) is None

    def test_deleting_the_marker_makes_it_reviewed(self, tmp_path):
        path = write_draft_truth("texto reconocido", tmp_path / "page.txt")
        path.write_text("texto corregido a mano", encoding="utf-8")
        assert read_reviewed_truth(path) == "texto corregido a mano"

    def test_an_existing_file_is_never_overwritten(self, tmp_path):
        """A corrected transcription is hand-made work; regenerating must not eat it."""
        path = tmp_path / "page.txt"
        path.write_text("transcripcion corregida a mano", encoding="utf-8")
        write_draft_truth("salida del ocr", path)
        assert path.read_text(encoding="utf-8") == "transcripcion corregida a mano"


class TestHarnessRefusesDrafts:
    @pytest.fixture()
    def folder(self, tmp_path, scene):
        frame, _ = scene
        cv2.imwrite(str(tmp_path / "page.jpg"), frame)
        return tmp_path

    def test_an_unreviewed_draft_is_skipped(self, folder, capsys):
        """The guard against a circular measurement."""
        write_draft_truth("salida del ocr sin corregir", folder / "page.txt")
        samples = load_real_samples(folder)
        assert samples == []
        assert "unreviewed" in capsys.readouterr().out.lower()

    def test_a_reviewed_transcription_is_loaded(self, folder):
        (folder / "page.txt").write_text("transcripcion revisada", encoding="utf-8")
        samples = load_real_samples(folder)
        assert len(samples) == 1
        assert samples[0].truth == "transcripcion revisada"
        assert samples[0].difficulty == "real"

    def test_a_photo_without_a_transcription_is_ignored(self, folder):
        assert load_real_samples(folder) == []

    def test_an_empty_transcription_is_ignored(self, folder):
        (folder / "page.txt").write_text("   \n", encoding="utf-8")
        assert load_real_samples(folder) == []
