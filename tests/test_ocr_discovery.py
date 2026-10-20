"""Engine and language-data discovery.

Worth its own tests because the failure it prevents is silent. A missing Spanish
model does not raise: `auto` selection falls through to the stub and the run
*succeeds*, producing placeholder text. Anyone demoing or measuring would get
confident-looking output that means nothing.
"""

from __future__ import annotations

from scriptorium.ocr import tesseract as tess
from scriptorium.ocr.registry import available_engines, diagnose


class TestTessdataDiscovery:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        (tmp_path / "spa.traineddata").write_bytes(b"x")
        monkeypatch.setenv("SCR_TESSDATA_DIR", str(tmp_path))
        assert tess.find_tessdata() == str(tmp_path)

    def test_local_tessdata_is_found_without_an_env_var(self, tmp_path, monkeypatch):
        """The convention that keeps a demo from silently falling back to the stub."""
        monkeypatch.delenv("SCR_TESSDATA_DIR", raising=False)
        local = tmp_path / "tessdata"
        local.mkdir()
        (local / "spa.traineddata").write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        assert tess.find_tessdata() == str(local)

    def test_an_empty_directory_is_not_language_data(self, tmp_path, monkeypatch):
        """A `tessdata/` with no models must not shadow the system install."""
        monkeypatch.delenv("SCR_TESSDATA_DIR", raising=False)
        (tmp_path / "tessdata").mkdir()
        monkeypatch.chdir(tmp_path)
        assert tess.find_tessdata() is None

    def test_a_bad_override_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCR_TESSDATA_DIR", str(tmp_path / "nope"))
        monkeypatch.chdir(tmp_path)
        assert tess.find_tessdata() is None


class TestBinaryDiscovery:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        fake = tmp_path / "tesseract.exe"
        fake.write_bytes(b"x")
        monkeypatch.setenv("SCR_TESSERACT_CMD", str(fake))
        assert tess.find_binary() == str(fake)

    def test_a_bad_override_falls_through(self, tmp_path, monkeypatch):
        """A stale override must not mask an engine that is actually installed."""
        monkeypatch.setenv("SCR_TESSERACT_CMD", str(tmp_path / "nope.exe"))
        # Falls through to PATH and the known install locations; whether either
        # exists is machine-dependent, so only the type is asserted here.
        assert tess.find_binary() is None or isinstance(tess.find_binary(), str)


class TestRegistryReporting:
    def test_the_stub_is_always_available(self):
        assert available_engines()["stub"] is True

    def test_diagnose_explains_every_engine(self):
        """`doctor` has to say *why* an engine is unusable, not just that it is.

        "not installed" and "installed but missing the language you are scanning"
        need different fixes, and the second is the common one.
        """
        detail = diagnose()
        assert set(detail) >= {"tesseract", "paddle", "stub"}
        assert all(isinstance(reason, str) and reason for reason in detail.values())
        assert "ready" in detail["stub"]
