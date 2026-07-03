import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).parents[1] / "examples" / "zip_folder" / "verifier.py"
_SPEC = importlib.util.spec_from_file_location("zip_folder_test_verifier", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ZipFolderVerifier = _MODULE.ZipFolderVerifier


class CleanupDriver:
    def __init__(self):
        self.calls = 0

    async def cleanup_owned(self):
        self.calls += 1
        return []


@pytest.mark.asyncio
async def test_zip_reset_restores_clean_fixture(tmp_path):
    desktop = tmp_path / "Desktop"
    folder = desktop / "test_folder"
    folder.mkdir(parents=True)
    (folder / "text.txt").write_text("old", encoding="utf-8")
    (desktop / "test_folder.zip").write_bytes(b"old")
    driver = CleanupDriver()

    await ZipFolderVerifier(desktop).reset(driver)

    assert driver.calls == 1
    assert not folder.exists()
    assert not (desktop / "test_folder.zip").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("content,passed", [(b"nonzero", True), (b"", False)])
async def test_zip_verify_checks_nonzero_size_only(tmp_path, content, passed):
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    (desktop / "test_folder.zip").write_bytes(content)

    result = await ZipFolderVerifier(desktop).verify(CleanupDriver())

    assert result.passed is passed
