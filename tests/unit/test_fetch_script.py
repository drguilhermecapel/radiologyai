"""Script de download do NIH — filtro de split e verificação de integridade.

O download real é de 42 GB, então estes testes exercitam a lógica sobre
tarballs sintéticos construídos em memória.
"""

from __future__ import annotations

import importlib.util
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_nih_cxr14.py"


@pytest.fixture(scope="module")
def fetch():
    spec = importlib.util.spec_from_file_location("fetch_nih", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fetch_nih"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tarball(tmp_path):
    """Tarball com 6 imagens; as 3 primeiras estão no split de teste."""
    names = [f"{i:08d}_000.png" for i in range(6)]
    source = tmp_path / "source"
    source.mkdir()
    for name in names:
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(source / name)

    archive = tmp_path / "images_001.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in names:
            tar.add(source / name, arcname=f"images/{name}")

    (tmp_path / "test_list.txt").write_text("\n".join(names[:3]) + "\n", encoding="utf-8")
    return archive, names


class TestTestListFiltering:
    def test_loads_official_split(self, fetch, tarball, tmp_path):
        _, names = tarball
        assert fetch.load_test_list(tmp_path) == set(names[:3])

    def test_absent_list_returns_none(self, fetch, tmp_path):
        assert fetch.load_test_list(tmp_path / "vazio") is None

    def test_extracts_only_split_members(self, fetch, tarball, tmp_path):
        """--test-only economiza ~30 GB extraindo só as 25.596 do teste."""
        archive, names = tarball
        keep = fetch.load_test_list(tmp_path)
        n = fetch.extract(archive, tmp_path / "out", keep)

        extracted = sorted(p.name for p in (tmp_path / "out").rglob("*.png"))
        assert n == 3
        assert extracted == sorted(names[:3])

    def test_extracts_everything_without_filter(self, fetch, tarball, tmp_path):
        archive, names = tarball
        n = fetch.extract(archive, tmp_path / "todos", None)
        assert n == len(names)

    def test_archive_without_split_members_yields_zero(self, fetch, tarball, tmp_path):
        archive, _ = tarball
        assert fetch.extract(archive, tmp_path / "out2", {"inexistente.png"}) == 0

    def test_already_extracted_is_skipped(self, fetch, tarball, tmp_path):
        archive, _ = tarball
        target = tmp_path / "out3"
        first = fetch.extract(archive, target, None)
        second = fetch.extract(archive, target, None)
        assert first == second


class TestMetadataIntegrity:
    def test_correct_counts_pass(self, fetch, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "EXPECTED_COUNTS", {"test_list.txt": 3})
        (tmp_path / "test_list.txt").write_text("a\nb\nc\n", encoding="utf-8")
        assert fetch.verify_metadata(tmp_path) is True

    def test_truncated_download_detected(self, fetch, tmp_path, monkeypatch):
        """Download truncado é detectado antes da inferência, não depois de horas."""
        monkeypatch.setattr(fetch, "EXPECTED_COUNTS", {"test_list.txt": 25596})
        (tmp_path / "test_list.txt").write_text("a\nb\n", encoding="utf-8")
        assert fetch.verify_metadata(tmp_path) is False

    def test_missing_file_detected(self, fetch, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "EXPECTED_COUNTS", {"ausente.txt": 10})
        assert fetch.verify_metadata(tmp_path) is False

    def test_wrong_csv_header_detected(self, fetch, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "EXPECTED_COUNTS", {"Data_Entry_2017_v2020.csv": 1})
        (tmp_path / "Data_Entry_2017_v2020.csv").write_text(
            "coluna,errada\nx,y\n", encoding="utf-8"
        )
        assert fetch.verify_metadata(tmp_path) is False


class TestOfficialConstants:
    def test_twelve_image_archives(self, fetch):
        assert len(fetch.IMAGE_ARCHIVES) == 12

    def test_official_counts_match_published(self, fetch):
        """112.120 imagens = 86.524 train_val + 25.596 test."""
        c = fetch.EXPECTED_COUNTS
        assert c["Data_Entry_2017_v2020.csv"] == 112_120
        assert c["test_list.txt"] + c["train_val_list.txt"] == 112_120

    def test_metadata_uses_verified_mirror(self, fetch):
        """Os links do Box para os metadados retornam 404; o espelho foi verificado."""
        for _, url in fetch.METADATA:
            assert url.startswith("https://huggingface.co/")

    def test_image_archives_use_official_nih_source(self, fetch):
        for _, url in fetch.IMAGE_ARCHIVES:
            assert url.startswith("https://nihcc.box.com/")
