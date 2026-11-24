"""Carregador do NIH ChestX-ray14 — substitui o stub do legado."""

from __future__ import annotations

import csv

import pytest

from radiologyai.data.nih_cxr14 import LIMITATIONS, _parse_age, build_manifest
from radiologyai.errors import EvaluationError
from radiologyai.modalities.xr.labels import NIH_CXR14_LABELS

FIELDS = [
    "Image Index",
    "Finding Labels",
    "Follow-up #",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "View Position",
]

RECORDS = [
    ("00000001_000.png", "Cardiomegaly", "1", "57", "M", "PA"),
    ("00000001_001.png", "Cardiomegaly|Emphysema", "1", "58", "M", "PA"),
    ("00000002_000.png", "No Finding", "2", "80", "F", "AP"),
    ("00000003_000.png", "Effusion|Infiltration", "3", "45", "F", "PA"),
    ("00000004_000.png", "No Finding", "4", "412", "M", "AP"),  # idade implausível
]


@pytest.fixture
def nih_root(tmp_path):
    with (tmp_path / "Data_Entry_2017_v2020.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for img, findings, pid, age, sex, view in RECORDS:
            w.writerow([img, findings, "0", pid, age, sex, view])

    (tmp_path / "test_list.txt").write_text(
        "00000003_000.png\n00000004_000.png\n", encoding="utf-8"
    )
    (tmp_path / "train_val_list.txt").write_text(
        "00000001_000.png\n00000001_001.png\n00000002_000.png\n", encoding="utf-8"
    )
    return tmp_path


class TestAgeParsing:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("058Y", 58.0), ("58", 58.0), ("018M", 1.5), ("", None), ("abc", None)],
    )
    def test_both_historical_formats(self, value, expected):
        assert _parse_age(value) == expected

    def test_implausible_age_discarded(self):
        """A fonte do NIH contém idades acima de 400 por erro de digitação."""
        assert _parse_age("412") is None
        assert _parse_age("0") is None


class TestBuildManifest:
    def test_test_split_uses_official_list(self, nih_root):
        m = build_manifest(nih_root, split="test")
        assert len(m) == 2
        assert {r.image_id for r in m} == {"00000003_000.png", "00000004_000.png"}

    def test_train_val_split(self, nih_root):
        assert len(build_manifest(nih_root, split="train_val")) == 3

    def test_all_split(self, nih_root):
        assert len(build_manifest(nih_root, split="all")) == len(RECORDS)

    def test_splits_are_patient_disjoint(self, nih_root):
        """O split oficial do NIH já é disjunto — é por isso que o usamos."""
        from radiologyai.evaluation import assert_patient_disjoint

        test = build_manifest(nih_root, split="test")
        tv = build_manifest(nih_root, split="train_val")
        assert_patient_disjoint({"test": test.patient_ids, "train_val": tv.patient_ids})

    def test_multi_label_parsed(self, nih_root):
        m = build_manifest(nih_root, split="all")
        row = next(r for r in m if r.image_id == "00000001_001.png")
        assert row.labels["Cardiomegaly"] == 1
        assert row.labels["Emphysema"] == 1
        assert row.labels["Pneumonia"] == 0

    def test_no_finding_is_all_zeros(self, nih_root):
        m = build_manifest(nih_root, split="all")
        row = next(r for r in m if r.image_id == "00000002_000.png")
        assert sum(row.labels.values()) == 0

    def test_all_fourteen_labels_present(self, nih_root):
        assert build_manifest(nih_root, split="all").label_names == NIH_CXR14_LABELS

    def test_demographics_extracted(self, nih_root):
        m = build_manifest(nih_root, split="all")
        row = next(r for r in m if r.image_id == "00000001_000.png")
        assert (row.patient_sex, row.view_position, row.patient_age_years) == ("M", "PA", 57.0)

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(EvaluationError, match="Data_Entry"):
            build_manifest(tmp_path, split="test")

    def test_missing_split_file_raises(self, tmp_path):
        (tmp_path / "Data_Entry_2017_v2020.csv").write_text("Image Index\n", encoding="utf-8")
        with pytest.raises(EvaluationError, match="test_list"):
            build_manifest(tmp_path, split="test")

    def test_unknown_split_raises(self, nih_root):
        with pytest.raises(EvaluationError, match="split desconhecido"):
            build_manifest(nih_root, split="inexistente")


def test_limitations_declare_nlp_labels():
    """A proveniência do rótulo precisa acompanhar todo resultado."""
    assert any("NLP" in limitation for limitation in LIMITATIONS)
    assert any("NÃO uma validação clínica" in limitation for limitation in LIMITATIONS)
