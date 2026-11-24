"""Manifest — o registro de reprodutibilidade."""

from __future__ import annotations

import pytest

from radiologyai.data.manifest import Manifest, ManifestRow
from radiologyai.errors import EvaluationError

LABELS = ("Pneumonia", "Effusion")


def make_manifest(n: int = 6) -> Manifest:
    rows = [
        ManifestRow(
            image_id=f"img_{i:03d}.png",
            patient_id=str(i // 2),
            labels={"Pneumonia": i % 2, "Effusion": int(i > 3)},
            view_position="PA" if i % 2 else "AP",
            patient_sex="M" if i % 2 else "F",
            patient_age_years=30.0 + i * 10,
        )
        for i in range(n)
    ]
    return Manifest(name="Teste", split="test", label_names=LABELS, rows=rows)


class TestManifest:
    def test_len_and_patients(self):
        m = make_manifest(6)
        assert len(m) == 6
        assert len(m.patient_ids) == 3

    def test_label_matrix_follows_label_order(self):
        m = make_manifest(4)
        matrix = m.label_matrix()
        assert len(matrix) == 4
        assert all(len(row) == len(LABELS) for row in matrix)

    def test_positives_count(self):
        m = make_manifest(6)
        assert m.positives("Pneumonia") == 3
        assert m.positives("Effusion") == 2

    def test_age_bands(self):
        assert ManifestRow("a", "1", {}, patient_age_years=25).age_band() == "18-39"
        assert ManifestRow("a", "1", {}, patient_age_years=50).age_band() == "40-59"
        assert ManifestRow("a", "1", {}, patient_age_years=70).age_band() == "60-74"
        assert ManifestRow("a", "1", {}, patient_age_years=80).age_band() == "75+"
        assert ManifestRow("a", "1", {}).age_band() == "desconhecida"


class TestRoundTrip:
    def test_csv_round_trip_preserves_content(self, tmp_path):
        original = make_manifest(8)
        path = original.to_csv(tmp_path / "m.csv")
        loaded = Manifest.from_csv(path, name="Teste", split="test")

        assert len(loaded) == len(original)
        assert loaded.label_names == original.label_names
        assert loaded.label_matrix() == original.label_matrix()
        assert loaded.patient_ids == original.patient_ids

    def test_sha256_survives_round_trip(self, tmp_path):
        """O hash é o que amarra um artefato de avaliação ao dado exato usado."""
        original = make_manifest(8)
        loaded = Manifest.from_csv(original.to_csv(tmp_path / "m.csv"), name="Teste", split="test")
        assert loaded.sha256() == original.sha256()

    def test_sha256_changes_when_labels_change(self):
        a = make_manifest(6)
        b = make_manifest(6)
        b.rows[0] = ManifestRow("img_000.png", "0", {"Pneumonia": 1, "Effusion": 1})
        assert a.sha256() != b.sha256()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(EvaluationError, match="não encontrado"):
            Manifest.from_csv(tmp_path / "x.csv", name="X", split="test")


class TestOrderIndependentHash:
    """O CSV oficial e o do Kaggle têm as mesmas linhas em ordens diferentes."""

    def test_shuffled_rows_same_hash(self):
        import random

        a = make_manifest(20)
        b = make_manifest(20)
        random.Random(7).shuffle(b.rows)
        assert [r.image_id for r in a.rows] != [r.image_id for r in b.rows]
        assert a.sha256() == b.sha256()

    def test_committed_manifest_matches_colab_artifact(self):
        """Regressão: o artefato de 2026-09-12 registrou 39f31d78… para o mesmo conteúdo."""
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "datasets" / "manifests" / "nih_cxr14_test.csv"
        m = Manifest.from_csv(path, name="NIH ChestX-ray14", split="test")
        assert m.sha256() == "39f31d789c3ccc1cc8413800cc07511546fa680aad03745729ce07ffacf44bda"
