"""Orquestrador da linha de base — o que o notebook do Colab chama.

Roda sobre um dataset sintético em formato NIH, nos dois layouts distribuídos
(oficial: Data_Entry_2017_v2020.csv; Kaggle: Data_Entry_2017.csv).
"""

from __future__ import annotations

import csv
import importlib.util
import json

import numpy as np
import pytest
from PIL import Image

from radiologyai.errors import EvaluationError, ModelNotFoundError
from radiologyai.evaluation.baseline import BaselineResult, prepare_manifest, run_baseline

HAS_ML = all(importlib.util.find_spec(m) is not None for m in ("torch", "torchxrayvision"))

FIELDS = [
    "Image Index",
    "Finding Labels",
    "Follow-up #",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "View Position",
]


def make_nih_dataset(root, *, csv_name="Data_Entry_2017_v2020.csv", n=40, with_images=True):
    """Dataset NIH sintético: n imagens, metade Effusion, um quinto Cardiomegaly."""
    img_dir = root / "images_001" / "images"
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, ids = [], []
    for i in range(n):
        img = f"{i // 2:08d}_{i % 2:03d}.png"
        ids.append(img)
        findings = "Effusion" if i % 2 == 0 else ("Cardiomegaly" if i % 5 == 0 else "No Finding")
        rows.append(
            [
                img,
                findings,
                "0",
                str(i // 2),
                f"{30 + i % 40:03d}Y",
                "M" if i % 2 else "F",
                "PA" if i % 3 else "AP",
            ]
        )
        if with_images:
            Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(img_dir / img)
    with (root / csv_name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(rows)
    (root / "test_list.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return root


class TestPrepareManifest:
    def test_official_csv_name(self, tmp_path):
        make_nih_dataset(tmp_path)
        m = prepare_manifest(tmp_path, tmp_path / "out" / "m.csv")
        assert len(m) == 40
        assert (tmp_path / "out" / "m.csv").is_file()

    def test_kaggle_csv_name(self, tmp_path):
        """O espelho do Kaggle distribui Data_Entry_2017.csv, sem _v2020."""
        make_nih_dataset(tmp_path, csv_name="Data_Entry_2017.csv")
        assert len(prepare_manifest(tmp_path, tmp_path / "m.csv")) == 40

    def test_official_name_preferred_when_both_exist(self, tmp_path):
        make_nih_dataset(tmp_path)
        (tmp_path / "Data_Entry_2017.csv").write_text(",".join(FIELDS) + "\n", encoding="utf-8")
        assert len(prepare_manifest(tmp_path, tmp_path / "m.csv")) == 40

    def test_limit_truncates(self, tmp_path):
        make_nih_dataset(tmp_path)
        assert len(prepare_manifest(tmp_path, tmp_path / "m.csv", limit=7)) == 7

    def test_missing_csv_names_both_candidates(self, tmp_path):
        with pytest.raises(EvaluationError, match="Data_Entry_2017.csv"):
            prepare_manifest(tmp_path, tmp_path / "m.csv")


class TestRunBaselineFailsLoud:
    """Cada falha vira exceção — é o que permite ao notebook parar na célula certa."""

    def test_no_images_raises_before_loading_model(self, tmp_path):
        make_nih_dataset(tmp_path, with_images=False)
        (tmp_path / "images_001").rename(tmp_path / "sem_imagens")
        with pytest.raises(EvaluationError, match="nenhum diretório de imagens"):
            run_baseline(tmp_path, tmp_path / "art", log=lambda _: None)

    def test_missing_first_image_raises_before_loading_model(self, tmp_path):
        make_nih_dataset(tmp_path)
        (tmp_path / "images_001" / "images" / "00000000_000.png").unlink()
        with pytest.raises(EvaluationError, match="não encontrada"):
            run_baseline(tmp_path, tmp_path / "art", log=lambda _: None)

    def test_unknown_card_raises(self, tmp_path):
        make_nih_dataset(tmp_path)
        with pytest.raises(ModelNotFoundError):
            run_baseline(tmp_path, tmp_path / "art", card_id="inexistente", log=lambda _: None)


@pytest.mark.skipif(not HAS_ML, reason="requer o extra [ml]")
class TestRunBaselineEndToEnd:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory) -> tuple[BaselineResult, list[str]]:
        """Executa uma vez por classe; devolve (resultado, logs)."""
        root = make_nih_dataset(tmp_path_factory.mktemp("nih"), csv_name="Data_Entry_2017.csv")
        logs: list[str] = []
        res = run_baseline(root, root / "art", n_bootstrap=50, batch_size=8, log=logs.append)
        return res, logs

    @pytest.fixture
    def result(self, run) -> BaselineResult:
        return run[0]

    @pytest.fixture
    def logs(self, run) -> list[str]:
        return run[1]

    def test_writes_artifact(self, result):
        assert (result.run_dir / "metrics.json").is_file()
        assert (result.run_dir / "manifest_used.csv").is_file()

    def test_manifest_written_where_requested(self, result):
        assert result.manifest_path.is_file()

    def test_counts(self, result):
        assert (result.n_images, result.n_patients) == (40, 20)

    def test_padchest_on_nih_is_external(self, result):
        assert result.leakage_status == "externo"
        assert result.metrics["dataset"]["external_to_training_data"] is True

    def test_provenance_recorded(self, result):
        m = result.metrics
        assert len(m["model"]["weights_sha256"]) == 64
        assert len(m["dataset"]["manifest_sha256"]) == 64
        assert m["config"]["seed"] == 20260101

    def test_untrained_heads_declared(self, result):
        reasons = {x["label"]: x["reason"] for x in result.metrics["not_evaluated"]}
        assert any(k.startswith("__untrained_") for k in reasons)

    def test_logs_report_progress_and_leakage(self, logs):
        joined = "\n".join(logs)
        assert "vazamento: externo" in joined
        assert "inferência 40/40 (100%)" in joined

    def test_limit_is_flagged_as_partial(self, tmp_path):
        root = make_nih_dataset(tmp_path)
        logs: list[str] = []
        run_baseline(root, root / "art", limit=12, n_bootstrap=20, batch_size=4, log=logs.append)
        assert any("NÃO é a medição completa" in line for line in logs)

    def test_metrics_json_is_valid(self, result):
        json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
