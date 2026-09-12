"""Artefatos de avaliação como única fonte de limiares."""

from __future__ import annotations

import json

import pytest

from radiologyai.errors import EvaluationError
from radiologyai.evaluation.artifacts import (
    evaluated_labels,
    find_run,
    latest_run_for_card,
    load_metrics,
    operating_points,
)


def write_run(root, run_id, per_label):
    d = root / run_id
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps({"run_id": run_id, "per_label": per_label}), encoding="utf-8"
    )
    return d


class TestOperatingPoints:
    def test_extracts_measured_thresholds(self):
        m = {
            "per_label": {
                "Effusion": {"operating_point": {"threshold": 0.31}},
                "Edema": {"operating_point": {"threshold": 0.12}},
            }
        }
        assert operating_points(m) == {"Effusion": 0.31, "Edema": 0.12}

    def test_label_without_operating_point_is_excluded(self):
        m = {
            "per_label": {
                "Hernia": {"auroc": 0.8},
                "Effusion": {"operating_point": {"threshold": 0.3}},
            }
        }
        assert operating_points(m) == {"Effusion": 0.3}

    def test_operating_point_error_is_excluded(self):
        m = {"per_label": {"X": {"operating_point": {"error": "nenhum threshold atinge"}}}}
        assert operating_points(m) == {}

    def test_empty_metrics_yield_nothing(self):
        assert operating_points({}) == {}
        assert evaluated_labels({}) == frozenset()


class TestRunLookup:
    def test_find_existing_run(self, tmp_path):
        write_run(tmp_path, "r1", {})
        assert find_run(tmp_path, "r1").name == "r1"

    def test_missing_run_raises_never_substitutes(self, tmp_path):
        with pytest.raises(EvaluationError, match="não encontrado"):
            find_run(tmp_path, "inexistente")

    def test_load_metrics(self, tmp_path):
        write_run(tmp_path, "r1", {"A": {}})
        assert load_metrics(tmp_path, "r1")["run_id"] == "r1"

    def test_latest_run_picks_newest_existing(self, tmp_path):
        write_run(tmp_path, "card__20260101T000000Z", {})
        write_run(tmp_path, "card__20260912T202549Z", {})
        chosen = latest_run_for_card(
            tmp_path, ("card__20260101T000000Z", "card__20260912T202549Z", "ausente")
        )
        assert chosen.name == "card__20260912T202549Z"

    def test_none_when_no_referenced_run_exists(self, tmp_path):
        assert latest_run_for_card(tmp_path, ("ausente",)) is None
        assert latest_run_for_card(tmp_path, ()) is None
