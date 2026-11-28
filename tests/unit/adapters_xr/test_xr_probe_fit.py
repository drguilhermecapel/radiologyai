"""Sondas lineares: ajuste determinístico, pontos de operação do treino e artefato íntegro."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from radiologyai.analysis.probes.artifact import (
    ProbeArtifact,
    ProbeArtifactError,
    load_probe,
    probe_logit,
    save_probe,
    training_data_sha256,
)

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None
needs_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn ausente (extra eval)")
needs_st = pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors ausente (extra foundation)")


def _toy(n: int = 240, d: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Dados de brinquedo para exercitar a receita; não medem desempenho de nada.
    rng = np.random.default_rng(7)
    y = (np.arange(n) % 3 == 0).astype(int)
    x = rng.normal(size=(n, d)) + y[:, None] * 0.8
    groups = np.arange(n) // 2  # dois exames por "paciente"
    return x, y, groups


def _artifact(d: int = 4, **meta: object) -> ProbeArtifact:
    return ProbeArtifact(
        label="tuberculosis",
        encoder_id="enc",
        embedding_dim=d,
        mean=np.zeros(d),
        scale=np.ones(d),
        coef=np.arange(d, dtype=np.float64),
        intercept=-1.0,
        training_data_sha256=training_data_sha256([("a" * 64, 1), ("b" * 64, 0)]),
        meta=dict(meta),
    )


@needs_sklearn
@pytest.mark.requirement("REQ-125")
def test_fit_is_deterministic() -> None:
    from radiologyai.analysis.probes.fit import fit_probe

    x, y, g = _toy()
    a = fit_probe(x, y, g, seed=11)
    b = fit_probe(x, y, g, seed=11)
    assert a.C == b.C
    np.testing.assert_array_equal(a.coef, b.coef)
    np.testing.assert_array_equal(a.oof_logit, b.oof_logit)
    assert a.intercept == b.intercept


@needs_sklearn
@pytest.mark.requirement("REQ-125")
def test_inner_folds_never_split_a_patient() -> None:
    from sklearn.model_selection import StratifiedGroupKFold

    x, y, g = _toy()
    for train, val in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(x, y, g):
        assert not set(g[train]) & set(g[val])


@needs_sklearn
@pytest.mark.requirement("REQ-125")
def test_operating_points_come_from_training_oof_and_meet_targets() -> None:
    from radiologyai.analysis.probes.fit import fit_probe, oof_operating_points

    x, y, g = _toy()
    fit = fit_probe(x, y, g, seed=11)
    ops = oof_operating_points(
        fit.oof_logit, y, sensitivity_target=0.9, specificity_target=0.9, source="teste"
    )
    by_role = {op["role"]: op for op in ops}
    assert set(by_role) == {"rule_out", "rule_in"}
    s = fit.oof_logit
    sens = float((s[y == 1] >= by_role["rule_out"]["threshold"]).mean())
    spec = float((s[y == 0] < by_role["rule_in"]["threshold"]).mean())
    assert sens >= 0.9
    assert spec >= 0.9
    assert all(op["on"] == "logit" and op["source"] for op in ops)


@needs_sklearn
def test_fit_refuses_single_class() -> None:
    from radiologyai.analysis.probes.fit import fit_probe
    from radiologyai.errors import EvaluationError

    x, _, g = _toy()
    with pytest.raises(EvaluationError):
        fit_probe(x, np.zeros(len(x), dtype=int), g)


def test_probe_logit_is_standardised_linear_function() -> None:
    art = _artifact()
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert probe_logit(art, x)[0] == pytest.approx(-1.0 + 0 * 1 + 1 * 2 + 2 * 3 + 3 * 4)
    with pytest.raises(ProbeArtifactError):
        probe_logit(art, np.ones((1, 3)))
    with pytest.raises(ProbeArtifactError):
        probe_logit(art, np.array([[np.nan, 0, 0, 0]]))


@pytest.mark.requirement("REQ-125")
def test_artifact_refuses_performance_metadata() -> None:
    with pytest.raises(ProbeArtifactError, match="métricas"):
        _artifact(auroc=0.9)


@needs_st
@pytest.mark.requirement("REQ-125")
def test_save_load_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    art = _artifact(C=0.01)
    tensors, meta = save_probe(art, tmp_path / "probe")
    loaded = load_probe(tensors.read_bytes(), meta.read_bytes())
    np.testing.assert_array_equal(loaded.coef, art.coef)
    assert loaded.intercept == art.intercept
    assert loaded.training_data_sha256 == art.training_data_sha256
    blob = bytearray(tensors.read_bytes())
    blob[-1] ^= 0xFF
    with pytest.raises(ProbeArtifactError, match="sha256"):
        load_probe(bytes(blob), meta.read_bytes())


def test_training_hash_is_order_independent() -> None:
    a = training_data_sha256([("a" * 64, 1), ("b" * 64, 0)])
    b = training_data_sha256([("b" * 64, 0), ("a" * 64, 1)])
    assert a == b
    assert a != training_data_sha256([("a" * 64, 0), ("b" * 64, 0)])
