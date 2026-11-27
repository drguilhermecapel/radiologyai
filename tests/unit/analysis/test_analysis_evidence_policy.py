"""EvidencePolicy: achados A/B/C decidem só com os pontos do registro que sustenta o nível.

Registros sintéticos e pequenos, gravados em ``tmp_path``; nunca o cache real.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from analysis_support import WEIGHT_SHA, condition, manifest_dict, op, projection

from radiologyai.analysis.artifacts import ArtifactStore
from radiologyai.analysis.evidence_policy import EvidencePolicy, select_basis
from radiologyai.analysis.pipeline import analyze
from radiologyai.analysis.types import AnalysisResult, ExamInputs, Finding
from radiologyai.catalog.catalog import Catalog
from radiologyai.catalog.schema import (
    EvidenceLevel,
    EvidenceRecord,
    EvidenceSource,
    ExamType,
    LevelEntry,
    LevelsFile,
    MeasurementProvenance,
    visibility_for,
)
from radiologyai.evidence.measure import MeasuredEvidence
from radiologyai.evidence.store import LoadedEvidence, load_evidence
from radiologyai.reporting.content import _point

PROTOCOL = "2026-11-v1"
TB = ("cxr.tuberculosis", "xr-probe-tb", "tuberculosis")
RULE_OUT, RULE_IN = -1.0, 1.0
# Pontos "fora-da-dobra" do manifesto: NUNCA podem decidir um achado C.
MANIFEST_POINTS = [
    {"label": "tuberculosis", "point": op("rule_out", -3.0, on="logit")},
    {"label": "tuberculosis", "point": op("rule_in", -2.5, on="logit")},
]


def _record(
    dataset: str,
    *,
    kind: str = "measured_internal",
    leakage: str = "known",
    n_total: int = 200,
    protocol_id: str = PROTOCOL,
    weights: str = WEIGHT_SHA,
    triple: tuple[str, str, str] = TB,
) -> EvidenceRecord:
    cid, mid, label = triple
    return EvidenceRecord(
        id=f"measured:{dataset}:{mid}:{label}:discrimination",
        condition_id=cid,
        model_id=mid,
        label=label,
        kind=kind,  # type: ignore[arg-type]
        dataset=dataset,
        n_total=n_total,
        n_positive=n_total // 2,
        leakage=leakage,  # type: ignore[arg-type]
        passes_protocol=True,
        provenance=MeasurementProvenance(
            git_sha="0" * 40,
            protocol_id=protocol_id,
            dataset_id=dataset,
            dataset_sha256="a" * 64,
            weights_sha256={"main.bin": weights},
            command="evidence_run.py --job teste",
            created_at=datetime(2026, 11, 27, tzinfo=UTC),
        ),
    )


def _envelope(
    record: EvidenceRecord,
    *,
    report_scale: str = "logit",
    on: str = "logit",
    rule_out: float = RULE_OUT,
    rule_in: float = RULE_IN,
    report_rule_in: float | None = None,
) -> dict[str, Any]:
    ref = f"evidence/measured/{record.dataset}/{record.model_id}/{record.label}.json#report"
    points = [
        {**op("rule_out", rule_out, on=on), "name": "tb:rule_out", "evidence_ref": ref},
        {**op("rule_in", rule_in, on=on), "name": "tb:rule_in", "evidence_ref": ref},
    ]
    for p in points:
        p["source"] = f"protocolo {PROTOCOL}: ajustado na metade de calibração"
    report = {
        "protocol_id": record.provenance.protocol_id if record.provenance else None,
        "score_scale": report_scale,
        "calibration": {"temperature": 0.8},
        "rule_out": {"role": "rule_out", "threshold": rule_out, "score_scale": report_scale},
        "rule_in": {
            "role": "rule_in",
            "threshold": rule_in if report_rule_in is None else report_rule_in,
            "score_scale": report_scale,
        },
    }
    env = MeasuredEvidence(record=record, report=report, operating_points=points)  # type: ignore[arg-type]
    return env.model_dump(mode="json")


def _write(root: Path, record: EvidenceRecord, data: Any) -> Path:
    path = root / "measured" / str(record.dataset) / record.model_id / f"{record.label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _catalog(
    level: EvidenceLevel, basis: tuple[str, ...], triple: tuple[str, str, str] = TB
) -> Catalog:
    cid, mid, label = triple
    entry = LevelEntry(
        condition_id=cid,
        model_id=mid,
        label=label,
        level=level,
        visibility=visibility_for(level),
        basis=basis,
    )
    levels = LevelsFile(protocol_id=PROTOCOL, evidence_digest="0" * 64, entries=(entry,))
    return Catalog([condition(cid)], levels)


def _tb_manifest() -> dict[str, Any]:
    return manifest_dict(
        "xr-probe-tb",
        "LogitScore",
        labels=[
            {
                "key": "tuberculosis",
                "source_label": "tuberculosis",
                "condition_id": "cxr.tuberculosis",
                "score_kind": "logit",
                "code": {"scheme": "99MEDAI", "value": "cxr.tuberculosis", "meaning": "tb"},
            }
        ],
        operating_points=MANIFEST_POINTS,
    )


def _run(
    make_registry: Any,
    store: ArtifactStore,
    catalog: Catalog,
    root: Path,
    pixel_level: float = 220.0,
    manifest: dict[str, Any] | None = None,
) -> AnalysisResult:
    manifest = manifest or _tb_manifest()
    return analyze(
        ExamInputs(images=(projection(level=pixel_level),), exam_type=ExamType.CXR),
        registry=make_registry(manifest),
        catalog=catalog,
        store=store,
        policy=EvidencePolicy.load(catalog, root),
        model_ids=[manifest["id"]],
        include_research=True,
    )


def _finding(result: AnalysisResult) -> Finding:
    assert [r.status for r in result.runs] == ["completed"], result.runs
    (finding,) = result.findings
    return finding


# --------------------------------------------------------------------------- #
# Caminho feliz: cxr.tuberculosis / xr-probe-tb, nível C
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-192")
@pytest.mark.parametrize(
    ("pixel_level", "expected"),
    [(220.0, "positive"), (128.0, "indeterminate"), (40.0, "negative")],
)
def test_tb_level_c_decides_with_record_thresholds(
    make_registry: Any, store: ArtifactStore, tmp_path: Path, pixel_level: float, expected: str
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    path = _write(root, record, _envelope(record))
    catalog = _catalog(EvidenceLevel.C, (record.id,))

    finding = _finding(_run(make_registry, store, catalog, root, pixel_level))

    assert finding.evidence_level is EvidenceLevel.C
    assert finding.decision == expected
    rel = path.relative_to(tmp_path).as_posix()
    by_role = {p.role: p for p in finding.operating_points}
    assert set(by_role) == {"rule_out", "rule_in"}
    assert by_role["rule_out"].threshold == RULE_OUT
    assert by_role["rule_in"].threshold == RULE_IN
    for role, point in by_role.items():
        assert point.evidence_ref == f"{rel}#report.{role}"
        assert record.id in point.source
        assert point.on == "logit"
    # os pontos do manifesto (-3.0 / -2.5) não aparecem nem decidem
    assert all(p.threshold not in (-3.0, -2.5) for p in finding.operating_points)


@pytest.mark.requirement("REQ-192")
def test_decision_provenance_is_shown_in_the_report_text(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    _write(root, record, _envelope(record))
    finding = _finding(_run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root))
    text = _point(finding)
    assert text is not None
    assert "registro: evidence/measured/tb-sintetico/xr-probe-tb/tuberculosis.json#report" in text
    assert "(rule_in)" in text
    assert "(rule_out)" in text


# --------------------------------------------------------------------------- #
# Escala diferente e abstenções
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-192")
def test_probability_threshold_on_logit_adapter_is_not_converted(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    _write(
        root,
        record,
        _envelope(record, report_scale="probability", on="sigmoid", rule_out=0.3, rule_in=0.7),
    )
    result = _run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root)
    finding = _finding(result)
    assert finding.decision == "score_only"
    assert finding.operating_points == ()
    assert any("sem conversão exata" in w for w in result.warnings)


@pytest.mark.requirement("REQ-192")
def test_point_scale_disagreeing_with_report_scale_abstains(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    _write(root, record, _envelope(record, report_scale="probability", on="logit"))
    result = _run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root)
    assert _finding(result).decision == "score_only"
    assert any("relatório foi ajustado" in w for w in result.warnings)


@pytest.mark.requirement("REQ-192")
def test_point_threshold_disagreeing_with_report_abstains(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    _write(root, record, _envelope(record, report_rule_in=1.5))
    result = _run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root)
    assert _finding(result).decision == "score_only"
    assert any("não confere com o relatório" in w for w in result.warnings)


@pytest.mark.requirement("REQ-192")
def test_missing_basis_record_abstains_and_ignores_manifest_points(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    catalog = _catalog(EvidenceLevel.C, ("measured:ausente:xr-probe-tb:tuberculosis:x",))
    result = _run(make_registry, store, catalog, root)
    finding = _finding(result)
    assert finding.decision == "score_only"
    assert finding.operating_points == ()
    assert any("nenhum registro da base" in w for w in result.warnings)


@pytest.mark.requirement("REQ-192")
def test_absent_evidence_directory_abstains(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    catalog = _catalog(EvidenceLevel.C, ("measured:x:xr-probe-tb:tuberculosis:discrimination",))
    result = _run(make_registry, store, catalog, tmp_path / "nao-existe")
    assert _finding(result).decision == "score_only"


@pytest.mark.requirement("REQ-192")
@pytest.mark.parametrize(
    ("overrides", "warning"),
    [
        ({"protocol_id": "2025-01-v0"}, "difere do de levels.json"),
        ({"weights": "b" * 64}, "pesos medidos no registro"),
    ],
)
def test_record_from_other_protocol_or_weights_abstains(
    make_registry: Any,
    store: ArtifactStore,
    tmp_path: Path,
    overrides: dict[str, Any],
    warning: str,
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico", **overrides)
    _write(root, record, _envelope(record))
    result = _run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root)
    assert _finding(result).decision == "score_only"
    assert any(warning in w for w in result.warnings)


@pytest.mark.requirement("REQ-192")
def test_level_b_published_basis_without_points_is_score_only(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    published = EvidenceRecord(
        id="published:tb:xr-probe-tb",
        condition_id=TB[0],
        model_id=TB[1],
        label=TB[2],
        kind="published",
        passes_protocol=True,
        source=EvidenceSource(
            title="estudo sintético",
            url="https://example.invalid/estudo",
            quote="AUROC 0.9",
            peer_reviewed=True,
            verified_at_source=True,
        ),
    )
    path = root / "published" / "tb.json"
    path.parent.mkdir(parents=True)
    path.write_text(published.model_dump_json(), encoding="utf-8")
    result = _run(make_registry, store, _catalog(EvidenceLevel.B, (published.id,)), root)
    finding = _finding(result)
    assert finding.evidence_level is EvidenceLevel.B
    assert finding.decision == "score_only"
    assert any("não traz pontos de operação" in w for w in result.warnings)


@pytest.mark.requirement("REQ-192")
def test_level_d_keeps_manifest_points_in_research_mode(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    catalog = _catalog(EvidenceLevel.D, ())
    finding = _finding(_run(make_registry, store, catalog, root, pixel_level=40.0))
    assert finding.evidence_level is EvidenceLevel.D
    assert finding.visibility == "research"
    # logit (40 - 128) / 16 = -5.5 < -3.0 (rule_out do manifesto)
    assert finding.decision == "negative"
    assert {p.threshold for p in finding.operating_points} == {-3.0, -2.5}


# --------------------------------------------------------------------------- #
# Regra de escolha do registro determinante
# --------------------------------------------------------------------------- #
def _loaded(root: Path, records: list[EvidenceRecord]) -> list[LoadedEvidence]:
    for record in records:
        _write(root, record, _envelope(record))
    return load_evidence(root)


@pytest.mark.requirement("REQ-192")
def test_selection_prefers_highest_level_contribution(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    clean = _record("externo-limpo", kind="measured_external", leakage="none", n_total=60)
    internal = _record("interno-grande", n_total=5000)
    chosen = select_basis(_loaded(root, [internal, clean]), tmp_path)
    assert chosen.record.id == clean.id  # A vence C mesmo com n menor


@pytest.mark.requirement("REQ-192")
def test_selection_then_prefers_larger_n_then_path(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    small = _record("aaa-pequeno", n_total=100)
    large = _record("zzz-grande", n_total=400)
    assert select_basis(_loaded(root, [small, large]), tmp_path).record.id == large.id

    root2 = tmp_path / "evidence2"
    first = _record("bbb", n_total=300)
    second = _record("aaa", n_total=300)
    loaded = _loaded(root2, [first, second])
    assert select_basis(loaded, tmp_path).record.id == second.id
    assert select_basis(list(reversed(loaded)), tmp_path).record.id == second.id


@pytest.mark.requirement("REQ-192")
def test_pipeline_uses_only_the_selected_record(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    small = _record("aaa-pequeno", n_total=100)
    large = _record("zzz-grande", n_total=400)
    _write(root, small, _envelope(small, rule_out=-9.0, rule_in=9.0))
    _write(root, large, _envelope(large))
    catalog = _catalog(EvidenceLevel.C, (small.id, large.id))
    finding = _finding(_run(make_registry, store, catalog, root, pixel_level=220.0))
    assert finding.decision == "positive"  # com o registro pequeno seria indeterminado
    assert {p.threshold for p in finding.operating_points} == {RULE_OUT, RULE_IN}
    assert all(large.id in p.source for p in finding.operating_points)


@pytest.mark.requirement("REQ-192")
def test_select_basis_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ao menos um"):
        select_basis([], tmp_path)


# --------------------------------------------------------------------------- #
# Suspensão de decisão (REQ-193)
# --------------------------------------------------------------------------- #
def _hold_file(root: Path, triple: tuple[str, str, str] = TB, extra: str = "") -> None:
    cid, mid, label = triple
    root.mkdir(parents=True, exist_ok=True)
    (root / "decision_holds.yaml").write_text(
        "format: radiologyai.evidence/decision-holds-v1\n"
        "holds:\n"
        f"  - condition_id: {cid}\n"
        f"    model_id: {mid}\n"
        f"    label: {label}\n"
        "    reason: aprovação dependente do sorteio\n"
        "    decided_on: 2026-11-27\n"
        "    decided_by: responsável\n"
        "    evidence_ref: HONEST_STATUS.md\n" + extra,
        encoding="utf-8",
    )


@pytest.mark.requirement("REQ-193")
def test_hold_removes_decision_but_keeps_level(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    _write(root, record, _envelope(record))
    _hold_file(root)
    result = _run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root)
    finding = _finding(result)
    assert finding.evidence_level is EvidenceLevel.C
    assert finding.decision == "score_only"
    assert finding.operating_points == ()
    assert any("decisão suspensa" in w and "sorteio" in w for w in result.warnings)


@pytest.mark.requirement("REQ-193")
def test_hold_for_other_triple_does_not_affect_decision(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    _write(root, record, _envelope(record))
    _hold_file(root, ("cxr.cardiomegaly", "xrv-densenet121-res224-pc", "cardiomegaly"))
    finding = _finding(_run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root))
    assert finding.decision == "positive"


@pytest.mark.requirement("REQ-193")
def test_invalid_or_duplicated_holds_fail_loudly(tmp_path: Path) -> None:
    from radiologyai.analysis.evidence_policy import load_holds
    from radiologyai.errors import EvidenceError

    root = tmp_path / "evidence"
    assert load_holds(root) == {}
    _hold_file(
        root,
        extra=(
            "  - condition_id: cxr.tuberculosis\n    model_id: xr-probe-tb\n"
            "    label: tuberculosis\n    reason: r\n    decided_on: 2026-11-27\n"
            "    decided_by: x\n    evidence_ref: y\n"
        ),
    )
    with pytest.raises(EvidenceError, match="repetido"):
        load_holds(root)
    (root / "decision_holds.yaml").write_text("format: outro\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="inválidas"):
        load_holds(root)


@pytest.mark.requirement("REQ-193")
def test_repo_holds_file_is_valid_and_points_to_catalog_triples() -> None:
    from radiologyai.analysis.evidence_policy import load_holds
    from radiologyai.evidence.store import EVIDENCE_ROOT

    holds = load_holds(EVIDENCE_ROOT)
    assert ("cxr.cardiomegaly", "xrv-densenet121-res224-pc", "cardiomegaly") in holds
    catalog = Catalog.load()
    for cid, mid, label in holds:
        assert catalog.entry(cid, mid, label) is not None, (cid, mid, label)


@pytest.mark.requirement("REQ-193")
def test_hold_reason_is_on_the_finding_and_in_the_report_text(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    _write(root, record, _envelope(record))
    _hold_file(root)
    finding = _finding(_run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root))
    assert finding.decision_note is not None
    assert "decisão suspensa" in finding.decision_note
    assert "sorteio" in finding.decision_note
    text = _point(finding)
    assert text is not None
    assert text.startswith("sem decisão (só escore):")
    assert "sorteio" in text


@pytest.mark.requirement("REQ-193")
def test_decision_note_only_allowed_with_score_only(
    make_registry: Any, store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "evidence"
    record = _record("tb-sintetico")
    _write(root, record, _envelope(record))
    finding = _finding(_run(make_registry, store, _catalog(EvidenceLevel.C, (record.id,)), root))
    assert finding.decision == "positive"
    assert finding.decision_note is None
    with pytest.raises(ValueError, match="decision_note"):
        Finding.model_validate({**finding.model_dump(), "decision_note": "x"})
