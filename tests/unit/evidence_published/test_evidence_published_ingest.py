"""Testes da ingestão de evidência publicada (``evidence/published/``).

Os testes sobre os arquivos gerados rodam sempre (os arquivos estão no
repositório). Os que re-executam a ingestão exigem os arquivos de pesquisa
conferidos e são pulados sem eles.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from radiologyai.catalog import load_protocol
from radiologyai.catalog.schema import Condition, EvidenceRecord, compute_level
from radiologyai.evidence.levels import build_levels
from radiologyai.evidence.store import load_records

REPO = Path(__file__).resolve().parents[3]
PUBLISHED = REPO / "evidence" / "published"
SCRIPT = REPO / "scripts" / "evidence_ingest_published.py"
RESEARCH = Path("/root/medai-cache/research")
RECORD_FILES = ("cxr.yaml", "ct_mr.yaml", "echo_mg_us.yaml")

# Ids de modelo usados pelos adaptadores (contrato do catálogo).
OUR_MODEL_IDS = frozenset(
    {f"xrv-densenet121-{s}" for s in ("all", "pc", "chex", "mimic_ch", "mimic_nb", "nih", "rsna")}
    | {
        "xrv-resnet50-512-all",
        "xrv-pspnet-anatomy",
        "ct-biomarkers",
        "mammoscreen",
        "monai-breast-density",
        "medsam2-echo-lv",
        "chexzero",
        "biomedclip",
        "merlin",
        "monai-brats",
        "lst-ai",
    }
)

# Publicações sobre modelos/pesos que NÃO rodamos: nunca podem virar registro anexado.
FOREIGN_MODEL_MARKERS = (
    "CheXzero",
    "EchoPrime",
    "PanEcho",
    "MedSAM",
    "Mirai",
    "TotalSegmentator v1",
    "TotalSegmentator MRI",
    "jfhealthcare",
    "CheXpert-DenseNet121",
    "qXR",
    "CAD4TB",
    "Lunit",
    "Google deep-learning",
    "liver_lesions_mr",
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evidence_ingest_published", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolvem anotações pelo módulo
    spec.loader.exec_module(module)
    return module


ingest = _load_script()
HAS_RESEARCH = all((RESEARCH / f).is_file() for f, _ in ingest.INPUTS)


def _yaml(name: str) -> dict[str, Any]:
    data = yaml.safe_load((PUBLISHED / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _records() -> list[EvidenceRecord]:
    return [
        EvidenceRecord.model_validate(item)
        for name in RECORD_FILES
        for item in _yaml(name)["records"]
    ]


# --------------------------------------------------------------------------- #
# Arquivos gerados
# --------------------------------------------------------------------------- #
class TestGeneratedRecords:
    def test_store_reads_every_file_without_error(self) -> None:
        records = load_records(REPO / "evidence")
        published_ids = {r.id for r in _records()}
        assert published_ids <= {r.id for r in records}
        assert len(published_ids) == len(_records()), "ids duplicados"

    def test_every_record_is_published_kind_with_verified_source(self) -> None:
        records = _records()
        assert records, "nenhum registro gerado"
        for rec in records:
            assert rec.kind in ("published", "guideline_threshold")
            assert rec.source is not None
            assert rec.source.quote.strip()
            assert rec.provenance is None
            assert rec.id.startswith("pub.")

    def test_quote_contains_every_metric_value_and_ci(self) -> None:
        for rec in _records():
            assert rec.source is not None
            assert rec.metrics, rec.id
            for name, metric in rec.metrics.items():
                assert ingest.value_in_quote(metric.value, rec.source.quote), (rec.id, name)
                for bound in (metric.ci_low, metric.ci_high):
                    if bound is not None:
                        assert ingest.value_in_quote(bound, rec.source.quote), (rec.id, name)

    def test_no_record_attached_to_foreign_model_or_weights(self) -> None:
        for rec in _records():
            assert rec.model_id in OUR_MODEL_IDS or rec.model_id.startswith("totalseg-"), rec.id
            assert rec.source is not None
            for marker in FOREIGN_MODEL_MARKERS:
                assert marker not in rec.source.title, (rec.id, marker)

    def test_mammoscreen_mean_of_three_splits_is_context_only(self) -> None:
        # O cartão publica a média de 3 conjuntos de pesos; distribuímos só o split 1.
        ids = {r.id for r in _records() if r.model_id == "mammoscreen"}
        assert "pub.mammoscreen_cancer_auroc" not in ids
        context = {c["source_entry"]: c for c in _yaml("context.yaml")["context"]}
        assert context["mammoscreen_cancer_auroc"]["reason_code"] == "pesos_diferentes"

    def test_passes_protocol_is_reproduced_by_the_judge(self) -> None:
        protocol = load_protocol()
        for rec in _records():
            assert rec.source is not None
            metrics = {k: m.model_dump() for k, m in rec.metrics.items()}
            passes, reasons = ingest.judge(
                kind=rec.kind,
                endpoint=rec.endpoint,
                metrics=metrics,
                n_total=rec.n_total,
                n_positive=rec.n_positive,
                peer_reviewed=rec.source.peer_reviewed,
                verified_at_source=rec.source.verified_at_source,
                protocol=protocol,
            )
            assert passes is rec.passes_protocol, rec.id
            assert tuple(reasons) == rec.failure_reasons, rec.id

    def test_passing_records_meet_the_published_bar_independently(self) -> None:
        protocol = load_protocol()
        for rec in _records():
            if not rec.passes_protocol:
                assert rec.failure_reasons, rec.id
                continue
            assert rec.source is not None
            assert rec.source.peer_reviewed
            assert rec.source.verified_at_source
            assert rec.n_total is not None
            assert rec.n_positive is not None
            assert rec.n_positive >= protocol.published_min_positive
            for m in rec.metrics.values():
                assert m.ci_low is not None
                assert m.ci_high is not None

    def test_records_without_n_or_ci_never_pass(self) -> None:
        for rec in _records():
            no_ci = any(m.ci_low is None or m.ci_high is None for m in rec.metrics.values())
            if rec.n_total is None or rec.n_positive is None or no_ci:
                assert rec.passes_protocol is False, rec.id

    def test_internal_validation_declares_leakage(self) -> None:
        for rec in _records():
            if any(c == "tipo de validação na fonte: internal" for c in rec.caveats):
                assert rec.leakage != "none", rec.id
                assert rec.leakage_note, rec.id


class TestProposedConditions:
    def test_every_referenced_condition_is_proposed_and_valid(self) -> None:
        data = _yaml("conditions_proposed.yaml")
        assert data["records"] == []
        attached = [Condition.model_validate(c) for c in data["conditions"]]
        context_only = [Condition.model_validate(c) for c in data["conditions_context_only"]]
        ids = [c.id for c in attached + context_only]
        assert len(ids) == len(set(ids))
        for cond in attached + context_only:
            assert cond.id.split(".", 1)[0] == cond.group
            assert cond.code.scheme == "99MEDAI"
            assert cond.code.value == cond.id
            assert cond.description_pt
            assert "confirmação médica" in cond.description_pt
        assert {r.condition_id for r in _records()} == {c.id for c in attached}
        ctx_ids = {c["condition_id"] for c in _yaml("context.yaml")["context"]} - {None}
        assert ctx_ids <= set(ids)

    def test_levels_build_with_merged_conditions(self) -> None:
        data = _yaml("conditions_proposed.yaml")
        conditions = [Condition.model_validate(c) for c in data["conditions"]]
        records = _records()
        levels = build_levels(records, conditions, load_protocol())
        triples = {(r.condition_id, r.model_id, r.label) for r in records}
        assert {(e.condition_id, e.model_id, e.label) for e in levels.entries} == triples
        for entry in levels.entries:
            group = [
                r
                for r in records
                if (r.condition_id, r.model_id, r.label)
                == (entry.condition_id, entry.model_id, entry.label)
            ]
            assert entry.level == compute_level(group)[0]


class TestContext:
    def test_context_items_are_explained_and_store_safe(self) -> None:
        data = _yaml("context.yaml")
        assert data["records"] == []
        assert data["context"]
        for item in data["context"]:
            assert item["reason_code"] in ingest.CONTEXT_REASONS
            assert item["reason"]
            assert item["source"]["quote"]
            assert item["source"]["url"].startswith("https://")


# --------------------------------------------------------------------------- #
# Funções puras
# --------------------------------------------------------------------------- #
class TestParsing:
    @pytest.mark.parametrize(
        ("raw", "metric", "expected", "bound"),
        [
            ("97%", "sensitivity", 0.97, None),
            ("75.8", "sensitivity", 0.758, None),
            ("0.9464", "auroc", 0.9464, None),
            (">75%", "sensitivity", 0.75, ">"),
            ("4.79", "mae", 4.79, None),
        ],
    )
    def test_parse_value(self, raw: str, metric: str, expected: float, bound: str | None) -> None:
        parsed = ingest.parse_value(raw, metric)
        assert parsed.value == pytest.approx(expected)
        assert parsed.bound == bound

    def test_parse_value_rejects_ranges(self) -> None:
        with pytest.raises(ingest.IngestError):
            ingest.parse_value("0.977-1.000", "auroc")

    def test_parse_ci(self) -> None:
        assert ingest.parse_ci("91.48-99.38", "sensitivity", True) == (0.9148, 0.9938)
        assert ingest.parse_ci([0.777, 0.852], "auroc", False) == (0.777, 0.852)
        assert ingest.parse_ci("0.74 to 0.80", "c_index", False) == (0.74, 0.8)
        assert ingest.parse_ci("±0.02", "other", False) is None
        assert ingest.parse_ci(None, "auroc", False) is None

    def test_value_in_quote(self) -> None:
        assert ingest.value_in_quote(0.97, "was 97% (95% CI 91.48−99.38%)")
        assert ingest.value_in_quote(0.6, "| 0.60 | 0.85 |")
        assert not ingest.value_in_quote(0.61, "| 0.60 | 0.85 |")
        assert not ingest.value_in_quote(0.8, "AUC 0.81")

    def test_unknown_model_is_refused(self) -> None:
        entry = {"id": "x", "model_or_method": "Modelo Inventado 9000"}
        with pytest.raises(ingest.IngestError, match="sem regra"):
            ingest.match_model(entry)


class TestJudge:
    def _judge(self, **kw: Any) -> tuple[bool, tuple[str, ...]]:
        base: dict[str, Any] = {
            "kind": "published",
            "endpoint": "discrimination",
            "metrics": {"auroc": {"value": 0.9, "ci_low": 0.85, "ci_high": 0.94}},
            "n_total": 500,
            "n_positive": 120,
            "peer_reviewed": True,
            "verified_at_source": True,
            "protocol": load_protocol(),
        }
        base.update(kw)
        result: tuple[bool, tuple[str, ...]] = ingest.judge(**base)
        return result

    def test_complete_peer_reviewed_record_passes(self) -> None:
        assert self._judge() == (True, ())

    @pytest.mark.parametrize(
        ("override", "fragment"),
        [
            ({"peer_reviewed": False}, "revisada por pares"),
            ({"verified_at_source": False}, "não conferida"),
            ({"n_positive": None}, "positivos não publicado"),
            ({"n_positive": 12}, "12 positivos"),
            ({"n_total": None}, "n do conjunto"),
            ({"metrics": {"auroc": {"value": 0.79, "ci_low": 0.7, "ci_high": 0.85}}}, "0.790"),
            ({"metrics": {"auroc": {"value": 0.9, "ci_low": None, "ci_high": None}}}, "IC 95%"),
            ({"metrics": {"accuracy": {"value": 0.96}}}, "não tem barra"),
        ],
    )
    def test_failures_are_explained(self, override: dict[str, Any], fragment: str) -> None:
        passes, reasons = self._judge(**override)
        assert passes is False
        assert any(fragment in r for r in reasons), reasons

    def test_detection_needs_both_sensitivity_and_specificity(self) -> None:
        passes, reasons = self._judge(
            endpoint="detection",
            metrics={"sensitivity": {"value": 0.95, "ci_low": 0.9, "ci_high": 0.98}},
        )
        assert passes is False
        assert any("especificidade" in r for r in reasons)

    def test_segmentation_uses_dice_bar(self) -> None:
        passes, reasons = self._judge(
            endpoint="segmentation",
            metrics={"dice": {"value": 0.84, "ci_low": 0.8, "ci_high": 0.88}},
        )
        assert passes is False
        assert any(re.search(r"Dice 0\.840 < barra de 0\.85", r) for r in reasons)


# --------------------------------------------------------------------------- #
# Re-execução (exige os arquivos de pesquisa conferidos)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_RESEARCH, reason="arquivos *.verified.md de pesquisa ausentes")
class TestReRun:
    def test_ingestion_is_deterministic_and_committed_output_is_current(self) -> None:
        protocol = load_protocol()
        first = ingest.render(*ingest.build(RESEARCH, protocol), protocol)
        second = ingest.render(*ingest.build(RESEARCH, protocol), protocol)
        assert first == second
        for name, text in first.items():
            assert (PUBLISHED / name).read_text(encoding="utf-8") == text, name

    def test_check_mode_reports_stale_output(self, tmp_path: Path) -> None:
        assert ingest.main(["--out", str(tmp_path)]) == 0
        assert ingest.main(["--out", str(tmp_path), "--check"]) == 0
        (tmp_path / "cxr.yaml").write_text("records: []\n", encoding="utf-8")
        assert ingest.main(["--out", str(tmp_path), "--check"]) == 1

    def test_every_input_entry_is_accounted_for(self) -> None:
        protocol = load_protocol()
        built, _ = ingest.build(RESEARCH, protocol)
        merged = {m for members in ingest.MERGES.values() for m in members}
        for filename, out_name in ingest.INPUTS:
            entries = {e["id"] for e in ingest.read_entries(RESEARCH / filename)}
            b = built[out_name]
            in_context = {c["source_entry"] for c in b.context}
            attached = {r["id"].removeprefix("pub.") for r in b.records}
            assert entries <= in_context | attached | merged
