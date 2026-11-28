"""R1 do guardião reconhece as métricas e os formatos numéricos do projeto (REQ-187)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_honesty.py"


@pytest.fixture
def guard(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("check_honesty_r1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_honesty_r1"] = module
    spec.loader.exec_module(module)
    src = tmp_path / "src" / "radiologyai"
    src.mkdir(parents=True)
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "SRC", src)
    module.tmp_repo = tmp_path
    module.tmp_src = src
    return module


# As seis linhas que o revisor mostrou passarem pelo R1 antigo.
UNBACKED = [
    "Macro-AUROC: 0,7089",
    "| xrv-densenet121-pc | 0,7938 |",
    "AUC: 0,94",
    "AUROC: 0.94",
    "sensibilidade 0,90",
    "mediu AUROC 0,664 e fibrose saiu em 0,448",
    "ICC(2,1) 0,861 no CAMUS",
    "Dice médio 0.91",
    "ECE sem calibração: `0,22526`",
    "sensibilidade de 87%",
]


def _md(guard, text: str) -> list:
    (guard.tmp_repo / "doc.md").write_text(text, encoding="utf-8")
    return guard.rule_1_performance_claims()


@pytest.mark.requirement("REQ-187")
@pytest.mark.parametrize("line", UNBACKED)
def test_project_metric_formats_are_caught(guard, line):
    if line.startswith("|"):
        line = "| Modelo | AUROC |\n|---|---|\n" + line
    assert _md(guard, line)


@pytest.mark.requirement("REQ-187")
@pytest.mark.parametrize(
    "line",
    [
        "AUROC 0,94 em `evidence/measured/tb-shenzhen/xr-probe-tb/tuberculosis.json`",
        "Macro-AUROC 0,6644 (`artifacts/eval/run/metrics.json`)",
        "AUROC 0,7089 — NÃO VERIFICÁVEL (execução sem artefato)",
        "barra do protocolo: AUROC >= 0,80 e limite inferior do IC >= 0,75",
        "ICC mínimo 0,75",
        "o ICC(2,1) é a forma usada",
        "AUROC acima de 0,80 é a barra",
    ],
)
def test_backed_claims_and_thresholds_pass(guard, line):
    assert not _md(guard, line)


@pytest.mark.requirement("REQ-187")
def test_table_provenance_in_caption_is_accepted(guard):
    text = (
        "Fonte: `artifacts/eval/run/metrics.json`\n\n"
        "| Achado | AUROC |\n|---|---|\n| Derrame | 0,81 |\n"
    )
    assert not _md(guard, text)
    assert _md(guard, "Tabela\n\n| Achado | AUROC |\n|---|---|\n| Derrame | 0,81 |\n")


@pytest.mark.requirement("REQ-187")
def test_table_without_metric_header_is_not_a_claim(guard):
    assert not _md(guard, "| Dataset | Pacientes |\n|---|---|\n| CAMUS | 100,5 |\n")


@pytest.mark.requirement("REQ-187")
def test_generated_report_backed_by_existing_artifact_passes(guard):
    reports = guard.tmp_repo / "reports"
    reports.mkdir()
    name = "EVALUATION_nih_m1__20260912T202549Z.md"
    (reports / name).write_text("| Achado | AUROC |\n|---|---|\n| x | 0.66 |\n", encoding="utf-8")
    assert guard.rule_1_performance_claims()  # artefato ainda não existe
    art = guard.tmp_repo / "artifacts" / "eval" / "m1__20260912T202549Z"
    art.mkdir(parents=True)
    (art / "metrics.json").write_text("{}", encoding="utf-8")
    assert not guard.rule_1_performance_claims()


@pytest.mark.requirement("REQ-187")
def test_docstring_claims_in_src_are_caught(guard):
    (guard.tmp_src / "m.py").write_text(
        '"""O codificador transferiu com AUROC 0,9397 contra 0,8281."""\n'
        "x = 1  # nenhum modelo validado clinicamente\n",
        encoding="utf-8",
    )
    found = guard.rule_1_performance_claims()
    assert [v.line for v in found] == [1]


@pytest.mark.requirement("REQ-187")
def test_code_formatting_is_not_a_claim(guard):
    (guard.tmp_src / "m.py").write_text(
        "def f(e):\n    return f\"{e['auroc']:.3f}\"\n", encoding="utf-8"
    )
    assert not guard.rule_1_performance_claims()
