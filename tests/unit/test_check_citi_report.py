"""Validador do relatório CITI — pega os três erros que custam uma rodada."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_citi_report.py"
TEM_PDF = all(importlib.util.find_spec(m) is not None for m in ("pypdf", "reportlab"))
pytestmark = pytest.mark.skipif(not TEM_PDF, reason="requer pypdf e reportlab")


@pytest.fixture(scope="module")
def citi():
    spec = importlib.util.spec_from_file_location("check_citi", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_citi"] = mod
    spec.loader.exec_module(mod)
    return mod


def pdf(tmp_path, linhas, nome="rel.pdf"):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    caminho = tmp_path / nome
    c = canvas.Canvas(str(caminho), pagesize=letter)
    y = 750
    for linha in linhas:
        c.drawString(60, y, linha)
        y -= 16
    c.save()
    return caminho


REPORT_BOM = [
    "CITI Program",
    "COMPLETION REPORT - PART 1 OF 2",
    "Name: Guilherme Capel Pasqua",
    "Institution Affiliation: Massachusetts Institute of Technology Affiliates",
    "Curriculum Group: Data or Specimens Only Research",
    "REQUIRED MODULES                     MOST RECENT      SCORE",
    "Belmont Report and Its Principles    12-Sep-2026      5/5",
    "Records-Based Research               12-Sep-2026      5/5",
    "Expiration Date: 12-Sep-2029",
]


class TestReportValido:
    def test_aprova(self, citi, tmp_path):
        r = citi.verificar(pdf(tmp_path, REPORT_BOM), hoje=date(2026, 9, 13))
        assert r.aprovado, r.erros

    def test_identifica_report(self, citi, tmp_path):
        r = citi.verificar(pdf(tmp_path, REPORT_BOM), hoje=date(2026, 9, 13))
        assert any("Completion Report" in i for i in r.infos)

    def test_identifica_curso(self, citi, tmp_path):
        r = citi.verificar(pdf(tmp_path, REPORT_BOM), hoje=date(2026, 9, 13))
        assert any("Data or Specimens Only" in i for i in r.infos)

    def test_le_validade(self, citi, tmp_path):
        r = citi.verificar(pdf(tmp_path, REPORT_BOM), hoje=date(2026, 9, 13))
        assert any("2029-09-12" in i for i in r.infos)

    def test_confere_nome(self, citi, tmp_path):
        r = citi.verificar(
            pdf(tmp_path, REPORT_BOM), nome="Guilherme Capel Pasqua", hoje=date(2026, 9, 13)
        )
        assert r.aprovado
        assert not any("não achei" in a for a in r.avisos)


class TestOsTresErros:
    def test_certificate_em_vez_de_report(self, citi, tmp_path):
        """O erro mais comum: o CITI oferece os dois lado a lado."""
        linhas = [
            "CITI Program",
            "COMPLETION CERTIFICATE",
            "Name: Guilherme Capel Pasqua",
            "Data or Specimens Only Research",
            "Completion Date 12-Sep-2026",
        ]
        r = citi.verificar(pdf(tmp_path, linhas), hoje=date(2026, 9, 13))
        assert not r.aprovado
        assert any("CERTIFICATE" in e for e in r.erros)
        assert any("View-Print-Share" in e for e in r.erros)

    def test_curso_errado(self, citi, tmp_path):
        linhas = [
            "COMPLETION REPORT - PART 1 OF 2",
            "Name: Guilherme Capel Pasqua",
            "Curriculum Group: Biomedical Research Investigators",
            "REQUIRED MODULES  MOST RECENT SCORE",
            "Expiration Date: 12-Sep-2029",
        ]
        r = citi.verificar(pdf(tmp_path, linhas), hoje=date(2026, 9, 13))
        assert not r.aprovado
        assert any("Data or Specimens Only Research" in e for e in r.erros)

    def test_treinamento_vencido(self, citi, tmp_path):
        linhas = [*REPORT_BOM[:-1], "Expiration Date: 01-Jan-2024"]
        r = citi.verificar(pdf(tmp_path, linhas), hoje=date(2026, 9, 13))
        assert not r.aprovado
        assert any("vencido" in e for e in r.erros)


class TestEntradasRuins:
    def test_arquivo_ausente(self, citi, tmp_path):
        r = citi.verificar(tmp_path / "nao_existe.pdf")
        assert not r.aprovado
        assert any("não encontrado" in e for e in r.erros)

    def test_pdf_sem_texto(self, citi, tmp_path):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        caminho = tmp_path / "vazio.pdf"
        canvas.Canvas(str(caminho), pagesize=letter).save()
        r = citi.verificar(caminho)
        assert not r.aprovado
        assert any("sem texto" in e for e in r.erros)

    def test_nome_divergente_avisa(self, citi, tmp_path):
        r = citi.verificar(
            pdf(tmp_path, REPORT_BOM), nome="Outra Pessoa Qualquer", hoje=date(2026, 9, 13)
        )
        assert any("não achei" in a for a in r.avisos)

    def test_nome_divergente_nao_reprova(self, citi, tmp_path):
        """Divergência de nome é aviso: apelidos e nomes do meio variam."""
        r = citi.verificar(
            pdf(tmp_path, REPORT_BOM), nome="Outra Pessoa Qualquer", hoje=date(2026, 9, 13)
        )
        assert r.aprovado


class TestNormalizacao:
    def test_ignora_acento_e_caixa(self, citi):
        assert citi.normalizar("Ação  DE   Pesquisa") == "acao de pesquisa"
