#!/usr/bin/env python3
"""Valida o relatório CITI antes de enviá-lo à PhysioNet.

Três erros custam uma rodada de revisão (dias de espera) e são detectáveis em
segundos:

1. Enviar o **Completion Certificate** em vez do **Completion Report**. São
   arquivos diferentes; a PhysioNet exige o Report, que lista os módulos e as
   notas. O Certificate é uma página só, decorativa.
2. Ter feito o curso errado. O exigido é *Data or Specimens Only Research*,
   não o curso completo de sujeitos humanos.
3. Relatório expirado. A PhysioNet recusa treinamento vencido.

Uso:
    python scripts/check_citi_report.py relatorio.pdf
    python scripts/check_citi_report.py relatorio.pdf --nome "Guilherme Capel"
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

CURSO_EXIGIDO = "data or specimens only research"

# O Report lista módulos e notas; o Certificate não.
MARCAS_DE_REPORT = ("completion report", "required modules", "most recent score")
MARCAS_DE_CERTIFICATE = ("completion certificate", "certificate of completion")

RE_DATA = re.compile(r"(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{4})")
RE_DATA_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
MESES = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}


@dataclass
class Resultado:
    """Achados da verificação."""

    caminho: Path
    erros: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    @property
    def aprovado(self) -> bool:
        return not self.erros

    def relatar(self, escrever=print) -> None:
        for i in self.infos:
            escrever(f"  ok      {i}")
        for a in self.avisos:
            escrever(f"  atenção {a}")
        for e in self.erros:
            escrever(f"  ERRO    {e}")
        escrever("")
        escrever(
            "APROVADO — pode enviar à PhysioNet."
            if self.aprovado
            else "NÃO ENVIE ainda: corrija os erros acima."
        )


def normalizar(texto: str) -> str:
    """Minúsculas, sem acento e com espaços colapsados."""
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem_acento.lower())


def extrair_texto(caminho: Path) -> str:
    """Extrai texto de um PDF. Exige pypdf.

    Raises:
        RuntimeError: pypdf ausente ou PDF ilegível.
    """
    import importlib.util

    if importlib.util.find_spec("pypdf") is None:
        raise RuntimeError("pypdf é necessário: pip install pypdf")

    from pypdf import PdfReader

    try:
        leitor = PdfReader(str(caminho))
        return "\n".join(pagina.extract_text() or "" for pagina in leitor.pages)
    except Exception as exc:  # noqa: BLE001 - reembalado com o caminho
        raise RuntimeError(f"não consegui ler {caminho}: {exc}") from exc


def _datas(texto: str) -> list[date]:
    achadas: list[date] = []
    for d, m, a in RE_DATA.findall(texto):
        mes = MESES.get(m.lower()[:3])
        if mes:
            try:
                achadas.append(date(int(a), mes, int(d)))
            except ValueError:
                continue
    for a, m, d in RE_DATA_ISO.findall(texto):
        try:
            achadas.append(date(int(a), int(m), int(d)))
        except ValueError:
            continue
    return achadas


def verificar(
    caminho: str | Path, *, nome: str | None = None, hoje: date | None = None
) -> Resultado:
    """Verifica um relatório CITI.

    Args:
        nome: se dado, confere que aparece no documento — a PhysioNet compara
            com o nome da conta, e divergência reprova.
        hoje: data de referência para expiração (injetável para teste).
    """
    caminho = Path(caminho)
    resultado = Resultado(caminho=caminho)

    if not caminho.is_file():
        resultado.erros.append(f"arquivo não encontrado: {caminho}")
        return resultado
    if caminho.suffix.lower() != ".pdf":
        resultado.avisos.append(f"extensão {caminho.suffix!r}; o CITI entrega PDF")

    try:
        texto = normalizar(extrair_texto(caminho))
    except RuntimeError as exc:
        resultado.erros.append(str(exc))
        return resultado

    if not texto.strip():
        resultado.erros.append("PDF sem texto extraível (digitalizado?). Baixe o original do CITI.")
        return resultado

    # 1. Report, não Certificate
    eh_report = any(m in texto for m in MARCAS_DE_REPORT)
    eh_certificate = any(m in texto for m in MARCAS_DE_CERTIFICATE)
    if eh_report:
        resultado.infos.append("é o Completion Report (lista módulos e notas)")
    elif eh_certificate:
        resultado.erros.append(
            "este é o Completion CERTIFICATE. A PhysioNet exige o Completion REPORT — "
            "no CITI, em Records, escolha 'View-Print-Share' e baixe o Report."
        )
    else:
        resultado.avisos.append(
            "não identifiquei se é Report ou Certificate; confira que lista os módulos"
        )

    # 2. Curso correto
    if CURSO_EXIGIDO in texto:
        resultado.infos.append("curso: Data or Specimens Only Research")
    else:
        resultado.erros.append(
            "não encontrei 'Data or Specimens Only Research'. É o curso exigido "
            "pela PhysioNet; o curso completo de sujeitos humanos não substitui."
        )

    # 3. Nome
    if nome:
        partes = [p for p in normalizar(nome).split() if len(p) > 2]
        faltando = [p for p in partes if p not in texto]
        if faltando:
            resultado.avisos.append(
                f"não achei no documento: {faltando}. "
                "O nome precisa bater com o da conta PhysioNet."
            )
        else:
            resultado.infos.append(f"nome encontrado: {nome}")

    # 4. Expiração
    hoje = hoje or datetime.now(UTC).date()
    futuras = [d for d in _datas(texto) if d > hoje]
    passadas = [d for d in _datas(texto) if d <= hoje]
    if "expiration" in texto or "expire" in texto:
        if futuras:
            resultado.infos.append(f"validade até {max(futuras).isoformat()}")
        elif passadas:
            resultado.erros.append(
                f"nenhuma data futura no documento; a mais recente é {max(passadas).isoformat()}. "
                "Treinamento possivelmente vencido — a PhysioNet recusa."
            )
    elif passadas:
        resultado.infos.append(f"concluído em {max(passadas).isoformat()}")

    return resultado


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("relatorio", type=Path)
    parser.add_argument("--nome", help="nome como consta na conta PhysioNet")
    args = parser.parse_args()

    print(f"Verificando {args.relatorio.name}\n")
    resultado = verificar(args.relatorio, nome=args.nome)
    resultado.relatar()
    return 0 if resultado.aprovado else 1


if __name__ == "__main__":
    sys.exit(main())
