#!/usr/bin/env python3
"""Gera a lista SOUP (Software of Unknown Provenance) — IEC 62304 §5.3.3, §8.1.2.

SOUP é todo software de terceiros incorporado ao dispositivo cuja qualificação
não foi feita por nós. A norma exige, para cada item: identificação, versão,
fabricante, propósito no sistema e a lista de anomalias publicadas.

É por isso que o lockfile é artefato regulatório, e por isso um
``requirements.txt`` apenas com ``>=`` é defeito de conformidade — não desleixo.
Sem versão exata não há SOUP verificável.

Uso:
    python scripts/soup.py                 # imprime a tabela
    python scripts/soup.py --write         # grava docs/regulatory/08-soup-list.md
    python scripts/soup.py --check         # falha se o documento estiver desatualizado
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "docs" / "regulatory" / "08-soup-list.md"

# Propósito de cada dependência no sistema. Exigido pela norma; não pode ser
# derivado automaticamente, então é mantido aqui e verificado por cobertura.
PURPOSE: dict[str, str] = {
    "numpy": "Aritmética de arrays em todo o pipeline de imagem e métricas",
    "pydicom": "Parsing de DICOM, acesso a tags e des-identificação",
    "pydantic": "Validação de contratos de dados (model cards, metadados, resultados)",
    "pydantic-settings": "Carga de configuração a partir do ambiente",
    "pyyaml": "Leitura de model cards em YAML",
    "typer": "Interface de linha de comando",
    "simpleitk": "Leitura de séries e reamostragem volumétrica (TC/RM)",
    "pillow": "Decodificação de imagem não-DICOM",
    "scikit-learn": "Métricas de referência e utilidades de avaliação",
    "scipy": "Estatística usada nos intervalos de confiança",
    "torch": "Runtime de inferência de rede neural",
    "torchvision": "Transformações de imagem para modelos torch",
    "timm": "Arquiteturas de backbone pré-treinadas",
    "torchxrayvision": "Modelos de radiografia de tórax pré-treinados e publicados",
    "monai": "Transformações e arquiteturas de imagem médica (2D e 3D)",
    "onnxruntime": "Runtime de inferência para implantação em CPU",
    "fastapi": "Camada de API REST",
    "uvicorn": "Servidor ASGI",
    "python-multipart": "Upload de arquivos na API",
}

# Classificação de criticidade para o dossiê: uma falha do item pode contribuir
# para uma situação perigosa? (ISO 14971 + IEC 62304 §7)
SAFETY_RELEVANT = {
    "numpy",
    "pydicom",
    "torch",
    "onnxruntime",
    "monai",
    "simpleitk",
    "timm",
    "torchxrayvision",
    "pydantic",
}


@dataclass(frozen=True)
class SoupItem:
    name: str
    version_spec: str
    extra: str
    purpose: str
    safety_relevant: bool


def parse_dependencies() -> list[SoupItem]:
    """Extrai as dependências declaradas em pyproject.toml."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]

    groups: list[tuple[str, list[str]]] = [("core", project.get("dependencies", []))]
    for name, deps in project.get("optional-dependencies", {}).items():
        if name != "dev":
            groups.append((name, deps))

    items: list[SoupItem] = []
    seen: set[str] = set()
    for extra, deps in groups:
        for dep in deps:
            name = (
                dep.split("[")[0]
                .split(">=")[0]
                .split("==")[0]
                .split("<")[0]
                .split("~=")[0]
                .strip()
                .lower()
            )
            if name in seen:
                continue
            seen.add(name)
            items.append(
                SoupItem(
                    name=name,
                    version_spec=dep,
                    extra=extra,
                    purpose=PURPOSE.get(name, "SEM PROPÓSITO DECLARADO"),
                    safety_relevant=name in SAFETY_RELEVANT,
                )
            )
    return sorted(items, key=lambda i: (i.extra, i.name))


def installed_versions() -> dict[str, str]:
    """Versões efetivamente instaladas no ambiente corrente."""
    try:
        from importlib.metadata import distributions

        return {d.metadata["Name"].lower(): d.version for d in distributions()}
    except Exception:  # noqa: BLE001
        return {}


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "desconhecido"


def render(items: list[SoupItem]) -> str:
    versions = installed_versions()
    lines = [
        "---",
        "doc_id: REG-08",
        "title: Lista SOUP (Software of Unknown Provenance)",
        "status: GERADO AUTOMATICAMENTE — não editar à mão",
        "generated_by: scripts/soup.py",
        f"source_commit: {git_sha()}",
        "---",
        "",
        "# Lista SOUP",
        "",
        "IEC 62304 §5.3.3 e §8.1.2. **Documento gerado** — regenere com",
        "`python scripts/soup.py --write` e verifique no CI com `--check`.",
        "",
        "SOUP é todo software de terceiros incorporado ao produto cuja qualificação",
        "não foi feita por nós. Para cada item a norma exige identificação, versão,",
        "propósito no sistema e lista de anomalias publicadas.",
        "",
        "**Relevante para a segurança** marca os itens cuja falha pode contribuir",
        "para uma situação perigosa (ISO 14971). Esses exigem monitoramento ativo de",
        "anomalias publicadas (CVE/OSV) e avaliação de impacto a cada atualização.",
        "",
        "| Item | Especificação | Instalado | Extra | Segurança | Propósito no sistema |",
        "|---|---|---|---|---|---|",
    ]
    for item in items:
        installed = versions.get(item.name, "—")
        flag = "**sim**" if item.safety_relevant else "não"
        lines.append(
            f"| `{item.name}` | `{item.version_spec}` | {installed} | {item.extra} "
            f"| {flag} | {item.purpose} |"
        )

    lines += [
        "",
        "## Monitoramento de anomalias",
        "",
        "O workflow `.github/workflows/security.yml` executa `pip-audit` contra a",
        "base OSV a cada push e semanalmente. Uma vulnerabilidade em item marcado",
        "como relevante para a segurança abre avaliação de impacto obrigatória",
        "antes da próxima liberação (IEC 62304 §7.4).",
        "",
        "## Pendências",
        "",
        "- [ ] Registrar fabricante e URL do repositório de anomalias por item",
        "- [ ] Anexar avaliação de impacto para cada item relevante à segurança",
        "- [ ] Congelar o lockfile (`uv.lock`) como parte do registro de liberação",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="grava o documento")
    parser.add_argument("--check", action="store_true", help="falha se desatualizado")
    args = parser.parse_args()

    items = parse_dependencies()

    undeclared = [i.name for i in items if i.purpose == "SEM PROPÓSITO DECLARADO"]
    if undeclared:
        print(
            f"ERRO: dependências sem propósito declarado: {undeclared}\n"
            "Adicione cada uma ao dicionário PURPOSE em scripts/soup.py. "
            "IEC 62304 §8.1.2 exige propósito documentado para todo item SOUP.",
            file=sys.stderr,
        )
        return 1

    content = render(items)

    if args.check:
        if not OUTPUT.exists():
            print(f"ERRO: {OUTPUT.relative_to(REPO)} não existe. Rode --write.", file=sys.stderr)
            return 1
        current = OUTPUT.read_text(encoding="utf-8")

        # Ignora as linhas voláteis (commit e versões instaladas).
        def norm(text: str) -> list[str]:
            return [
                ln
                for ln in text.splitlines()
                if not ln.startswith("source_commit:") and "|" not in ln
            ]

        if norm(current) != norm(content):
            print(
                f"ERRO: {OUTPUT.relative_to(REPO)} está desatualizada.\n"
                "Rode: python scripts/soup.py --write",
                file=sys.stderr,
            )
            return 1
        print("SOUP list atualizada.")
        return 0

    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(content, encoding="utf-8")
        print(f"Gravado: {OUTPUT.relative_to(REPO)} ({len(items)} itens SOUP)")
        return 0

    print(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
