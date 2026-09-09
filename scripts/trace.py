#!/usr/bin/env python3
"""Matriz de rastreabilidade RISCO -> REQUISITO -> TESTE (IEC 62304 §5.7).

O valor inteiro está no portão de CI: uma matriz de rastreabilidade que pode
divergir do código não vale nada; uma que bloqueia merge é evidência.

Quebra o build quando:
  - um requisito não tem teste que o verifique;
  - um teste referencia um REQ que não existe;
  - um requisito não declara controle de risco.

Uso:
    python scripts/trace.py            # imprime a matriz
    python scripts/trace.py --write    # grava docs/regulatory/13-traceability.md
    python scripts/trace.py --check    # falha se houver órfãos
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REQ_DIR = REPO / "docs" / "regulatory" / "06-requirements"
TESTS_DIR = REPO / "tests"
OUTPUT = REPO / "docs" / "regulatory" / "13-traceability.md"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.S)
REQ_ID_RE = re.compile(r"^REQ-\d{3}$")


@dataclass
class Requirement:
    id: str
    title: str
    type: str
    risk_controls: list[str]
    status: str
    tests: list[str] = field(default_factory=list)


def load_requirements() -> dict[str, Requirement]:
    import yaml

    reqs: dict[str, Requirement] = {}
    if not REQ_DIR.is_dir():
        return reqs
    for path in sorted(REQ_DIR.glob("REQ-*.md")):
        m = FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
        if not m:
            raise SystemExit(f"ERRO: {path.name} não tem frontmatter YAML")
        meta = yaml.safe_load(m.group(1))
        reqs[meta["id"]] = Requirement(
            id=meta["id"],
            title=meta.get("title", ""),
            type=meta.get("type", "?"),
            risk_controls=list(meta.get("risk_controls") or []),
            status=meta.get("status", "?"),
        )
    return reqs


def collect_test_markers() -> dict[str, list[str]]:
    """Extrai `@pytest.mark.requirement("REQ-xxx")` por AST, sem executar testes."""
    found: dict[str, list[str]] = {}

    def record(req_id: str, location: str) -> None:
        found.setdefault(req_id, []).append(location)

    def marker_ids(node: ast.AST) -> list[str]:
        ids: list[str] = []
        for dec in getattr(node, "decorator_list", []):
            call = dec if isinstance(dec, ast.Call) else None
            target = call.func if call else dec
            parts: list[str] = []
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if parts[:1] != ["requirement"]:
                continue
            for arg in (call.args if call else []):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    ids.append(arg.value)
        return ids

    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for req_id in marker_ids(node):
                record(req_id, f"{rel}::{node.name}")
    return found


def build() -> tuple[dict[str, Requirement], list[str]]:
    reqs = load_requirements()
    markers = collect_test_markers()

    problems: list[str] = []

    for req_id, locations in markers.items():
        if not REQ_ID_RE.match(req_id):
            problems.append(f"marcador com id malformado: {req_id!r} em {locations[0]}")
        elif req_id not in reqs:
            problems.append(
                f"teste referencia requisito inexistente {req_id} ({locations[0]})"
            )
        else:
            reqs[req_id].tests.extend(locations)

    for req in reqs.values():
        if not req.tests:
            problems.append(f"{req.id} não tem teste verificador (IEC 62304 §5.7)")
        if not req.risk_controls:
            problems.append(f"{req.id} não declara controle de risco (ISO 14971)")

    return reqs, problems


def render(reqs: dict[str, Requirement]) -> str:
    lines = [
        "---",
        "doc_id: REG-13",
        "title: Matriz de Rastreabilidade",
        "status: GERADO AUTOMATICAMENTE — não editar à mão",
        "generated_by: scripts/trace.py",
        "---",
        "",
        "# Matriz de rastreabilidade",
        "",
        "IEC 62304 §5.7 — cada requisito é verificado por ao menos um teste, e cada",
        "requisito rastreia até um controle de risco (ISO 14971).",
        "",
        "Gerada por `scripts/trace.py --write` e **verificada no CI** com `--check`.",
        "O build falha se um requisito ficar sem teste ou se um teste citar um",
        "requisito inexistente.",
        "",
        "| Requisito | Tipo | Controle de risco | Estado | Verificado por |",
        "|---|---|---|---|---|",
    ]
    for req in sorted(reqs.values(), key=lambda r: r.id):
        tests = "<br>".join(f"`{t.split('::')[-1]}`" for t in sorted(set(req.tests)))
        risks = ", ".join(req.risk_controls)
        lines.append(
            f"| **{req.id}** {req.title} | {req.type} | {risks} | {req.status} | {tests} |"
        )

    total_tests = sum(len(set(r.tests)) for r in reqs.values())
    lines += [
        "",
        f"**{len(reqs)} requisitos · {total_tests} verificações.**",
        "",
        "> Os controles de risco (RISK-xxx) serão detalhados em",
        "> `04-risk-file.md`. As sementes H-01…H-10 estão em `ROADMAP.md` §6.3.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    reqs, problems = build()

    if problems:
        print(f"Rastreabilidade incompleta ({len(problems)} problema(s)):\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    if args.check:
        if not OUTPUT.exists():
            print(f"ERRO: {OUTPUT.relative_to(REPO)} não existe. Rode --write.", file=sys.stderr)
            return 1
        if OUTPUT.read_text(encoding="utf-8") != render(reqs):
            print(
                f"ERRO: {OUTPUT.relative_to(REPO)} desatualizada. "
                "Rode: python scripts/trace.py --write",
                file=sys.stderr,
            )
            return 1
        print(f"Rastreabilidade completa: {len(reqs)} requisitos, todos verificados.")
        return 0

    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(render(reqs), encoding="utf-8")
        print(f"Gravado: {OUTPUT.relative_to(REPO)} ({len(reqs)} requisitos)")
        return 0

    print(render(reqs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
