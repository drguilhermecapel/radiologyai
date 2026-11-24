#!/usr/bin/env python3
"""Guardião de honestidade — verificação de CI.

O problema do repositório v1 nunca foi falta de competência técnica. Foi que
*nada no sistema jamais objetou* quando um número foi inventado. Este script
torna a objeção automática.

Quebra o build em qualquer um destes:

1. Alegação numérica de desempenho em Markdown fora de ``legacy/`` que não
   esteja num bloco gerado a partir de ``artifacts/eval/``.
2. ``np.random`` / ``random.`` fora dos locais permitidos.
3. Campo sha256 em model card que não seja 64 hex minúsculos.
4. ``except ImportError`` em ``src/`` (detectado por AST, não por texto).
5. Marcadores de simulação em ``src/``.
6. Model card com desempenho alegado mas sem artefato de avaliação vinculado.

Uso:
    python scripts/check_honesty.py            # verifica o repositório
    python scripts/check_honesty.py --quiet    # só o resultado
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "radiologyai"

EXCLUDED_DIRS = {"legacy", ".git", ".venv", "artifacts", "node_modules", "__pycache__"}

# Documentos que existem exatamente para retratar números fabricados.
HONESTY_DOCS = {"HONEST_STATUS.md", "ROADMAP.md"}

# 1. Alegação de desempenho
PERF_PATTERNS = (
    re.compile(
        r"\b\d{2,3}(?:[.,]\d+)?\s*%\s*(?:de\s+)?"
        r"(?:acur[áa]cia|precis[ãa]o|sensibilidade|especificidade)",
        re.I,
    ),
    re.compile(r"\b(?:acur[áa]cia|sensibilidade|especificidade)\s*(?:validada)?\s*[:=]\s*\d", re.I),
    re.compile(r"\bAUC\s*[:=]\s*0?\.\d+", re.I),
    re.compile(r"\bvalidad[oa]s?\s+clinicamente\b", re.I),
)
# Um bloco só pode alegar número se citar sua proveniência.
PROVENANCE = re.compile(
    r"artifacts/eval/|reports/EVALUATION_|METAS|RETRATAD|fabricad|"
    r"N[ÃA]O MEDIDA|sint[ée]tic|HONEST_STATUS|legacy/|relatório da v1",
    re.I,
)

# 2. Aleatoriedade
RANDOM_RE = re.compile(
    r"\bnp\.random\b|\bnumpy\.random\b|"
    r"\brandom\.(?:random|randint|choice|uniform|normal)\b"
)
RANDOM_ALLOWED = {"evaluation/metrics.py"}

# 3. sha256
SHA_FIELD_RE = re.compile(
    # Aceita YAML (chave nua) e JSON (chave entre aspas). O model_registry.json
    # legado usava a forma JSON — sem o ["\']? antes do ":" ela escapava.
    r"(?:weights_)?sha256(?:_hash)?[\"\']?\s*:\s*[\"\']?([^\"\'\s,}]+)",
    re.I,
)
SHA_VALID_RE = re.compile(r"^[0-9a-f]{64}$")

# 5. Marcadores de simulação
SIMULATION_MARKERS = (
    "_simulate_",
    "_fallback_analysis",
    "_analyze_image_fallback",
    "_create_dummy_model",
    "Mock ground truth",
    "# TODO: real",
)


def code_lines(path: Path) -> dict[int, str]:
    """Linhas de um .py contendo apenas código — sem strings nem comentários.

    Necessário porque este pacote *descreve* os anti-padrões do legado em
    docstrings. Uma docstring que cita ``_analyze_image_fallback`` para explicar
    o que foi eliminado não é uma ocorrência dele.
    """
    source = path.read_text(encoding="utf-8")
    lines = dict(enumerate(source.splitlines(), 1))
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return lines

    blanked: dict[int, list[str]] = {n: list(text) for n, text in lines.items()}
    for tok in tokens:
        if tok.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        for row in range(srow, erow + 1):
            if row not in blanked:
                continue
            chars = blanked[row]
            start = scol if row == srow else 0
            end = ecol if row == erow else len(chars)
            for idx in range(start, min(end, len(chars))):
                chars[idx] = " "
    return {n: "".join(chars) for n, chars in blanked.items()}


@dataclass
class Violation:
    rule: str
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"  [{self.rule}] {self.path}:{self.line}\n      {self.detail}"


def walk(root: Path, suffix: str) -> list[Path]:
    return [
        p for p in root.rglob(f"*{suffix}") if not (EXCLUDED_DIRS & set(p.relative_to(root).parts))
    ]


def rule_1_performance_claims() -> list[Violation]:
    out: list[Violation] = []
    for path in walk(REPO, ".md"):
        rel = path.relative_to(REPO).as_posix()
        if path.name in HONESTY_DOCS:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern in PERF_PATTERNS:
                if pattern.search(line) and not PROVENANCE.search(line):
                    out.append(
                        Violation(
                            "R1-desempenho",
                            rel,
                            i,
                            f"alegação sem proveniência: {line.strip()[:90]}",
                        )
                    )
                    break
    return out


def rule_2_randomness() -> list[Violation]:
    out: list[Violation] = []
    if not SRC.is_dir():
        return out
    for path in walk(SRC, ".py"):
        rel_src = path.relative_to(SRC).as_posix()
        if rel_src in RANDOM_ALLOWED:
            continue
        for i, line in code_lines(path).items():
            if RANDOM_RE.search(line):
                out.append(
                    Violation(
                        "R2-aleatoriedade",
                        path.relative_to(REPO).as_posix(),
                        i,
                        f"aleatoriedade fora da avaliação: {line.strip()[:90]}",
                    )
                )
    return out


def rule_3_sha256() -> list[Violation]:
    out: list[Violation] = []
    for suffix in (".yaml", ".yml", ".json"):
        for path in walk(REPO, suffix):
            rel = path.relative_to(REPO).as_posix()
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                m = SHA_FIELD_RE.search(line)
                if m and not SHA_VALID_RE.match(m.group(1).strip().strip('",')):
                    out.append(
                        Violation(
                            "R3-sha256",
                            rel,
                            i,
                            f"não é sha256 hex de 64 chars: {m.group(1)[:70]}",
                        )
                    )
    return out


def rule_4_silent_imports() -> list[Violation]:
    out: list[Violation] = []
    if not SRC.is_dir():
        return out
    for path in walk(SRC, ".py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            names = (
                [node.type]
                if isinstance(node.type, ast.Name)
                else list(getattr(node.type, "elts", []))
            )
            if any(getattr(n, "id", None) in {"ImportError", "ModuleNotFoundError"} for n in names):
                out.append(
                    Violation(
                        "R4-import-silencioso",
                        path.relative_to(REPO).as_posix(),
                        node.lineno,
                        "except ImportError degrada em silêncio; deixe falhar alto",
                    )
                )
    return out


def rule_5_simulation_markers() -> list[Violation]:
    out: list[Violation] = []
    if not SRC.is_dir():
        return out
    for path in walk(SRC, ".py"):
        for i, line in code_lines(path).items():
            for marker in SIMULATION_MARKERS:
                if marker in line:
                    out.append(
                        Violation(
                            "R5-simulacao",
                            path.relative_to(REPO).as_posix(),
                            i,
                            f"marcador de simulação {marker!r} em código de produção",
                        )
                    )
    return out


def rule_6_unbacked_cards() -> list[Violation]:
    out: list[Violation] = []
    cards = SRC / "models" / "cards"
    if not cards.is_dir():
        return out
    import yaml

    for path in sorted(cards.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        forbidden = {"accuracy", "auc", "sensitivity", "specificity"} & set(data)
        if forbidden:
            out.append(
                Violation(
                    "R6-card-sem-lastro",
                    path.relative_to(REPO).as_posix(),
                    1,
                    f"campos de desempenho no card: {sorted(forbidden)}. "
                    "Métrica vive em artifacts/eval/, não no manifesto.",
                )
            )
    return out


RULES = (
    ("R1 alegação de desempenho sem proveniência", rule_1_performance_claims),
    ("R2 aleatoriedade fora da avaliação", rule_2_randomness),
    ("R3 sha256 inválido", rule_3_sha256),
    ("R4 import silencioso", rule_4_silent_imports),
    ("R5 marcador de simulação", rule_5_simulation_markers),
    ("R6 model card sem lastro", rule_6_unbacked_cards),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    all_violations: list[Violation] = []
    for label, rule in RULES:
        found = rule()
        all_violations.extend(found)
        if not args.quiet:
            mark = "FALHOU" if found else "ok"
            print(f"{mark:>7}  {label}" + (f" ({len(found)})" if found else ""))

    if all_violations:
        print(f"\n{len(all_violations)} violação(ões):\n")
        for v in all_violations:
            print(v)
        print(
            "\nEste verificador existe porque, no repositório v1, nada objetou "
            "quando um número foi inventado. Ver HONEST_STATUS.md."
        )
        return 1

    if not args.quiet:
        print("\nNenhuma violação.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
