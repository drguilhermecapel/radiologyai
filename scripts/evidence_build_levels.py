#!/usr/bin/env python3
"""Gera ``src/radiologyai/catalog/levels.json`` a partir de ``evidence/``.

O arquivo é função pura de ``evidence/measured``, ``evidence/published``,
``conditions.yaml`` e ``protocol.yaml``: mesmas entradas, mesmos bytes.

Uso:
    python scripts/evidence_build_levels.py           # (re)grava levels.json
    python scripts/evidence_build_levels.py --check   # CI: falha se desatualizado
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from radiologyai.catalog import load_protocol
    from radiologyai.errors import RadiologyAIError
    from radiologyai.evidence.levels import (
        CATALOG_DIR,
        build_levels,
        check_levels,
        load_conditions,
        render_levels,
    )
    from radiologyai.evidence.store import EVIDENCE_ROOT, load_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--catalog-dir", type=Path, default=CATALOG_DIR)
    parser.add_argument("--output", type=Path, help="padrão: <catalog-dir>/levels.json")
    args = parser.parse_args(argv)
    output = args.output or args.catalog_dir / "levels.json"

    try:
        protocol = load_protocol(args.catalog_dir)
        conditions = load_conditions(args.catalog_dir / "conditions.yaml")
        records = load_records(args.evidence_root)
        levels = build_levels(records, conditions, protocol)
    except RadiologyAIError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2

    if args.check:
        ok, message = check_levels(levels, output)
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1

    output.write_text(render_levels(levels), encoding="utf-8")
    counts: dict[str, int] = {}
    for entry in levels.entries:
        counts[entry.level.value] = counts.get(entry.level.value, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nenhum trio"
    print(f"Gravado: {output} ({len(records)} registros; {summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
