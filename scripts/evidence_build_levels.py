#!/usr/bin/env python3
"""Gera ``src/radiologyai/catalog/levels.json`` a partir de ``evidence/``.

O arquivo é função pura de ``evidence/measured`` (registros e ``plan.yaml``),
``evidence/published``, ``conditions.yaml``, ``protocol.yaml`` e dos manifestos de
modelo: mesmas entradas, mesmos bytes. Cada registro medido é re-julgado contra o
protocolo e o plano antes de contar (ver ``radiologyai.evidence.levels``).

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
        MANIFESTS_DIR,
        build_levels,
        check_levels,
        load_conditions,
        render_levels,
    )
    from radiologyai.evidence.plan import load_plan
    from radiologyai.evidence.store import EVIDENCE_ROOT, load_evidence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--catalog-dir", type=Path, default=CATALOG_DIR)
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=MANIFESTS_DIR,
        help="manifestos do registro de modelos (pesos distribuídos e visibilidade)",
    )
    parser.add_argument("--output", type=Path, help="padrão: <catalog-dir>/levels.json")
    args = parser.parse_args(argv)
    output = args.output or args.catalog_dir / "levels.json"

    try:
        protocol = load_protocol(args.catalog_dir)
        conditions = load_conditions(args.catalog_dir / "conditions.yaml")
        loaded = load_evidence(args.evidence_root)
        plan_path = args.evidence_root / "measured" / "plan.yaml"
        plan = load_plan(plan_path) if plan_path.is_file() else None
        records = [item.envelope or item.record for item in loaded]
        levels = build_levels(
            records, conditions, protocol, manifests_dir=args.manifests_dir, plan=plan
        )
    except RadiologyAIError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2

    if args.check:
        ok, message = check_levels(levels, output)
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1

    output.write_text(render_levels(levels), encoding="utf-8")
    # Contagem por nível só de modelos distribuídos: trio sem artefato no registro de
    # modelos não entra em indicador algum (REQ-185).
    counts: dict[str, int] = {}
    for entry in levels.entries:
        if entry.distributed:
            counts[entry.level.value] = counts.get(entry.level.value, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nenhum trio"
    hidden = sum(1 for e in levels.entries if not e.distributed)
    print(
        f"Gravado: {output} ({len(records)} registros; modelos distribuídos: {summary}; "
        f"{hidden} trios de modelo não distribuído, fora da contagem)"
    )
    for entry in levels.entries:
        if entry.basis or (entry.limited_by and entry.distributed):
            flag = "" if entry.distributed else " [não distribuído]"
            print(f"  {entry.level.value} {entry.condition_id} {entry.model_id}{flag}")
            for limit in entry.limited_by:
                print(f"      - {limit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
