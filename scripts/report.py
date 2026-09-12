#!/usr/bin/env python3
"""Gera reports/EVALUATION_<...>.md a partir de um artifacts/eval/<run>/metrics.json.

O relatório é DERIVADO do artefato — nunca editado à mão. Todo número nele
aponta para o run que o produziu, o que satisfaz o guardião de honestidade.

Uso:
    python scripts/report.py artifacts/eval/<run_id>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def milhar(n: int) -> str:
    """25596 -> '25.596'. Aplicado ao campo, nunca à linha: uma substituição global
    de vírgula por ponto corrompia o intervalo de confiança ('[0.783, 0.808]')."""
    return f"{n:,}".replace(",", ".")


def render(m: dict, rel_artifact: str) -> str:
    env, ds, model, cfg = m["environment"], m["dataset"], m["model"], m["config"]
    lines = [
        f"# Avaliação — {model['card_id']} em {ds['name']}",
        "",
        f"> **Gerado por `scripts/report.py` a partir de `{rel_artifact}`.** Não editar à mão.",
        "> Todo número abaixo é reproduzível a partir desse artefato (artifacts/eval/).",
        "",
        "## Proveniência",
        "",
        "| Campo | Valor |",
        "|---|---|",
        f"| run_id | `{m['run_id']}` |",
        f"| Executado em | {m['timestamp_utc'][:19]} UTC |",
        f"| Código (git_sha) | `{env['git_sha']}` |",
        f"| Modelo | {model['card_id']} v{model['version']} "
        f"— pesos `{model['weights_sha256'][:16]}…` |",
        f"| Treinado em | {', '.join(model['trained_on'])} |",
        f"| Dataset | {ds['name']}, split {ds['split']} "
        f"— manifest `{ds['manifest_sha256'][:16]}…` |",
        f"| Imagens / pacientes | {milhar(ds['n_images'])} / {milhar(ds['n_patients'])} |",
        f"| Externo ao treino | **{'sim' if ds['external_to_training_data'] else 'NÃO'}** "
        f"({ds['leakage_status']}) |",
        f"| Python / torch / xrv | {env['python']} / {env['libraries'].get('torch')} "
        f"/ {env['libraries'].get('torchxrayvision')} |",
        f"| Seed / bootstrap | {cfg['seed']} / {cfg['n_bootstrap']} |",
        "",
        "## Resultado",
        "",
        f"**AUROC macro: {m['macro_auroc']:.3f}** sobre {m['n_labels_evaluated']} rótulos "
        f"(fonte: `{rel_artifact}`).",
        "",
        "| Achado | AUROC | IC 95% | n+ | Prevalência | Esp. @ sens. 90% | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, e in sorted(m["per_label"].items(), key=lambda kv: -kv[1]["auroc"]):
        lo, hi = e["auroc_ci95"]
        op = e.get("operating_point") or {}
        spec = f"{op['specificity']:.3f}" if "specificity" in op else "—"
        ece = f"{e['ece']:.3f}" if "ece" in e else "—"
        lines.append(
            f"| {name} | {e['auroc']:.3f} | [{lo:.3f}, {hi:.3f}] | {milhar(e['n_pos'])} | "
            f"{e['prevalence']:.3f} | {spec} | {ece} |"
        )
    lines += [
        "",
        "## Subgrupos (AUROC macro)",
        "",
        "| Grupo | Valor | n | AUROC macro |",
        "|---|---|---:|---:|",
    ]
    for group, values in m["subgroups"].items():
        for k, v in values.items():
            val = f"{v['macro_auroc']:.3f}" if v.get("macro_auroc") is not None else "—"
            lines.append(f"| {group} | {k} | {milhar(v['n'])} | {val} |")
    lines += ["", "## Não avaliados", ""]
    for x in m["not_evaluated"]:
        lines.append(f"- `{x['label']}` — {x['reason']}")
    lines += ["", "## Limitações declaradas", ""]
    for limitation in m["limitations"]:
        lines.append(f"- {limitation}")
    lines += [
        "",
        "---",
        "",
        "Este relatório mede o desempenho isolado de um algoritmo de terceiros, "
        "retrospectivamente, sobre rótulos minerados por NLP. **Não é validação clínica** "
        "e não sustenta nenhuma alegação de uso clínico. Ver `HONEST_STATUS.md`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        print(f"ERRO: {metrics_path} não existe", file=sys.stderr)
        return 1
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    rel = metrics_path.relative_to(REPO).as_posix()
    date = m["timestamp_utc"][:10]
    out = (
        REPO
        / "reports"
        / f"EVALUATION_{slug(m['model']['card_id'])}_{slug(m['dataset']['name'])}_{date}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(m, rel), encoding="utf-8")
    print(f"Gravado: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
