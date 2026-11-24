#!/usr/bin/env python
"""Roda o PIPELINE REAL (``analyze``) de TC/RM em amostras públicas e registra tempo/RSS.

Uma amostra por invocação, para tomar a trava pesada por exame::

    /root/medai-cache/bin/heavy /root/medai-cache/bin/timev \\
        .venv/bin/python -I scripts/predict_ct_samples.py --sample lidc0078

Grava ``/root/medai-cache/eval-preds/ct-samples/<amostra>.json`` com o estado de
cada modelo (concluído/abstenção/falha e motivo), achados, medidas e tempo. Não
é avaliação (sem padrão de referência): é a prova de execução real ponta a ponta.

Fatos de aquisição não presentes no arquivo NÃO são inventados: a fase de
contraste só é passada quando o próprio conjunto de dados a documenta
(``contrast_phase`` abaixo, com a fonte).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

D = Path("/root/medai-cache/datasets")
OUT = Path("/root/medai-cache/eval-preds/ct-samples")
WEIGHTS = Path("/root/medai-cache/weights/registry")

SAMPLES: dict[str, dict[str, Any]] = {
    "totalseg-ct": {
        "path": D / "samples-ct/totalseg-tests/example_ct_sm.nii.gz",
        "modality": "CT",
        "exam_type": "ct_abdomen",
        "value_range": (-1024.0, 3071.0),
    },
    "totalseg-mr": {
        "path": D / "samples-ct/totalseg-tests/example_mr_sm.nii.gz",
        "modality": "MR",
        "exam_type": "mr_body",
    },
    "lidc0078": {
        "path": next(iter((D / "samples-ct/tcia/LIDC-IDRI-0078").iterdir()), None),
        "exam_type": "ct_chest",
    },
    "hcc005-recon2": {
        "path": D / "samples-ct/tcia/HCC_005/"
        "1.3.6.1.4.1.14519.5.2.1.1706.8374.237142072858504762657736089106",
        "exam_type": "ct_abdomen",
    },
    "cq500-427": {
        "path": D / "cq500/dicom/CQ500-CT-427/CQ500CT427 CQ500CT427/Unknown Study/CT 2.55mm",
        "exam_type": "ct_head",
        # CQ500: "non-contrast head CT scans" (descrição do conjunto, qure.ai) — fato documentado
        "meta": {"contrast_phase": "none"},
    },
}


def run(name: str) -> dict[str, Any]:
    from radiologyai.analysis.artifacts import ArtifactStore
    from radiologyai.analysis.imaging import load_nifti, load_series
    from radiologyai.analysis.pipeline import analyze
    from radiologyai.analysis.registry import Registry
    from radiologyai.analysis.types import ExamInputs, Modality
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.catalog.schema import ExamType
    from radiologyai.config.settings import Settings

    spec = SAMPLES[name]
    path = Path(spec["path"])
    t0 = time.monotonic()
    if path.is_dir():
        image = load_series(path, **spec.get("meta", {}))
    else:
        image = load_nifti(
            path,
            modality=Modality(spec["modality"]),
            value_range=spec.get("value_range"),
            **spec.get("meta", {}),
        )
    t_load = time.monotonic() - t0
    settings = Settings(weights_dir=WEIGHTS, device="cpu", max_loaded_models=2)
    registry = Registry(weights_dir=WEIGHTS, settings=settings)
    t1 = time.monotonic()
    result = analyze(
        ExamInputs(images=(image,), exam_type=ExamType(spec["exam_type"])),
        registry=registry,
        catalog=Catalog.load(),
        store=ArtifactStore(OUT / "store"),
        include_research=True,
    )
    registry.unload_all()
    return {
        "sample": name,
        "input": str(path),
        "shape_zyx": list(image.pixels.shape),
        "spacing_zyx_mm": list(image.spacing_mm or ()),
        "acquisition": {
            "kvp": image.meta.kvp,
            "kernel": image.meta.convolution_kernel,
            "contrast_agent": image.meta.contrast_agent,
            "contrast_phase": image.meta.contrast_phase,
            "slice_thickness_mm": image.meta.slice_thickness_mm,
            "age": image.meta.patient_age_years,
        },
        "seconds_load": round(t_load, 1),
        "seconds_analyze": round(time.monotonic() - t1, 1),
        "peak_rss_gb_parent": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
        "peak_rss_gb_children": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 3
        ),
        "status": result.status,
        "runs": [
            {
                "model": r.model_id,
                "status": r.status,
                "seconds": None if r.seconds is None else round(r.seconds, 1),
                "error": r.error,
                "reasons": [x.value for x in r.qa.reasons] if r.qa else [],
            }
            for r in result.runs
        ],
        "measurements": [
            {
                "name": m.name,
                "value": m.value,
                "unit": m.unit,
                "abstention": m.abstention_reason.value if m.abstention_reason else None,
                "note": m.note,
            }
            for m in result.measurements
        ],
        "findings": [
            {
                "condition": f.condition_id,
                "label": f.label,
                "decision": f.decision,
                "raw_score": f.raw_score,
                "level": f.evidence_level.value,
                "abstention": f.abstention_reason.value if f.abstention_reason else None,
            }
            for f in result.findings
        ],
        "warnings": list(result.warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=sorted(SAMPLES), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = run(args.sample)
    target = OUT / f"{args.sample}.json"
    target.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sample", "status", "seconds_analyze")}))
    for r in report["runs"]:
        print(
            f"  {r['model']:40s} {r['status']:10s} {r['seconds']} {r['reasons']} "
            f"{(r['error'] or '')[:200]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
