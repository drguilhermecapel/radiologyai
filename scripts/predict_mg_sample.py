#!/usr/bin/env python3
"""Sanidade em exames reais de mamografia (CMMD amostra de 8 pacientes).

Dois modos:

``pipeline`` (padrão)
    Cada paciente vira um ``ExamInputs`` (todas as incidências carregadas com
    :func:`load_mammogram`) e passa por :func:`radiologyai.analysis.pipeline.analyze`
    com ``mammoscreen`` (+ ``mammo-crop`` como dependência) e
    ``monai-breast-density``. Imprime e grava em JSON os achados, as execuções
    e a proveniência. É sanidade de execução, não medição de desempenho.

``parity``
    Paridade do código vendorizado com o código remoto ORIGINAL do cartão:
    roda os adaptadores com a conversão 8 bits SEM VOI (a mesma entrada usada
    pela execução de referência em ``spikes/mammo/out/cmmd8_reference_vs_vendored.json``,
    produzida com o ``modeling.py`` original + albumentations 2.0.8 + cv2) e
    compara caixas de recorte, escore de câncer e densidade.

Rode sob a trava::

    /root/medai-cache/bin/heavy /root/medai-cache/bin/timev \\
        .venv/bin/python scripts/predict_mg_sample.py pipeline
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

DATA = Path("/root/medai-cache/datasets/cmmd-sample")
WEIGHTS = Path("/root/medai-cache/weights/registry")
REFERENCE = Path("/root/medai-cache/spikes/mammo/out/cmmd8_reference_vs_vendored.json")
OUT = Path("/root/medai-cache/eval-preds/mammoscreen/cmmd/sample8_pipeline.json")


def _patients(data: Path) -> list[Path]:
    return sorted(p for p in data.iterdir() if p.is_dir() and p.name.startswith("D"))


def cmd_pipeline(args: argparse.Namespace) -> int:
    from radiologyai.analysis.artifacts import ArtifactStore
    from radiologyai.analysis.pipeline import analyze
    from radiologyai.analysis.registry import Registry
    from radiologyai.analysis.types import ExamInputs
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.catalog.schema import ExamType
    from radiologyai.modalities.mg.loader import load_mammogram

    registry = Registry(weights_dir=args.weights_dir)
    catalog = Catalog.load()
    report: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(Path(tmp))
        for folder in _patients(args.data):
            images = tuple(load_mammogram(f) for f in sorted(folder.glob("*.dcm")))
            exam = ExamInputs(images=images, exam_type=ExamType.MG)
            t0 = time.monotonic()
            result = analyze(
                exam,
                registry=registry,
                catalog=catalog,
                store=store,
                model_ids=["mammoscreen", "monai-breast-density"],
                include_research=True,
            )
            seconds = time.monotonic() - t0
            rec = {
                "patient": folder.name,
                "views": [f"{im.meta.laterality}-{im.meta.view_position}" for im in images],
                "status": result.status,
                "seconds": round(seconds, 2),
                "runs": [
                    {"model": r.model_id, "status": r.status, "error": r.error} for r in result.runs
                ],
                "findings": [
                    {
                        "label": f.label,
                        "condition": f.condition_id,
                        "raw_score": f.raw_score,
                        "decision": f.decision,
                        "category": f.category,
                        "class_scores": f.class_scores,
                        "level": f.evidence_level.value,
                        "visibility": f.visibility,
                    }
                    for f in result.findings
                ],
                "warnings": list(result.warnings),
            }
            report.append(rec)
            summary = ", ".join(
                f"{f.label}={f.category or round(float(f.raw_score or 0.0), 5)}"
                for f in result.findings
            )
            print(f"{folder.name} {rec['status']} {seconds:.1f}s :: {summary}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"-> {args.out}")
    return 0 if all(r["status"] == "completed" for r in report) else 1


def cmd_parity(args: argparse.Namespace) -> int:
    from radiologyai.analysis.base import ModelInputs
    from radiologyai.analysis.registry import Registry
    from radiologyai.catalog.schema import ExamType
    from radiologyai.modalities.mg.loader import load_mammogram
    from radiologyai.modalities.mg.voi import VOI_EXTRA_KEY

    ref = {(r["patient"], r["laterality"]): r for r in json.loads(args.reference.read_text())}
    registry = Registry(weights_dir=args.weights_dir)
    crop = registry.load("mammo-crop")
    screen = registry.load("mammoscreen")
    worst = {"cancer": 0.0, "density": 0.0, "box": 0}
    for folder in _patients(args.data):
        images = []
        for f in sorted(folder.glob("*.dcm")):
            image = load_mammogram(f)
            voi = dict(image.extra[VOI_EXTRA_KEY])
            voi.update(window_center=[], window_width=[], function=None, lut=None)
            images.append(dataclasses.replace(image, extra={**image.extra, VOI_EXTRA_KEY: voi}))
        for lat in ("L", "R"):
            sel = tuple(im for im in images if im.meta.laterality == lat)
            if (folder.name, lat) not in ref or not sel:
                continue
            roles = {f"{im.meta.laterality}-{im.meta.view_position}": i for i, im in enumerate(sel)}
            inputs = ModelInputs(images=sel, exam_type=ExamType.MG, roles=roles)
            crop_out = crop.predict(inputs)
            inputs = ModelInputs(
                images=sel, exam_type=ExamType.MG, roles=roles, upstream={"mammo-crop": crop_out}
            )
            out = screen.predict(inputs)
            side = "left" if lat == "L" else "right"
            cancer = next(s.value for s in out.scores if s.label == f"malignancy_{side}")
            density = list(out.class_scores[f"density_{side}"].values())
            r = ref[(folder.name, lat)]
            boxes = [crop_out.extras["crop_boxes_by_role"][k] for k in sorted(roles)]
            d_c = abs(cancer - r["cancer_ref"])
            d_d = max(abs(a - b) for a, b in zip(density, r["density_ref"], strict=True))
            d_b = max(
                abs(a - b)
                for ba, bb in zip(boxes, r["crop_ref"], strict=True)
                for a, b in zip(ba, bb, strict=True)
            )
            worst = {
                "cancer": max(worst["cancer"], d_c),
                "density": max(worst["density"], d_d),
                "box": max(int(worst["box"]), d_b),
            }
            print(
                f"{folder.name} {lat} cancer={cancer:.6f} ref={r['cancer_ref']:.6f} "
                f"|d|={d_c:.2e} dens|d|={d_d:.2e} box|d|={d_b}",
                flush=True,
            )
    print(f"pior diferença: {worst}")
    return 0 if worst["cancer"] < 1e-4 and worst["box"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd")
    pp = sub.add_parser("pipeline")
    pp.add_argument("--data", type=Path, default=DATA)
    pp.add_argument("--weights-dir", type=Path, default=WEIGHTS)
    pp.add_argument("--out", type=Path, default=OUT)
    pa = sub.add_parser("parity")
    pa.add_argument("--data", type=Path, default=DATA)
    pa.add_argument("--weights-dir", type=Path, default=WEIGHTS)
    pa.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args(argv)
    if args.cmd == "parity":
        return cmd_parity(args)
    if args.cmd is None:
        args = parser.parse_args(["pipeline", *(argv or sys.argv[1:])])
    return cmd_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
