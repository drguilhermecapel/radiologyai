#!/usr/bin/env python
"""CQ500: seleção pré-especificada, download e predição do TotalSegmentator ``cerebral_bleed``.

Subcomandos::

    predict_ct_cq500.py select     # grava selection.csv (ordem determinística)
    predict_ct_cq500.py download   # baixa, extrai só a série escolhida, apaga o zip
    predict_ct_cq500.py predict    # roda o adaptador REAL (subprocesso) por exame

Regras PRÉ-ESPECIFICADAS (escritas antes de qualquer predição desta rodada):

* Padrão de referência: voto majoritário (>= 2 de 3 leitores) de ``R*:ICH`` em
  ``reads.csv``; negativo = 0 de 3 leitores (exames com 1 voto ficam de fora,
  como no subamostra ``cq500_eval80`` do *spike*).
* Ordem: sha256("medai-cq500-v1" + nome) — sem gerador aleatório; 40 positivos e
  40 negativos, intercalados (pos, neg, pos, ...), de modo que qualquer prefixo
  é balanceado.
* Série: entre as séries axiais do estudo com >= 20 cortes e sem "BONE" na
  descrição, a de espessura mais próxima de 5 mm (empate: mais cortes).
* Achado derivado: volume segmentado total >= 0,1 mL (ponto de operação do
  manifesto ``totalseg-cerebral-bleed``); a AUROC usa o volume contínuo.

A predição roda o adaptador do produto (``ct_totalseg``) pelo registro, com o
carregador de série do produto (``load_series``). Cada exame toma a trava
pesada separadamente (use ``/root/medai-cache/bin/heavy`` por exame com
``--one NAME``). Grava ``/root/medai-cache/eval-preds/totalseg-cerebral_bleed/cq500/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

DATA = Path("/root/medai-cache/datasets/cq500")
OUT = Path("/root/medai-cache/eval-preds/totalseg-cerebral_bleed/cq500")
SALT = "medai-cq500-v1"
N_POS = 40
N_NEG = 40
URL = "https://s3.ap-south-1.amazonaws.com/qure.headct.study/{name}.zip"
LABELS = ("ICH", "IPH", "IVH", "SDH", "EDH", "SAH", "MassEffect", "MidlineShift")
TARGET_THICKNESS_MM = 5.0


def _rows() -> list[dict[str, str]]:
    with (DATA / "reads.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select() -> list[dict[str, Any]]:
    """Seleção determinística (sem RNG): 40 positivos e 40 negativos intercalados."""
    rows = []
    for row in _rows():
        votes = {lab: sum(int(row[f"R{i}:{lab}"]) for i in (1, 2, 3)) for lab in LABELS}
        rows.append(
            {
                "name": row["name"],
                "category": row["Category"],
                "votes_ich": votes["ICH"],
                **{f"maj_{lab}": int(v >= 2) for lab, v in votes.items()},
                "order": hashlib.sha256((SALT + row["name"]).encode()).hexdigest(),
            }
        )
    rows.sort(key=lambda r: r["order"])
    pos = [r for r in rows if r["maj_ICH"] == 1][:N_POS]
    neg = [r for r in rows if r["votes_ich"] == 0][:N_NEG]
    out: list[dict[str, Any]] = []
    for a, b in zip(pos, neg, strict=True):
        out.extend((a, b))
    return out


def write_selection() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select()
    path = OUT / "selection.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sel[0]))
        writer.writeheader()
        writer.writerows(sel)
    return path


def _series_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("*.dcm")})


def choose_series(study_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Regra pré-especificada de escolha da série (ver docstring do módulo)."""
    import pydicom

    candidates: list[tuple[float, int, Path, dict[str, Any]]] = []
    for sdir in _series_dirs(study_dir):
        files = sorted(sdir.glob("*.dcm"))
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        desc = str(getattr(ds, "SeriesDescription", "") or sdir.name)
        kernel = str(getattr(ds, "ConvolutionKernel", "") or "")
        try:
            thick = float(ds.SliceThickness)
        except (AttributeError, TypeError, ValueError):
            continue
        facts = {
            "series_dir": sdir.name,
            "description": desc,
            "kernel": kernel,
            "slice_thickness_mm": thick,
            "n_files": len(files),
            "gantry_tilt_deg": str(getattr(ds, "GantryDetectorTilt", "")),
        }
        if len(files) < 20 or "BONE" in desc.upper() or "BONE" in kernel.upper():
            continue
        candidates.append((abs(thick - TARGET_THICKNESS_MM), -len(files), sdir, facts))
    if not candidates:
        raise RuntimeError(f"nenhuma série elegível em {study_dir}")
    candidates.sort(key=lambda c: (c[0], c[1], str(c[2])))
    return candidates[0][2], candidates[0][3]


def download(limit: int | None) -> None:
    """Baixa cada exame selecionado, extrai, mantém só a série escolhida e apaga o zip."""
    sel = select()[: limit or None]
    zips = DATA / "zips"
    dicom = DATA / "dicom"
    zips.mkdir(parents=True, exist_ok=True)
    for row in sel:
        name = row["name"]
        dest = dicom / name
        marker = dest / ".medai-series.json"
        if marker.is_file():
            continue
        zpath = zips / f"{name}.zip"
        if not zpath.is_file():
            tmp = zpath.with_suffix(".part")
            subprocess.run(
                ["curl", "-sS", "--fail", "--retry", "3", "-o", str(tmp), URL.format(name=name)],
                check=True,
            )
            tmp.replace(zpath)
        staging = dicom / f".{name}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                target = (staging / info.filename).resolve()
                if staging.resolve() not in target.parents:
                    raise RuntimeError(f"membro inseguro no zip {name}: {info.filename!r}")
            zf.extractall(staging)  # noqa: S202 - membros validados acima
        series, facts = choose_series(staging)
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        shutil.move(str(series), str(dest / "series"))
        shutil.rmtree(staging)
        facts["zip_sha256"] = hashlib.sha256(zpath.read_bytes()).hexdigest()
        marker.write_text(json.dumps(facts, indent=1), encoding="utf-8")
        zpath.unlink()
        print(name, facts["description"], facts["slice_thickness_mm"], facts["n_files"], flush=True)


def series_dir(name: str) -> Path:
    """Diretório da série escolhida (layout novo) ou, para os exames do *spike*, pela regra."""
    new = DATA / "dicom" / name / "series"
    if new.is_dir():
        return new
    path, _ = choose_series(DATA / "dicom" / name)
    return path


def predict_one(name: str, weights_dir: Path) -> dict[str, Any]:
    """Roda o adaptador real sobre um exame e devolve a linha de predição."""
    import resource

    from radiologyai.analysis.base import ModelInputs
    from radiologyai.analysis.imaging import load_series
    from radiologyai.analysis.registry import Registry
    from radiologyai.catalog.schema import ExamType
    from radiologyai.config.settings import Settings

    registry = Registry(weights_dir=weights_dir, settings=Settings(weights_dir=weights_dir))
    sdir = series_dir(name)
    t0 = time.monotonic()
    image = load_series(sdir)
    upstream_id = "totalseg-total-3mm"
    total = registry.load(upstream_id)
    base_inputs = ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD)
    qa_total = total.applicability(base_inputs)
    if not qa_total.applicable:
        raise RuntimeError(f"{name}: totalseg-total-3mm não aplicável: {qa_total.reasons}")
    total_out = total.predict(base_inputs)
    t_total = time.monotonic() - t0
    bleed = registry.load("totalseg-cerebral-bleed")
    inputs = ModelInputs(
        images=(image,), exam_type=ExamType.CT_HEAD, upstream={upstream_id: total_out}
    )
    qa = bleed.applicability(inputs)
    row: dict[str, Any] = {
        "name": name,
        "series_dir": str(sdir),
        "input_sha256": image.sha256,
        "slice_thickness_mm": image.meta.slice_thickness_mm,
        "n_slices": int(image.pixels.shape[0]),
        "spacing_zyx_mm": json.dumps(image.spacing_mm),
        "applicable": qa.applicable,
        "abstention": ",".join(r.value for r in qa.reasons),
        "seconds_total3mm": round(t_total, 1),
    }
    if qa.applicable:
        t1 = time.monotonic()
        out = bleed.predict(inputs)
        row["seconds_bleed"] = round(time.monotonic() - t1, 1)
        meas = {m.key: m for m in out.measurements}
        vol = meas["intracerebral_hemorrhage_volume_ml"]
        row["bleed_volume_ml"] = vol.value
        row["bleed_components"] = vol.note
        row["warnings"] = " | ".join(out.warnings)
    row["peak_rss_gb_parent"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    row["peak_rss_gb_children"] = round(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 3
    )
    registry.unload_all()
    return row


def predict(names: list[str], weights_dir: Path) -> None:
    rows_dir = OUT / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = rows_dir / f"{name}.json"
        if target.is_file():
            continue
        try:
            row = predict_one(name, weights_dir)
        except Exception as exc:  # noqa: BLE001 - registrado como falha, nunca substituído
            row = {"name": name, "error": f"{type(exc).__name__}: {exc}"}
        target.write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
        print(json.dumps(row, default=str), flush=True)


def collect() -> Path:
    """Junta as linhas com o padrão de referência em ``predictions.parquet``."""
    import pandas as pd

    sel = {r["name"]: r for r in select()}
    rows = []
    for path in sorted((OUT / "rows").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        ref = sel.get(row["name"])
        if ref is None:
            continue
        row.update({k: v for k, v in ref.items() if k.startswith(("maj_", "votes_"))})
        row["category"] = ref["category"]
        marker = DATA / "dicom" / row["name"] / ".medai-series.json"
        if marker.is_file():
            facts = json.loads(marker.read_text(encoding="utf-8"))
            row["series_description"] = facts.get("description")
            row["gantry_tilt_deg"] = facts.get("gantry_tilt_deg")
            row["kernel"] = facts.get("kernel")
        vol = row.get("bleed_volume_ml")
        row["label_pred_ge_0_1ml"] = None if vol is None else int(vol >= 0.1)
        rows.append(row)
    frame = pd.DataFrame(rows)
    target = OUT / "predictions.parquet"
    frame.to_parquet(target, index=False)
    meta = {
        "model_id": "totalseg-cerebral-bleed",
        "dataset": "CQ500 (qure.ai), CC BY-NC-SA 4.0",
        "reference": "majority (>=2/3) R*:ICH from reads.csv; negatives = 0/3",
        "selection": f"sha256('{SALT}'+name) order, {N_POS} pos / {N_NEG} neg interleaved",
        "operating_point": "bleed_volume_ml >= 0.1 (manifest totalseg-cerebral-bleed)",
        "n_rows": len(rows),
        "n_pos": int(sum(r.get("maj_ICH", 0) for r in rows)),
        "n_errors": int(sum("error" in r for r in rows)),
        "script": "scripts/predict_ct_cq500.py",
    }
    (OUT / "predictions.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    dl = sub.add_parser("download")
    dl.add_argument("--limit", type=int, default=None)
    pr = sub.add_parser("predict")
    pr.add_argument("--one", action="append", default=[])
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--weights-dir", type=Path, default=Path("/root/medai-cache/weights/registry"))
    sub.add_parser("collect")
    args = parser.parse_args()
    if args.cmd == "select":
        print(write_selection())
    elif args.cmd == "download":
        download(args.limit)
    elif args.cmd == "predict":
        names = args.one or [
            r["name"]
            for r in select()[: args.limit or None]
            if (DATA / "dicom" / r["name"]).is_dir()
        ]
        predict(names, args.weights_dir)
    elif args.cmd == "collect":
        print(collect())
    return 0


if __name__ == "__main__":
    sys.exit(main())
