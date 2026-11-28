#!/usr/bin/env python
"""Inferência em lote dos classificadores TorchXRayVision sobre conjuntos públicos.

Usa os MESMOS adaptadores do produto (``Registry.load`` → pesos verificados por
sha256 → ``xrv_input``), e grava por imagem os logits brutos de todas as cabeças
declaradas, os rótulos do dataset e as covariáveis (paciente, sexo, idade,
incidência) para a fábrica de evidência. Não calcula métrica nenhuma.

Retomável: cada fragmento (``--shard``) vira um arquivo em ``parts/`` e é
pulado se já existe. Rode cada fragmento sob o lock pesado, por exemplo::

    for k in 0 1 2 3 4 5 6 7 8 9; do
      /root/medai-cache/bin/heavy /root/medai-cache/bin/timev \\
        .venv/bin/python scripts/predict_xr_xrv.py --dataset nih-cxr14-test --shard $k
    done
    .venv/bin/python scripts/predict_xr_xrv.py --dataset nih-cxr14-test --merge

Conjuntos: ``nih-cxr14-test`` (split oficial de teste, 10 fragmentos parquet),
``openi-frontal`` (Open-i com ViewPosition PA/AP no índice, 10 fragmentos).
O subconjunto do Google (releitura adjudicada) é derivado no ``--merge`` de
``nih-cxr14-test`` por junção com ``google_labels_joined.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from predict_xr_common import (  # noqa: E402
    EVAL_PREDS,
    WEIGHTS_DIR,
    command_line,
    git_sha,
    raster_image,
    write_json,
)

DATASETS = Path("/root/medai-cache/datasets")
NIH_DIR = DATASETS / "nih-cxr14"
OPENI_DIR = DATASETS / "openi"
DEFAULT_MODELS = (
    "xrv-densenet121-res224-pc,xrv-densenet121-res224-chex,xrv-densenet121-res224-mimic_ch,"
    "xrv-densenet121-res224-mimic_nb,xrv-densenet121-res224-all,xrv-densenet121-res224-nih"
)
N_SHARDS = 10
BATCH = 32
_ROOT = [EVAL_PREDS]

NIH_TO_SLUG = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural_effusion",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "pulmonary_edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural_thickening",
    "Hernia": "hernia",
}
OPENI_TO_SLUG = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Edema": "pulmonary_edema",
    "Effusion": "pleural_effusion",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Fracture": "fracture",
    "Hernia": "hernia",
    "Infiltration": "infiltration",
    "Lung Lesion": "lung_lesion",
    "Mass": "mass",
    "Nodule": "nodule",
    "Lung Opacity": "lung_opacity",
    "Pleural_Thickening": "pleural_thickening",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}
GOOGLE_TO_SLUG = {
    "Fracture": "fracture",
    "Pneumothorax": "pneumothorax",
    "Airspace opacity": "lung_opacity",
    "Nodule or mass": "nodule_or_mass",
}


# --------------------------------------------------------------------------- #
# Fontes
# --------------------------------------------------------------------------- #
def nih_rows(shard: int) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = NIH_DIR / "test" / f"test-{shard:05d}-of-{N_SHARDS:05d}.parquet"
    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg)
        for row in table.to_pylist():
            names = set(row["label_names"] or [])
            labels = {f"y_{slug}": int(name in names) for name, slug in NIH_TO_SLUG.items()}
            labels["y_no_finding"] = int(not (names & set(NIH_TO_SLUG)))
            yield {
                "image_id": row["image_id"],
                "patient_id": str(row["patient_id"]),
                "sex": row["patient_sex"],
                "age": float(row["patient_age"]) if row["patient_age"] is not None else None,
                "view": row["view_position"],
                "bytes": row["image"]["bytes"],
                **labels,
            }


def openi_index() -> Any:
    import pandas as pd

    d = pd.read_csv(OPENI_DIR / "openi_index.csv")
    d = d[d["View Position"].isin(["PA", "AP"]) & d["has_png"]].sort_values("imageid")
    return d.reset_index(drop=True)


def openi_rows(shard: int) -> Iterator[dict[str, Any]]:
    d = openi_index()
    part = d.iloc[shard::N_SHARDS]
    for _, row in part.iterrows():
        data = (OPENI_DIR / "png" / f"{row['imageid']}.png").read_bytes()
        labels = {}
        for name, slug in OPENI_TO_SLUG.items():
            v = row.get(f"lab_{name}")
            labels[f"y_{slug}"] = None if v != v else int(v)  # NaN -> None
        yield {
            "image_id": row["imageid"],
            "patient_id": str(row["uid"]),
            "sex": None,
            "age": None,
            "view": row["View Position"],
            "bytes": data,
            **labels,
        }


SOURCES = {"nih-cxr14-test": nih_rows, "openi-frontal": openi_rows}


# --------------------------------------------------------------------------- #
# Inferência
# --------------------------------------------------------------------------- #
def load_adapters(model_ids: list[str]) -> dict[str, Any]:
    from radiologyai.analysis.registry import Registry
    from radiologyai.config.settings import Settings

    settings = Settings(weights_dir=WEIGHTS_DIR, device="cpu", max_loaded_models=len(model_ids) + 2)
    registry = Registry(weights_dir=WEIGHTS_DIR, settings=settings)
    return {mid: (registry.load(mid), registry.provenance(mid)) for mid in model_ids}


def run_shard(dataset: str, shard: int, model_ids: list[str], limit: int | None = None) -> None:
    import pandas as pd
    import torch

    from radiologyai.analysis.adapters.xr_common import xrv_input

    torch.set_num_threads(4)
    pending = [m for m in model_ids if not (part_path(m, dataset, shard)).is_file()]
    if not pending:
        print(f"[{dataset}#{shard}] já feito para todos os modelos")
        return
    adapters = load_adapters(pending)
    meta_rows: list[dict[str, Any]] = []
    logits: dict[str, list[np.ndarray]] = {m: [] for m in pending}
    started = time.monotonic()
    batch_planes: list[np.ndarray] = []

    def flush() -> None:
        if not batch_planes:
            return
        tensor = torch.from_numpy(np.stack(batch_planes))[:, None]
        for mid in pending:
            adapter = adapters[mid][0]
            logits[mid].append(adapter.logits(tensor).numpy().astype(np.float32))
        batch_planes.clear()

    for i, row in enumerate(SOURCES[dataset](shard)):
        if limit is not None and i >= limit:
            break
        data = row.pop("bytes")
        image = raster_image(data, view=row["view"], age=row["age"], sex=row["sex"])
        batch_planes.append(xrv_input(image, 224))
        row["image_sha256"] = hashlib.sha256(data).hexdigest()
        meta_rows.append(row)
        if len(batch_planes) == BATCH:
            flush()
        if (i + 1) % 512 == 0:
            rate = (i + 1) / (time.monotonic() - started)
            print(f"[{dataset}#{shard}] {i + 1} imagens, {rate:.2f} img/s", flush=True)
    flush()
    base = pd.DataFrame(meta_rows)
    for mid in pending:
        adapter, prov = adapters[mid]
        arr = np.concatenate(logits[mid], axis=0)
        frame = base.copy()
        for key, idx in adapter._heads.items():
            frame[f"logit_{key}"] = arr[:, idx]
        out = part_path(mid, dataset, shard)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(out)
        write_json(
            out.with_suffix(".json"),
            {
                "model_id": mid,
                "provenance": prov.model_dump(),
                "dataset": dataset,
                "shard": shard,
                "n": len(frame),
                "git_sha": git_sha(),
                "command": command_line(),
                "seconds": round(time.monotonic() - started, 1),
                "preprocessing": "radiologyai.analysis.adapters.xr_common.xrv_input(size=224)",
            },
        )
        print(f"[{dataset}#{shard}] {mid}: {len(frame)} linhas -> {out}")


def part_path(model_id: str, dataset: str, shard: int) -> Path:
    return _ROOT[0] / model_id / dataset / "parts" / f"shard-{shard:02d}.parquet"


def merge(dataset: str, model_ids: list[str]) -> None:
    import json

    import pandas as pd

    for mid in model_ids:
        parts = [part_path(mid, dataset, k) for k in range(N_SHARDS)]
        missing = [p.name for p in parts if not p.is_file()]
        if missing:
            print(f"{mid}/{dataset}: faltam {missing}; nada mesclado")
            continue
        frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if frame["image_id"].duplicated().any():
            raise SystemExit(f"{mid}/{dataset}: image_id duplicado")
        out = EVAL_PREDS / mid / dataset / "predictions.parquet"
        frame.to_parquet(out, index=False)
        shard_meta = [json.loads(p.with_suffix(".json").read_text()) for p in parts]
        write_json(
            out.with_suffix(".json"),
            {
                "model_id": mid,
                "dataset": dataset,
                "n": len(frame),
                "n_patients": int(frame["patient_id"].nunique()),
                "provenance": shard_meta[0]["provenance"],
                "git_sha": sorted({m["git_sha"] for m in shard_meta}),
                "dataset_files": dataset_files(dataset),
                "columns": list(frame.columns),
                "score_kind": "logit",
            },
        )
        print(f"{mid}/{dataset}: {len(frame)} imagens -> {out}")
        if dataset == "nih-cxr14-test":
            merge_google(mid, frame)


def merge_google(mid: str, frame: Any) -> None:
    import pandas as pd

    g = pd.read_csv(NIH_DIR / "google_labels_joined.csv")
    g = g[g["in_nih_test"]]
    keep = ["image_id"] + list(GOOGLE_TO_SLUG)
    g = g[keep].rename(columns={k: f"y_{v}" for k, v in GOOGLE_TO_SLUG.items()})
    for col in [f"y_{v}" for v in GOOGLE_TO_SLUG.values()]:
        g[col] = g[col].map({"YES": 1, "NO": 0})
    base = frame.drop(columns=[c for c in frame.columns if c.startswith("y_")])
    joined = base.merge(g, on="image_id", how="inner")
    out = EVAL_PREDS / mid / "nih-google-adjudicated" / "predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out, index=False)
    write_json(
        out.with_suffix(".json"),
        {
            "model_id": mid,
            "dataset": "nih-google-adjudicated (subconjunto do teste oficial do NIH)",
            "n": len(joined),
            "labels_source": str(NIH_DIR / "google_labels_joined.csv"),
            "labels_sha256": hashlib.sha256(
                (NIH_DIR / "google_labels_joined.csv").read_bytes()
            ).hexdigest(),
            "derived_from": str(EVAL_PREDS / mid / "nih-cxr14-test" / "predictions.parquet"),
        },
    )
    print(f"{mid}/google: {len(joined)} imagens -> {out}")


def dataset_files(dataset: str) -> dict[str, str]:
    if dataset == "nih-cxr14-test":
        sums = (NIH_DIR / "test" / "SHA256SUMS").read_text().split("\n")
        return {line.split()[1]: line.split()[0] for line in sums if line.strip()}
    index = OPENI_DIR / "openi_index.csv"
    return {
        "openi_index.csv": hashlib.sha256(index.read_bytes()).hexdigest(),
        "png_SHA256SUMS": hashlib.sha256((OPENI_DIR / "SHA256SUMS").read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, help="só para teste de fumaça (grava em outro lugar)")
    args = parser.parse_args()
    model_ids = [m for m in args.models.split(",") if m]
    if args.limit is not None:
        _ROOT[0] = Path("/tmp/claude-0/xr-smoke")
    if args.merge:
        merge(args.dataset, model_ids)
    elif args.shard is not None:
        run_shard(args.dataset, args.shard, model_ids, args.limit)
    else:
        parser.error("use --shard K ou --merge")


if __name__ == "__main__":
    main()
