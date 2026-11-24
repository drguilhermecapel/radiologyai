#!/usr/bin/env python
"""Mede os portões de RX de tórax (incidência e autoencoder OOD) em tórax x não tórax.

Roda os MESMOS modelos/pré-processamento dos adaptadores ``xr_gates`` sobre:

* tórax frontal: NIH (teste oficial, fragmento 1), Open-i frontal, Shenzhen,
  Montgomery, DICOMs TCIA de amostra;
* tórax lateral: Open-i com ViewPosition LL/LATERAL;
* não tórax: RX musculoesquelético (FracAtlas, GRAZPEDWRI-DX), mamografia (CMMD,
  CBIS-DDSM), ultrassom (BUSI, BUS-BRA), cortes de TC (TCIA, CQ500), fotografias
  (``skimage.data``) e ruído sintético (gerador com semente fixa; aqui, num
  script de medição, nunca no pacote).

Regra do portão OOD, PRÉ-ESPECIFICADA antes da medição: o limiar é o percentil
99,5 do erro de reconstrução nas imagens de tórax frontal do conjunto de
CALIBRAÇÃO (metade determinística por paciente de cada fonte de tórax); a outra
metade (teste) e todas as fontes não tórax medem a taxa de aceitação/recusa.
Grava ``<eval-preds>/_gates/gates.parquet`` e ``summary.json``.
"""

from __future__ import annotations

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

from radiologyai.analysis.types import ImageData, ImageKind, ImageMeta, Modality  # noqa: E402

D = Path("/root/medai-cache/datasets")
OUT = EVAL_PREDS / "_gates"
PER_SOURCE = 150
QUANTILE = 99.5


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_image(arr: np.ndarray, modality: Modality = Modality.OT) -> ImageData:
    arr = np.asarray(arr, dtype=np.float32)
    meta = ImageMeta(source_format="raster", content_sha256=_sha(arr.tobytes()))
    return ImageData(
        pixels=arr,
        kind=ImageKind.PROJECTION,
        modality=modality,
        meta=meta,
        value_range=(0.0, 255.0),
    )


def sources() -> Iterator[tuple[str, str, str, str, ImageData]]:
    """(fonte, grupo, id da imagem, id do paciente, imagem)."""
    import pandas as pd
    import pyarrow.parquet as pq

    from radiologyai.analysis.imaging import load_dicom

    # --- tórax frontal
    pf = pq.ParquetFile(D / "nih-cxr14" / "test" / "test-00001-of-00010.parquet")
    rows = pf.read_row_group(0, columns=["image", "image_id", "patient_id"]).to_pylist()
    for r in rows[: PER_SOURCE * 2]:
        yield (
            "nih",
            "chest_frontal",
            r["image_id"],
            f"nih:{r['patient_id']}",
            raster_image(r["image"]["bytes"]),
        )
    idx = pd.read_csv(D / "openi" / "openi_index.csv").sort_values("imageid")
    frontal = idx[idx["View Position"].isin(["PA", "AP"])].iloc[: PER_SOURCE * 2]
    for _, r in frontal.iterrows():
        data = (D / "openi" / "png" / f"{r.imageid}.png").read_bytes()
        yield ("openi", "chest_frontal", r.imageid, f"openi:{r.uid}", raster_image(data))
    for name in ("tb-shenzhen", "tb-montgomery"):
        t = pd.read_csv(D / name / "index.tsv", sep="\t", dtype={"image_id": str})
        for _, r in t.iloc[:PER_SOURCE].iterrows():
            data = (D / name / "raw" / "CXR_png" / f"{r.image_id}.png").read_bytes()
            yield (name, "chest_frontal", r.image_id, f"{name}:{r.case_number}", raster_image(data))
    for p in sorted((D / "samples-cxr-dicom").glob("*/*.dcm")):
        yield ("tcia-cxr", "chest_frontal", p.parent.name, f"tcia:{p.parent.name}", load_dicom(p))
    # --- tórax lateral
    lateral = idx[idx["View Position"].isin(["LL", "LATERAL"])].iloc[:PER_SOURCE]
    for _, r in lateral.iterrows():
        data = (D / "openi" / "png" / f"{r.imageid}.png").read_bytes()
        yield ("openi-lateral", "chest_lateral", r.imageid, f"openi:{r.uid}", raster_image(data))
    # --- RX não tórax
    fa = sorted((D / "fracatlas" / "raw" / "FracAtlas" / "images").glob("*/*.jpg"))
    for p in fa[:: max(1, len(fa) // PER_SOURCE)][:PER_SOURCE]:
        yield ("fracatlas", "msk_xr", p.name, f"fa:{p.name}", raster_image(p.read_bytes()))
    gz = sorted((D / "grazpedwri" / "images_sample").glob("*.png"))
    for p in gz[:: max(1, len(gz) // PER_SOURCE)][:PER_SOURCE]:
        yield (
            "grazpedwri",
            "msk_xr",
            p.stem,
            f"gz:{p.stem.split('_')[0]}",
            raster_image(p.read_bytes()),
        )
    # --- mamografia
    for p in sorted(
        [*(D / "cmmd-sample").glob("*/*.dcm"), *(D / "cbis-ddsm-sample").glob("*/*.dcm")]
    ):
        yield (
            "mammo",
            "mammography",
            p.parent.name + "/" + p.name,
            f"mg:{p.parent.name}",
            load_dicom(p),
        )
    # --- ultrassom
    bp = pq.ParquetFile(D / "busi" / "raw" / "train-00000-of-00001.parquet")
    busi = bp.read_row_group(0, columns=["image", "image_id"]).to_pylist()
    for r in busi[:: max(1, len(busi) // PER_SOURCE)][:PER_SOURCE]:
        yield (
            "busi",
            "ultrasound",
            r["image_id"],
            f"busi:{r['image_id']}",
            raster_image(r["image"]["bytes"], modality=Modality.US),
        )
    bb = sorted((D / "bus-bra" / "extracted" / "BUSBRA" / "Images").glob("*.png"))
    for p in bb[:: max(1, len(bb) // PER_SOURCE)][:PER_SOURCE]:
        yield (
            "bus-bra",
            "ultrasound",
            p.stem,
            f"busbra:{p.stem.split('-')[0]}",
            raster_image(p.read_bytes(), modality=Modality.US),
        )
    # --- cortes de TC
    ct = sorted((D / "samples-ct" / "tcia").glob("*/*/*.dcm"))
    for p in ct[:: max(1, len(ct) // PER_SOURCE)][:PER_SOURCE]:
        yield ("ct-tcia", "ct_slice", str(p.relative_to(D)), f"ct:{p.parts[-3]}", load_dicom(p))
    cq = sorted((D / "cq500" / "dicom").rglob("*.dcm"))
    for p in cq[:: max(1, len(cq) // 60)][:60]:
        yield ("ct-cq500", "ct_slice", str(p.relative_to(D)), f"cq:{p.parts[-5]}", load_dicom(p))
    # --- fotografias
    import skimage.data

    for name in (
        "astronaut",
        "coffee",
        "chelsea",
        "rocket",
        "camera",
        "coins",
        "moon",
        "page",
        "text",
        "horse",
        "brick",
        "grass",
        "gravel",
        "hubble_deep_field",
        "immunohistochemistry",
    ):
        img = getattr(skimage.data, name)()
        arr = img.mean(axis=2) if img.ndim == 3 else img
        if arr.dtype == bool:
            arr = arr.astype(np.float32) * 255.0
        yield ("photo", "photo", name, f"photo:{name}", _array_image(arr))
    # --- ruído sintético (semente fixa)
    rng = np.random.default_rng(20261124)
    for i in range(10):
        yield (
            "noise",
            "noise",
            f"uniform{i}",
            f"noise:u{i}",
            _array_image(rng.uniform(0, 255, (512, 512))),
        )
        yield (
            "noise",
            "noise",
            f"gauss{i}",
            f"noise:g{i}",
            _array_image(np.clip(rng.normal(128, 40, (512, 512)), 0, 255)),
        )
    for v in (0.0, 128.0, 255.0):
        yield (
            "noise",
            "noise",
            f"const{int(v)}",
            f"noise:c{int(v)}",
            _array_image(np.full((512, 512), v)),
        )


def is_calibration(patient: str) -> bool:
    return int(hashlib.sha256(f"gate-calib:{patient}".encode()).hexdigest()[:8], 16) % 2 == 0


def main() -> None:
    import pandas as pd
    import torch

    from radiologyai.analysis.adapters.xr_common import load_torch_verified, xrv_input
    from radiologyai.analysis.adapters.xr_gates import (
        ae_reconstruction_mae,
        build_autoencoder,
        build_view_model,
        view_probabilities,
    )
    from radiologyai.analysis.base import ResolvedWeights

    torch.set_num_threads(4)
    ae_file = WEIGHTS_DIR / "xrv-ae-ood" / "nihpcrsnamimic_ch-resnet101-2-ae-test2-elastic-e250.pt"
    ae_sha = hashlib.sha256(ae_file.read_bytes()).hexdigest()
    rw = ResolvedWeights(files={"model": ae_file}, sha256={"model": ae_sha})
    ae = build_autoencoder(load_torch_verified(rw, "model", allow_pickle=False))
    from radiologyai.analysis.registry import Registry
    from radiologyai.config.settings import Settings

    reg = Registry(
        weights_dir=WEIGHTS_DIR, settings=Settings(weights_dir=WEIGHTS_DIR, device="cpu")
    )
    view_rw = reg.verify("xrv-view-xinario")
    view = build_view_model(load_torch_verified(view_rw, "model", allow_pickle=False))

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for i, (src, group, image_id, patient, image) in enumerate(sources()):
        try:
            x = torch.from_numpy(xrv_input(image, 224))[None, None]
        except Exception as exc:  # noqa: BLE001 - registra recusa do pré-processamento
            rows.append(
                {
                    "source": src,
                    "group": group,
                    "image_id": image_id,
                    "patient": patient,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        mae = float(ae_reconstruction_mae(ae, x)[0])
        pf, pl = view_probabilities(view, x)[0]
        rows.append(
            {
                "source": src,
                "group": group,
                "image_id": image_id,
                "patient": patient,
                "modality": image.modality.value,
                "ae_mae": mae,
                "p_frontal": float(pf),
                "p_lateral": float(pl),
                "error": None,
            }
        )
        if (i + 1) % 100 == 0:
            print(
                f"{i + 1} imagens, {(i + 1) / (time.monotonic() - started):.2f} img/s", flush=True
            )
    frame = pd.DataFrame(rows)
    frame["calibration"] = frame["patient"].map(is_calibration) & (
        frame["group"] == "chest_frontal"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / "gates.parquet", index=False)

    ok = frame[frame["error"].isna()]
    calib = ok[ok["calibration"]]
    threshold = float(np.percentile(calib["ae_mae"], QUANTILE))
    summary: dict[str, Any] = {
        "rule": f"limiar = percentil {QUANTILE} do ae_mae no tórax frontal de calibração",
        "ae_mae_threshold": round(threshold, 6),
        "n_calibration": int(len(calib)),
        "weights_sha256": {"ae": ae_sha, "view": view_rw.sha256["model"]},
        "git_sha": git_sha(),
        "command": command_line(),
        "seconds": round(time.monotonic() - started, 1),
        "by_group": {},
        "preprocessing_refusals": frame[frame["error"].notna()]
        .groupby("source")
        .size()
        .astype(int)
        .to_dict(),
    }
    test = ok[~ok["calibration"]]
    for (group, src), g in test.groupby(["group", "source"]):
        summary["by_group"][f"{group}/{src}"] = {
            "n": int(len(g)),
            "ae_accept_frac": round(float((g["ae_mae"] <= threshold).mean()), 4),
            "view_frontal_frac": round(float((g["p_frontal"] >= g["p_lateral"]).mean()), 4),
            "ae_mae_p5_p50_p95": [
                round(float(v), 5) for v in np.percentile(g["ae_mae"], [5, 50, 95])
            ],
        }
    write_json(OUT / "summary.json", summary)
    import json

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
