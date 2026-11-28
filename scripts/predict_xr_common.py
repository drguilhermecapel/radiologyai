"""Auxiliares dos scripts de predição/sonda de RX (não faz parte do pacote).

Constrói ``ImageData`` a partir de bytes de raster com a MESMA semântica de
``radiologyai.analysis.imaging.load_raster`` (modo L/I;16, conversão RGB→L
ITU-R 601, faixa representável pelo tipo), para que as predições em lote usem
exatamente o pré-processamento do produto.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from radiologyai.analysis.types import ImageData, ImageKind, ImageMeta, Modality

REPO = Path(__file__).resolve().parent.parent
EVAL_PREDS = Path(os.environ.get("MEDAI_EVAL_PREDS", "/root/medai-cache/eval-preds"))
WEIGHTS_DIR = Path(os.environ.get("MEDAI_WEIGHTS_DIR", "/root/medai-cache/weights/registry"))


def raster_image(
    data: bytes,
    *,
    modality: Modality = Modality.DX,
    view: str | None = None,
    age: float | None = None,
    sex: str | None = None,
    body_part: str | None = "CHEST",
) -> ImageData:
    """Bytes PNG/JPEG → ``ImageData`` (igual a ``load_raster``, sem arquivo)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        mode = img.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(img, dtype=np.float32)
            bits = 16 if mode.startswith("I;16") else 32
            value_range = (0.0, float(2**bits - 1))
        elif mode in ("L", "P", "LA", "1"):
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            value_range = (0.0, 255.0)
        else:
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            value_range = (0.0, 255.0)
    meta = ImageMeta.model_validate(
        {
            "source_format": "raster",
            "content_sha256": hashlib.sha256(data).hexdigest(),
            "rows": int(arr.shape[0]),
            "columns": int(arr.shape[1]),
            "view_position": view,
            "patient_age_years": age,
            "patient_sex": sex if sex in ("F", "M", "O") else None,
            "body_part": body_part,
        }
    )
    return ImageData(
        pixels=np.ascontiguousarray(arr, dtype=np.float32),
        kind=ImageKind.PROJECTION,
        modality=modality,
        meta=meta,
        value_range=value_range,
    )


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def command_line() -> str:
    return " ".join([Path(sys.executable).name, *sys.argv])
