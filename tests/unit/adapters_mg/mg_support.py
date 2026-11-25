"""Construtores de entradas sintéticas de mamografia para os testes de MG."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from radiologyai.analysis.types import ImageData, ImageKind, ImageMeta, Modality
from radiologyai.modalities.mg.voi import VOI_EXTRA_KEY

WEIGHTS = Path("/root/medai-cache/weights/registry")
CMMD = Path("/root/medai-cache/datasets/cmmd-sample")
REFERENCE = Path(__file__).with_name("mg_original_reference.json")


def voi(
    *,
    photometric: str = "MONOCHROME2",
    bits: int = 12,
    center: float | None = None,
    width: float | None = None,
    function: str | None = None,
    lut: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "photometric": photometric,
        "bits_stored": bits,
        "pixel_representation": 0,
        "rescale_slope": 1.0,
        "rescale_intercept": 0.0,
        "modality_lut_bits": None,
        "window_center": [] if center is None else [center],
        "window_width": [] if width is None else [width],
        "function": function,
        "lut": lut,
    }


def mg_image(
    pixels: np.ndarray,
    *,
    laterality: str | None = "L",
    view: str | None = "CC",
    voi_params: dict[str, Any] | None = None,
    source_format: str = "dicom",
    value_range: tuple[float, float] = (0.0, 4095.0),
    age: float | None = 55.0,
    intent: str | None = "FOR PRESENTATION",
    tag: str = "",
) -> ImageData:
    arr = np.ascontiguousarray(pixels, dtype=np.float32)
    digest = hashlib.sha256(arr.tobytes() + f"{laterality}{view}{tag}".encode()).hexdigest()
    meta = ImageMeta(
        source_format=source_format,  # type: ignore[arg-type]
        content_sha256=digest,
        laterality=laterality,  # type: ignore[arg-type]
        view_position=view,
        patient_age_years=age,
        presentation_intent=intent,
        rows=int(arr.shape[0]),
        columns=int(arr.shape[1]),
    )
    extra: dict[str, Any] = {}
    if source_format == "dicom":
        extra[VOI_EXTRA_KEY] = voi_params if voi_params is not None else voi()
    return ImageData(
        pixels=arr,
        kind=ImageKind.PROJECTION,
        modality=Modality.MG,
        meta=meta,
        value_range=value_range,
        extra=extra,
    )


def gradient(h: int = 300, w: int = 260, top: int = 4095) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = (xx / (w - 1)) * top * 0.7 + (yy / (h - 1)) * top * 0.3
    return np.round(img)
