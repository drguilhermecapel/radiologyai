"""Construtores de volumes sintéticos de geometria conhecida (só para testes; nunca distribuído).

Volumes ``(Z, Y, X)`` com afim LPS axial: coluna -> +L, linha -> +P, corte -> +S.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from radiologyai.analysis.biomarkers.ct import CTContext
from radiologyai.analysis.types import ImageData, ImageKind, ImageMeta, Modality

SHA = "0" * 64


def meta(**overrides: Any) -> ImageMeta:
    data: dict[str, Any] = {"source_format": "nifti", "content_sha256": SHA}
    data.update(overrides)
    return ImageMeta.model_validate(data)


def ct_meta(**overrides: Any) -> ImageMeta:
    """Metadados de uma TC que satisfaz todas as pré-condições (salvo sobrescrita)."""
    base: dict[str, Any] = {
        "kvp": 120.0,
        "patient_age_years": 60.0,
        "contrast_phase": "none",
        "slice_thickness_mm": 1.0,
        "convolution_kernel": "B30f",
        "series_description": "TORAX INSP",
    }
    base.update(overrides)
    return meta(**base)


def affine(spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    out = np.eye(4)
    out[:3, 0] = (sx, 0.0, 0.0)
    out[:3, 1] = (0.0, sy, 0.0)
    out[:3, 2] = (0.0, 0.0, sz)
    return out


def grid_mm(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> tuple[Any, ...]:
    """Coordenadas (z, y, x) em mm de cada voxel."""
    z, y, x = np.meshgrid(
        *(np.arange(n) * s for n, s in zip(shape, spacing, strict=True)), indexing="ij"
    )
    return z, y, x


def context(
    hu: np.ndarray,
    labelmap: np.ndarray,
    names: dict[str, int],
    spacing: tuple[float, float, float],
    image_meta: ImageMeta | None = None,
) -> CTContext:
    return CTContext(
        hu=hu.astype(np.float32),
        spacing_zyx=spacing,
        affine_lps=affine(spacing),
        meta=image_meta or ct_meta(),
        labelmap=labelmap,
        names=names,
        source="total",
    )


def volume_image(
    pixels: np.ndarray,
    spacing: tuple[float, float, float],
    modality: Modality = Modality.CT,
    image_meta: ImageMeta | None = None,
) -> ImageData:
    return ImageData(
        pixels=pixels.astype(np.float32),
        kind=ImageKind.VOLUME,
        modality=modality,
        meta=image_meta or ct_meta(),
        spacing_mm=spacing,
        affine_lps=affine(spacing),
        value_range=(-1024.0, 3071.0),
    )
