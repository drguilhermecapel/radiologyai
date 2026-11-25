"""Construtores sintéticos para os testes de US/eco (geometria conhecida; sem pixel de paciente)."""

from __future__ import annotations

import hashlib
import importlib.util
from typing import Any

import numpy as np
from cache_paths import MEDAI_CACHE

from radiologyai.analysis.types import ImageData, ImageKind, ImageMeta, Modality

CACHE = MEDAI_CACHE
WEIGHTS = MEDAI_CACHE / "weights/registry"
DATASETS = MEDAI_CACHE / "datasets"


def have(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


def half_ellipse(
    a_mm: float, b_mm: float, spacing: tuple[float, float], *, apex_up: bool = True
) -> np.ndarray:
    """Meia elipse (semi-eixo longo ``a`` vertical, ``b`` horizontal), base reta horizontal.

    Girada em torno do eixo longo, é meio esferoide prolato: volume ``2/3·π·b²·a``.
    """
    h = int(a_mm * 1.4 / spacing[0])
    w = int(b_mm * 2.8 / spacing[1])
    r = (np.arange(h) + 0.5) * spacing[0]
    c = (np.arange(w) + 0.5) * spacing[1]
    rr, cc = np.meshgrid(r, c, indexing="ij")
    base = a_mm * 1.2
    cx = w * spacing[1] / 2.0
    mask = (((rr - base) / a_mm) ** 2 + ((cc - cx) / b_mm) ** 2 <= 1.0) & (rr <= base)
    return mask if apex_up else mask[::-1].copy()


def ellipse(
    shape: tuple[int, int],
    center: tuple[float, float],
    axes: tuple[float, float],
    spacing: tuple[float, float],
) -> np.ndarray:
    """Elipse cheia em mm: ``center`` e ``axes`` em (linha, coluna) mm."""
    r = (np.arange(shape[0]) + 0.5) * spacing[0]
    c = (np.arange(shape[1]) + 0.5) * spacing[1]
    rr, cc = np.meshgrid(r, c, indexing="ij")
    return ((rr - center[0]) / axes[0]) ** 2 + ((cc - center[1]) / axes[1]) ** 2 <= 1.0


def us_image(
    pixels: np.ndarray,
    *,
    kind: ImageKind = ImageKind.PROJECTION,
    spacing: tuple[float, float] | None = (0.2, 0.2),
    color: bool = False,
    modality: Modality = Modality.US,
    **meta: Any,
) -> ImageData:
    px = np.ascontiguousarray(pixels, dtype=np.float32)
    digest = hashlib.sha256(px.tobytes()).hexdigest()
    m = ImageMeta.model_validate(
        {
            "source_format": "raster",
            "content_sha256": digest,
            "n_frames": int(px.shape[0]) if kind is ImageKind.CINE else 1,
            "pixel_spacing_source": "UltrasoundRegions" if spacing else "none",
            **meta,
        }
    )
    return ImageData(
        pixels=px,
        kind=kind,
        modality=modality,
        meta=m,
        spacing_mm=spacing,
        value_range=(0.0, 255.0),
        color=color,
    )


def sector_frame(h: int = 200, w: int = 240) -> np.ndarray:
    """Quadro modo B sintético: setor (cunha) com textura determinística, fundo preto."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w / 2.0
    inside = (yy > 10) & (np.abs(xx - cx) < (yy - 10) * 0.8) & (yy < h - 10)
    tex = 60.0 + 40.0 * np.sin(xx / 7.0) * np.cos(yy / 5.0)
    return np.where(inside, tex, 0.0).astype(np.float32)
