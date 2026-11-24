"""Synthetic inputs for unit tests ONLY.

Nothing in this package is importable from ``medai``; product code never sees
synthetic data. Images here exist to exercise plumbing, not to measure accuracy.
"""

from __future__ import annotations

import hashlib

import numpy as np

from medai.core.types import ImageData, ImageKind, ImageMeta, Modality


def _meta(pixels: np.ndarray, **overrides: object) -> ImageMeta:
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    base: dict[str, object] = {"source_format": "raster", "content_sha256": digest}
    base.update(overrides)
    return ImageMeta.model_validate(base)


def projection(
    seed: int = 0,
    shape: tuple[int, int] = (256, 256),
    modality: Modality = Modality.DX,
    level: float = 100.0,
    **meta: object,
) -> ImageData:
    rng = np.random.default_rng(seed)
    pixels = (level + 20.0 * rng.standard_normal(shape)).clip(0, 255).astype(np.float32)
    return ImageData(
        pixels=pixels,
        kind=ImageKind.PROJECTION,
        modality=modality,
        meta=_meta(pixels, **{"body_part": "CHEST", **meta}),
        spacing_mm=(0.2, 0.2),
        value_range=(0.0, 255.0),
    )


def volume(seed: int = 0, shape: tuple[int, int, int] = (8, 32, 32)) -> ImageData:
    rng = np.random.default_rng(seed)
    pixels = (40.0 + 30.0 * rng.standard_normal(shape)).astype(np.float32)
    return ImageData(
        pixels=pixels,
        kind=ImageKind.VOLUME,
        modality=Modality.CT,
        meta=_meta(pixels, body_part="ABDOMEN", source_format="nifti"),
        spacing_mm=(3.0, 1.0, 1.0),
        affine_lps=np.diag([1.0, 1.0, 3.0, 1.0]),
        value_range=(-1024.0, 3071.0),
    )
