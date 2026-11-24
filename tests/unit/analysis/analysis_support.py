"""Auxiliares dos testes da análise multimodelo (manifestos, catálogos, imagens sintéticas).

Imagens daqui exercitam encanamento, não medem desempenho. Nada disto é
importável a partir de ``radiologyai``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from radiologyai.analysis.types import ImageData, ImageKind, ImageMeta, Modality
from radiologyai.catalog.catalog import Catalog
from radiologyai.catalog.schema import (
    CodedConcept,
    Condition,
    EvidenceLevel,
    ExamType,
    LevelEntry,
    LevelsFile,
    visibility_for,
)

FAKE_MODULE = "radiologyai.analysis.adapters._test_fakes"
WEIGHT_BYTES = b"nao-e-um-modelo-real: fixture do teste de integridade\n"
WEIGHT_SHA = hashlib.sha256(WEIGHT_BYTES).hexdigest()
CODE = {"scheme": "99MEDAI", "value": "x", "meaning": "x"}


def manifest_dict(model_id: str, adapter_class: str, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": model_id,
        "version": "0.0-test",
        "name": model_id,
        "status": "available",
        "adapter": f"{FAKE_MODULE}:{adapter_class}",
        "task": "classification",
        "capabilities": ["classify"],
        "summary": "dublê de teste",
        "weights": [
            {
                "key": "main",
                "url": f"https://weights.invalid/{model_id}.bin",
                "filename": "main.bin",
                "sha256": WEIGHT_SHA,
                "bytes": len(WEIGHT_BYTES),
            }
        ],
        "license": {"code": "Apache-2.0", "weights": "teste", "commercial_use": "yes"},
        "applicability": {
            "exam_types": [ExamType.CXR.value],
            "modalities": ["DX", "CR", "MG"],
            "image_kinds": ["projection"],
        },
        "labels": [
            {
                "key": "bright",
                "source_label": "Bright",
                "condition_id": "cxr.bright",
                "code": CODE,
            }
        ],
    }
    data.update(overrides)
    return data


def op(role: str, threshold: float, on: str = "sigmoid", direction: str = "ge") -> dict[str, Any]:
    return {
        "name": role,
        "role": role,
        "threshold": threshold,
        "on": on,
        "direction": direction,
        "source": "teste",
    }


def condition(condition_id: str) -> Condition:
    return Condition(
        id=condition_id,
        name_pt=condition_id,
        name_en=condition_id,
        group="cxr",
        exam_types=(ExamType.CXR,),
        system="teste",
        code=CodedConcept(scheme="99MEDAI", value=condition_id, meaning=condition_id),
    )


def _meta(pixels: np.ndarray, **overrides: Any) -> ImageMeta:
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    base: dict[str, Any] = {"source_format": "raster", "content_sha256": digest}
    base.update(overrides)
    return ImageMeta.model_validate(base)


def projection(
    level: float = 100.0,
    shape: tuple[int, int] = (64, 64),
    modality: Modality = Modality.DX,
    **meta: Any,
) -> ImageData:
    """Rampa determinística centrada em ``level`` — plumbing, não desempenho."""
    ramp = np.linspace(-20.0, 20.0, shape[1], dtype=np.float32)
    pixels = np.clip(np.tile(ramp, (shape[0], 1)) + level, 0, 255).astype(np.float32)
    return ImageData(
        pixels=pixels,
        kind=ImageKind.PROJECTION,
        modality=modality,
        meta=_meta(pixels, **{"body_part": "CHEST", **meta}),
        spacing_mm=(0.2, 0.2),
        value_range=(0.0, 255.0),
    )


def build_catalog(levels: dict[tuple[str, str, str], EvidenceLevel] | None = None) -> Catalog:
    levels = levels or {}
    ids = {key[0] for key in levels} | {"cxr.bright", "mg.density", "mg.asymmetry"}
    conditions = [condition(cid) for cid in sorted(ids | {"cxr.bright_large"})]
    entries = tuple(
        LevelEntry(
            condition_id=cid,
            model_id=mid,
            label=label,
            level=level,
            visibility=visibility_for(level),
            basis=() if level is EvidenceLevel.D else ("registro-teste",),
        )
        for (cid, mid, label), level in levels.items()
    )
    return Catalog(
        conditions,
        LevelsFile(protocol_id="teste", evidence_digest="0" * 64, entries=entries),
    )
