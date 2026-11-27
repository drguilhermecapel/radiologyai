"""Auxiliares dos testes da plataforma (SOMENTE testes; nunca distribuídos).

Os dublês de adaptador calculam a saída a partir dos pixels REAIS da entrada
(média, limiar), para que os testes possam afirmar que a plataforma transporta
o que o "modelo" calculou — sem pesos reais e sem rede. Não medem desempenho.
"""

# ruff: noqa: ARG002 - dublês implementam a assinatura do contrato sem usar tudo
from __future__ import annotations

import hashlib
import importlib.util
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from radiologyai.analysis.base import (
    AdapterOutput,
    ExplanationMap,
    LabelMap,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
    ResolvedWeights,
    Score,
)
from radiologyai.analysis.registry import Registry
from radiologyai.analysis.types import Primitive
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
from radiologyai.config.settings import Settings

HAS_PLATFORM = all(
    importlib.util.find_spec(m) is not None
    for m in ("fastapi", "sqlalchemy", "alembic", "argon2", "httpx", "multipart", "highdicom")
)
FAKE_MODULE = "radiologyai.analysis.adapters._platform_test_fakes"
WEIGHT_BYTES = b"nao-e-um-modelo-real: fixture da plataforma\n"
WEIGHT_SHA = hashlib.sha256(WEIGHT_BYTES).hexdigest()
PASSWORD = "senha-de-teste-123"


class PixelMean(ModelAdapter):
    """Dois escores binários derivados da média real dos pixels + máscara + mapa."""

    def load(self, weights: ResolvedWeights, device: str) -> None:
        self.weights = weights

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        image = inputs.single
        lo, hi = image.value_range
        mean = float((image.pixels.mean() - lo) / (hi - lo))
        mask = (image.pixels > image.pixels.mean()).astype(np.uint8)
        return AdapterOutput(
            scores=[Score("bright", mean, "sigmoid"), Score("dark", 1.0 - mean, "sigmoid")],
            # ``bright_region`` liga-se ao achado ``bright`` (nível B); ``dark_region``
            # e a medida ``dark_fraction`` só ao rótulo ``dark`` (nível D): exploratórios.
            labelmaps=[
                LabelMap(name="bright_region", array=mask, labels={1: "bright"}),
                LabelMap(name="dark_region", array=(1 - mask).astype(np.uint8), labels={1: "dark"}),
            ],
            measurements=[
                RawMeasurement(
                    key="dark_fraction",
                    value=float(1.0 - mask.mean()),
                    unit="1",
                    method="fração de pixels abaixo da média (dublê)",
                    derived_from=["dark_region"],
                )
            ],
        )

    def explain(self, inputs: ModelInputs, labels: Sequence[str]) -> list[ExplanationMap]:
        image = inputs.single.pixels
        heat = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
        h, w = image.shape
        box = Primitive(type="bbox", xy=((0.0, 0.0), (float(w), float(h))))
        return [
            ExplanationMap(label=label, heatmap=heat.astype(np.float32), placement=box)
            for label in labels
        ]


def manifest(model_id: str = "fake-pixel-mean") -> dict[str, Any]:
    code = {"scheme": "99MEDAI", "value": "x", "meaning": "x"}
    return {
        "id": model_id,
        "version": "0.0-test",
        "name": model_id,
        "status": "available",
        "adapter": f"{FAKE_MODULE}:PixelMean",
        "task": "classification",
        "capabilities": ["classify", "segment", "explain"],
        "summary": "dublê de teste da plataforma",
        "route": "main",
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
            "exam_types": ["cxr"],
            "modalities": ["CR", "DX"],
            "image_kinds": ["projection"],
        },
        "labels": [
            {"key": "bright", "source_label": "Bright", "condition_id": "cxr.bright", "code": code},
            {"key": "dark", "source_label": "Dark", "condition_id": "cxr.dark", "code": code},
        ],
        "operating_points": [
            {
                "label": "bright",
                "point": {
                    "name": "teste",
                    "role": "rule_in",
                    "threshold": 0.3,
                    "on": "sigmoid",
                    "direction": "ge",
                    "source": "limiar de teste (fixture)",
                },
            }
        ],
    }


def research_manifest() -> dict[str, Any]:
    """Modelo de rota ``research``: só roda em modo pesquisa (nenhum nível no catálogo)."""
    data = manifest("fake-research-seg")
    data.update(route="research")
    return data


def planned_manifest() -> dict[str, Any]:
    data = manifest("fake-planned")
    data.update(status="planned", planned_reason="ainda não roda de verdade")
    return data


def build_registry(tmp: Path) -> Registry:
    manifest_dir, weights_dir = tmp / "manifests", tmp / "weights"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for data in (manifest(), research_manifest(), planned_manifest()):
        (manifest_dir / f"{data['id']}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        if data["status"] == "available":
            target = weights_dir / data["id"] / "main.bin"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(WEIGHT_BYTES)
    settings = Settings(weights_dir=weights_dir, device="cpu", max_loaded_models=4)
    return Registry(manifest_dir=manifest_dir, weights_dir=weights_dir, settings=settings)


def build_catalog() -> Catalog:
    """``cxr.bright`` é nível B (visão principal); ``cxr.dark`` é nível D (modo pesquisa)."""

    def cond(cid: str) -> Condition:
        return Condition(
            id=cid,
            name_pt=f"condição {cid}",
            name_en=cid,
            group="cxr",
            exam_types=(ExamType.CXR,),
            system="teste",
            code=CodedConcept(scheme="99MEDAI", value=cid, meaning=cid),
        )

    levels = {
        ("cxr.bright", "fake-pixel-mean", "bright"): EvidenceLevel.B,
        ("cxr.dark", "fake-pixel-mean", "dark"): EvidenceLevel.D,
    }
    entries = tuple(
        LevelEntry(
            condition_id=c,
            model_id=m,
            label=label,
            level=lvl,
            visibility=visibility_for(lvl),
            basis=() if lvl is EvidenceLevel.D else ("registro-teste",),
        )
        for (c, m, label), lvl in levels.items()
    )
    return Catalog(
        [cond("cxr.bright"), cond("cxr.dark")],
        LevelsFile(protocol_id="teste", evidence_digest="0" * 64, entries=entries),
    )


def dicom_bytes(dataset: Any) -> bytes:
    import pydicom

    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, dataset, enforce_file_format=True)
    return buffer.getvalue()


def png_bytes(array: np.ndarray, **text: str) -> bytes:
    from PIL import Image, PngImagePlugin

    info = PngImagePlugin.PngInfo()
    for key, value in text.items():
        info.add_text(key, value)
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()
