"""Artefato de sonda linear: vetores em ``safetensors`` + metadados em JSON.

O artefato contém apenas o que é preciso para aplicar a sonda de forma
reprodutível: média e escala do padronizador, coeficientes, intercepto, o id do
codificador cujos *embeddings* a sonda consome, o hash dos dados de treino e os
pontos de operação ajustados nas predições fora-da-dobra do treino (com origem).
Nenhuma métrica de desempenho é gravada aqui.

``probe_logit`` devolve ``intercept + coef · (x - mean) / scale`` — a função de
decisão da regressão logística, em escala logit.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.errors import BackendUnavailableError, RadiologyAIError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "FORMAT_VERSION",
    "ProbeArtifact",
    "ProbeArtifactError",
    "load_probe",
    "probe_logit",
    "save_probe",
    "training_data_sha256",
]

FORMAT_VERSION = "radiologyai-linear-probe/1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN = {"accuracy", "auc", "auroc", "sensitivity", "specificity", "ppv", "npv", "f1"}


class ProbeArtifactError(RadiologyAIError):
    """Artefato de sonda inválido, inconsistente ou adulterado."""


@dataclass(frozen=True)
class ProbeArtifact:
    """Sonda linear pronta para aplicar."""

    label: str
    encoder_id: str
    embedding_dim: int
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    coef: npt.NDArray[np.float64]
    intercept: float
    training_data_sha256: str
    operating_points: tuple[dict[str, Any], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import numpy as np

        d = self.embedding_dim
        for name in ("mean", "scale", "coef"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.shape != (d,):
                raise ProbeArtifactError(f"{name} deve ter shape ({d},)")
            if not np.isfinite(arr).all():
                raise ProbeArtifactError(f"{name} contém valores não finitos")
        if not (self.scale > 0).all():
            raise ProbeArtifactError("scale deve ser estritamente positivo")
        if not math.isfinite(self.intercept):
            raise ProbeArtifactError("intercept não finito")
        if not _SHA.match(self.training_data_sha256):
            raise ProbeArtifactError("training_data_sha256 deve ter 64 hex minúsculos")
        bad = _FORBIDDEN & {k.lower() for k in self.meta}
        if bad:
            raise ProbeArtifactError(f"artefato não carrega métricas de desempenho: {sorted(bad)}")


def _require_safetensors() -> None:
    if importlib.util.find_spec("safetensors") is None:
        raise BackendUnavailableError(
            "safetensors não instalado; instale o extra: pip install 'radiologyai[foundation]'"
        )


def training_data_sha256(records: list[tuple[str, int]]) -> str:
    """Hash canônico do conjunto de treino: pares ``(sha256 da imagem, rótulo)`` ordenados."""
    canonical = json.dumps(sorted(records), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_probe(artifact: ProbeArtifact, stem: Path) -> tuple[Path, Path]:
    """Grava ``<stem>.safetensors`` e ``<stem>.json``; o JSON carrega o sha256 dos vetores."""
    _require_safetensors()
    import numpy as np
    from safetensors.numpy import save

    tensors = {
        "mean": np.ascontiguousarray(artifact.mean, dtype=np.float64),
        "scale": np.ascontiguousarray(artifact.scale, dtype=np.float64),
        "coef": np.ascontiguousarray(artifact.coef, dtype=np.float64),
        "intercept": np.array([artifact.intercept], dtype=np.float64),
    }
    blob = save(tensors, metadata={"format": FORMAT_VERSION})
    tensor_path = stem.with_suffix(".safetensors")
    json_path = stem.with_suffix(".json")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.write_bytes(blob)
    doc = {
        "format": FORMAT_VERSION,
        "label": artifact.label,
        "encoder_id": artifact.encoder_id,
        "embedding_dim": artifact.embedding_dim,
        "tensors_sha256": hashlib.sha256(blob).hexdigest(),
        "training_data_sha256": artifact.training_data_sha256,
        "operating_points": list(artifact.operating_points),
        "meta": artifact.meta,
    }
    json_path.write_text(
        json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tensor_path, json_path


def load_probe(tensor_bytes: bytes, json_bytes: bytes) -> ProbeArtifact:
    """Reconstrói a sonda a partir dos bytes (já conferidos pelo registro).

    Raises:
        ProbeArtifactError: formato desconhecido, hash dos vetores divergente do
            JSON ou conteúdo inconsistente.
    """
    _require_safetensors()
    import numpy as np
    from safetensors.numpy import load

    try:
        doc = json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeArtifactError(f"JSON da sonda ilegível: {exc}") from exc
    if doc.get("format") != FORMAT_VERSION:
        raise ProbeArtifactError(f"formato de sonda desconhecido: {doc.get('format')!r}")
    if hashlib.sha256(tensor_bytes).hexdigest() != doc.get("tensors_sha256"):
        raise ProbeArtifactError("sha256 dos vetores difere do registrado no JSON da sonda")
    tensors = load(tensor_bytes)
    intercept = tensors["intercept"]
    if intercept.shape != (1,):
        raise ProbeArtifactError("intercept deve ser um escalar")
    return ProbeArtifact(
        label=str(doc["label"]),
        encoder_id=str(doc["encoder_id"]),
        embedding_dim=int(doc["embedding_dim"]),
        mean=np.asarray(tensors["mean"], dtype=np.float64),
        scale=np.asarray(tensors["scale"], dtype=np.float64),
        coef=np.asarray(tensors["coef"], dtype=np.float64),
        intercept=float(intercept[0]),
        training_data_sha256=str(doc["training_data_sha256"]),
        operating_points=tuple(doc.get("operating_points", ())),
        meta=dict(doc.get("meta", {})),
    )


def probe_logit(artifact: ProbeArtifact, embeddings: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Escore logit da sonda para ``embeddings`` de shape ``(N, D)`` ou ``(D,)``.

    Raises:
        ProbeArtifactError: dimensão incompatível ou embedding não finito.
    """
    import numpy as np

    x = np.asarray(embeddings, dtype=np.float64)
    single = x.ndim == 1
    x = x[None] if single else x
    if x.ndim != 2 or x.shape[1] != artifact.embedding_dim:
        raise ProbeArtifactError(
            f"embedding de dimensão {x.shape} incompatível com a sonda ({artifact.embedding_dim})"
        )
    if not np.isfinite(x).all():
        raise ProbeArtifactError("embedding com valores não finitos")
    out: npt.NDArray[np.float64] = (
        (x - artifact.mean) / artifact.scale
    ) @ artifact.coef + artifact.intercept
    return out
