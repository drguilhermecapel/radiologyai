"""Utilidades compartilhadas pelos adaptadores de mamografia (MG).

Conversão para 8 bits (fiel ao ``load_image_from_dicom`` do mammoscreen)
------------------------------------------------------------------------
1. Desfaz a inversão MONOCHROME1 do carregador comum (``ImageData`` está sempre
   em MONOCHROME2; a inversão foi ``lo + hi - v`` pela faixa representável).
2. Aplica a VOI do arquivo com o próprio ``pydicom`` (janela LINEAR /
   LINEAR_EXACT / SIGMOID ou VOI LUT), a partir de ``extra["dicom_voi"]``.
3. MONOCHROME1: ``max - v``; depois min-max e ``(v * 255).astype(uint8)``.

DICOM sem ``extra["dicom_voi"]`` é RECUSADO (a função VOI não é conhecida e
uma janela linear no lugar de uma sigmoide mudaria a entrada em silêncio):
carregue com :func:`radiologyai.modalities.mg.loader.load_mammogram`.
Raster (PNG/JPEG) de 8 bits entra como está — é o uso do cartão do modelo
(``cv2.imread(..., IMREAD_GRAYSCALE)``).

Papéis e aplicabilidade
-----------------------
Os modelos por mama aceitam de 1 a 4 incidências (``L-CC``, ``L-MLO``,
``R-CC``, ``R-MLO``). O portão estático do manifesto exige TODOS os papéis de
``inputs.roles``; :func:`mg_applicability` aplica o mesmo portão só às imagens
com papel reconhecido e exige ao menos uma mama com uma incidência.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.xr_common import read_verified, require_backend
from radiologyai.analysis.base import ModelInputs, static_applicability
from radiologyai.analysis.routing import role_of
from radiologyai.analysis.types import AbstentionReason, ImageData, QACheck, QAReport
from radiologyai.errors import UnsupportedInputError
from radiologyai.modalities.mg.voi import VOI_EXTRA_KEY, apply_voi

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.analysis.base import ResolvedWeights
    from radiologyai.analysis.manifest import ModelManifest

__all__ = [
    "MG_EXTRA",
    "MG_ROLES",
    "breast_views",
    "load_safetensors_verified",
    "mg_applicability",
    "require_mg_backend",
    "role_inputs",
    "to_uint8",
]

MG_ROLES: tuple[str, ...] = ("L-CC", "L-MLO", "R-CC", "R-MLO")
MG_EXTRA = "ml,foundation,mammography"


def require_mg_backend(*extra_modules: str) -> None:
    """torch + timm + safetensors + OpenCV (resize bit a bit do original).

    Raises:
        BackendUnavailableError: dependência ausente; a mensagem nomeia os extras.
    """
    require_backend("torch", "timm", "safetensors", "cv2", *extra_modules, extra=MG_EXTRA)


def load_safetensors_verified(weights: ResolvedWeights, key: str) -> dict[str, Any]:
    """``state_dict`` de um safetensors a partir dos MESMOS bytes conferidos por sha256."""
    from safetensors.torch import load as load_bytes

    state: dict[str, Any] = load_bytes(read_verified(weights, key))
    return state


def _integral(arr: npt.NDArray[Any]) -> bool:
    import numpy as np

    return bool(np.array_equal(arr, np.round(arr)))


def to_uint8(image: ImageData) -> npt.NDArray[np.uint8]:
    """Imagem 8 bits como o ``load_image_from_dicom`` do original (ver módulo).

    Raises:
        UnsupportedInputError: colorida, não 2D, DICOM sem VOI completa,
            raster que não é 8 bits, ou imagem constante.
    """
    import numpy as np

    if image.color or image.pixels.ndim != 2:
        raise UnsupportedInputError("mamografia exige plano 2D em tons de cinza")
    if not np.isfinite(image.pixels).all():
        raise UnsupportedInputError("mamografia com pixels não finitos")
    fmt = image.meta.source_format
    if fmt == "raster":
        if image.value_range != (0.0, 255.0) or not _integral(image.pixels):
            raise UnsupportedInputError("raster de mamografia deve ser 8 bits (0-255)")
        out8: npt.NDArray[np.uint8] = image.pixels.astype(np.uint8)
        return out8
    if fmt != "dicom":
        raise UnsupportedInputError(f"mamografia de origem {fmt!r} não suportada")
    voi = image.extra.get(VOI_EXTRA_KEY)
    if not isinstance(voi, Mapping):
        raise UnsupportedInputError(
            "DICOM de mamografia sem parâmetros VOI completos (função LINEAR/SIGMOID, "
            "VOILUTSequence); carregue com radiologyai.modalities.mg.loader.load_mammogram"
        )
    lo, hi = image.value_range
    values = image.pixels.astype(np.float64)
    photometric = str(voi.get("photometric", ""))
    if photometric == "MONOCHROME1":
        values = (lo + hi) - values  # desfaz a inversão do carregador comum
    arr = apply_voi(values, dict(voi))
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr - arr.min()
    peak = float(arr.max())
    if peak <= 0.0:
        raise UnsupportedInputError("imagem constante: nada a analisar")
    arr = arr / peak
    out: npt.NDArray[np.uint8] = (arr * 255).astype(np.uint8)
    return out


def role_inputs(inputs: ModelInputs) -> ModelInputs:
    """Só as imagens com papel de mamografia reconhecido (``L-CC`` ... ``R-MLO``).

    Usa ``inputs.roles`` quando o pipeline já os resolveu; senão, ``role_of``.
    """
    roles: dict[str, int] = {}
    source = dict(inputs.roles)
    if not source:
        for idx, image in enumerate(inputs.images):
            role = role_of(image)
            if role is not None and role not in source:
                source[role] = idx
    images: list[ImageData] = []
    for role in MG_ROLES:
        if role in source:
            roles[role] = len(images)
            images.append(inputs.images[source[role]])
    return ModelInputs(
        images=tuple(images),
        exam_type=inputs.exam_type,
        roles=roles,
        upstream=inputs.upstream,
        prompts=inputs.prompts,
    )


def breast_views(inputs: ModelInputs) -> dict[str, list[tuple[str, int]]]:
    """``{"L"|"R": [(incidência, índice), ...]}`` na ordem CC, MLO."""
    out: dict[str, list[tuple[str, int]]] = {}
    for role in MG_ROLES:
        if role in inputs.roles:
            lat, view = role.split("-")
            out.setdefault(lat, []).append((view, inputs.roles[role]))
    return out


def _content_checks(image: ImageData, idx: int) -> tuple[list[QACheck], list[AbstentionReason]]:
    checks: list[QACheck] = []
    reasons: list[AbstentionReason] = []
    intent = (image.meta.presentation_intent or "").upper()
    ok_intent = "PROCESSING" not in intent
    checks.append(
        QACheck(
            name=f"presentation_intent[{idx}]",
            passed=ok_intent,
            value=image.meta.presentation_intent,
            detail=None
            if ok_intent
            else "imagem FOR PROCESSING (crua); modelo usa FOR PRESENTATION",
        )
    )
    if not ok_intent:
        reasons.append(AbstentionReason.ACQUISITION)
    fmt = image.meta.source_format
    ok_voi = fmt == "raster" or (
        fmt == "dicom" and isinstance(image.extra.get(VOI_EXTRA_KEY), Mapping)
    )
    checks.append(
        QACheck(
            name=f"voi_parameters[{idx}]",
            passed=ok_voi,
            detail=None
            if ok_voi
            else "VOI do DICOM desconhecida; carregue com modalities.mg.loader.load_mammogram",
        )
    )
    if not ok_voi:
        reasons.append(AbstentionReason.INPUT)
    ok_gray = not image.color
    if not ok_gray:
        checks.append(QACheck(name=f"grayscale[{idx}]", passed=False))
        reasons.append(AbstentionReason.INPUT)
    return checks, reasons


def mg_applicability(manifest: ModelManifest, inputs: ModelInputs) -> QAReport:
    """Portão do manifesto aplicado às imagens com papel + verificações de conteúdo.

    Exige ao menos uma imagem com papel; papéis ausentes NÃO recusam (o modelo
    aceita de 1 a 4 incidências, uma ou duas por mama).
    """
    selected = role_inputs(inputs)
    present = tuple(role for role in MG_ROLES if role in selected.roles)
    if not present:
        detail = "nenhuma imagem MG com lateralidade L/R e incidência CC/MLO"
        return QAReport.reject(
            AbstentionReason.MISSING_INPUT,
            checks=(QACheck(name="mg_roles", passed=False, detail=detail),),
        )
    relaxed = manifest
    if manifest.inputs.mode == "multi":
        relaxed = manifest.model_copy(
            update={"inputs": manifest.inputs.model_copy(update={"roles": present})}
        )
    base = static_applicability(relaxed, selected)
    checks = [*base.checks, QACheck(name="mg_roles", passed=True, value=",".join(present))]
    reasons = list(base.reasons)
    for idx, image in enumerate(selected.images):
        extra_checks, extra_reasons = _content_checks(image, idx)
        checks.extend(extra_checks)
        reasons.extend(r for r in extra_reasons if r not in reasons)
    if reasons:
        return QAReport.reject(*reasons, checks=tuple(checks))
    return QAReport.ok(tuple(checks))


def torch_state_from_bytes(data: bytes) -> Any:
    """``torch.load(weights_only=True)`` sobre bytes já verificados."""
    import torch

    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
