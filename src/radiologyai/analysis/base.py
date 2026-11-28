"""Protocolo de adaptador: a única forma de um modelo contribuir para uma análise.

CONTRATO.

Um adaptador embrulha UM modelo pré-treinado real (ou um algoritmo real que
deriva medidas das saídas de outros modelos). Devolve escores brutos, mapas de
rótulos e medidas — e nada mais. Nunca decide o que é "positivo", nunca inventa
limiar e nunca produz saída que não calculou a partir da entrada: calibração,
pontos de operação, abstenção e nível de evidência são do pipeline. Se o
modelo não pode rodar, o adaptador LEVANTA exceção; não substitui valores.

Este módulo não importa torch: ``load`` recebe o dispositivo como texto
(``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``) e o adaptador o converte.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from radiologyai.analysis.types import (
    AbstentionReason,
    ImageData,
    Primitive,
    QACheck,
    QAReport,
    ScoreKind,
)
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import UnsupportedInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.analysis.manifest import ModelManifest

__all__ = [
    "AdapterOutput",
    "ExplanationMap",
    "LabelMap",
    "ModelAdapter",
    "ModelInputs",
    "RawMeasurement",
    "ResolvedWeights",
    "Score",
    "static_applicability",
]


@dataclass(frozen=True, slots=True)
class ResolvedWeights:
    """Arquivos de pesos que passaram na verificação de integridade. Só o registro constrói."""

    files: Mapping[str, Path]  # chave -> arquivo, ou diretório extraído para arquivos compactados
    sha256: Mapping[str, str]
    unsafe_pickle: Mapping[str, bool] = field(default_factory=dict)

    def path(self, key: str) -> Path:
        """Caminho verificado de um peso.

        Raises:
            KeyError: chave não declarada no manifesto.
        """
        return self.files[key]


@dataclass(slots=True)
class Score:
    """Escore bruto de um rótulo binário declarado no manifesto."""

    label: str
    value: float
    kind: ScoreKind

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"escore não finito para {self.label!r}: {self.value!r}")


@dataclass(slots=True)
class LabelMap:
    """Mapa de rótulos inteiro na geometria de ``inputs.images[image_idx]``."""

    name: str
    array: npt.NDArray[np.integer[Any]]
    labels: dict[int, str]  # índice (>0) -> chave de rótulo; 0 é fundo
    image_idx: int = 0


@dataclass(slots=True)
class RawMeasurement:
    """Medida bruta. ``value=None`` exige ``abstention`` (nunca valor inventado)."""

    key: str  # nome estável, ex.: "cardiothoracic_ratio"
    value: float | None
    unit: str
    method: str
    geometry: list[Primitive] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)  # nomes de LabelMap
    preconditions: dict[str, str] = field(default_factory=dict)
    abstention: AbstentionReason | None = None
    note: str | None = None
    image_idx: int = 0

    def __post_init__(self) -> None:
        if self.value is None and self.abstention is None:
            raise ValueError(f"medida {self.key!r} sem valor exige motivo de abstenção")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"medida {self.key!r} não finita: {self.value!r}")


@dataclass(slots=True)
class ExplanationMap:
    """Mapa de explicação calculado pelo modelo real (CAM, Grad-CAM, atenção...)."""

    label: str
    heatmap: npt.NDArray[np.float32]  # 2D, valores em [0, 1], qualquer resolução
    placement: Primitive  # bbox em pixels da imagem de origem coberta pelo mapa
    method: Literal["cam", "gradcam", "attention", "occlusion"] = "cam"
    image_idx: int = 0


@dataclass(slots=True)
class AdapterOutput:
    """Tudo o que um adaptador devolve. Nada aqui é decisão clínica."""

    scores: list[Score] = field(default_factory=list)
    class_scores: dict[str, dict[str, float]] = field(default_factory=dict)  # rótulo->classe->p
    labelmaps: list[LabelMap] = field(default_factory=list)
    measurements: list[RawMeasurement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)  # só para adaptadores a jusante


@dataclass(frozen=True, slots=True)
class ModelInputs:
    """O que o pipeline entrega a um adaptador."""

    images: tuple[ImageData, ...]
    exam_type: ExamType
    roles: Mapping[str, int] = field(default_factory=dict)  # papel -> índice em images
    upstream: Mapping[str, AdapterOutput] = field(default_factory=dict)  # id -> saída
    prompts: tuple[Primitive, ...] = ()

    @property
    def single(self) -> ImageData:
        """A única imagem da entrada.

        Raises:
            UnsupportedInputError: não há exatamente uma imagem.
        """
        if len(self.images) != 1:
            raise UnsupportedInputError(
                f"esperada exatamente uma imagem; recebidas {len(self.images)}"
            )
        return self.images[0]

    def by_role(self, role: str) -> ImageData:
        """Imagem de um papel (ex.: ``"L-CC"``, ``"t1c"``, ``"a4c"``).

        Raises:
            UnsupportedInputError: papel ausente.
        """
        if role not in self.roles:
            raise UnsupportedInputError(f"papel de entrada obrigatório {role!r} ausente")
        return self.images[self.roles[role]]

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def upstream_output(self, model_id: str) -> AdapterOutput:
        """Saída de um modelo do qual este depende.

        Raises:
            UnsupportedInputError: o modelo a montante não produziu saída.
        """
        if model_id not in self.upstream:
            raise UnsupportedInputError(f"saída do modelo a montante {model_id!r} ausente")
        return self.upstream[model_id]


class ModelAdapter(ABC):
    """Base de todo adaptador. Subclasses vivem em ``radiologyai.analysis.adapters``."""

    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest
        self.device = "cpu"
        self.loaded = False

    # -- ciclo de vida ---------------------------------------------------- #
    @abstractmethod
    def load(self, weights: ResolvedWeights, device: str) -> None:
        """Constrói o modelo a partir de pesos já verificados. Sem rede aqui."""

    def unload(self) -> None:  # noqa: B027 - gancho opcional
        """Libera memória. Padrão: nada a fazer."""

    # -- inferência ------------------------------------------------------- #
    def applicability(self, inputs: ModelInputs) -> QAReport:
        """Decide se o modelo pode rodar nesta entrada.

        O padrão aplica o portão estático do manifesto. Sobrescreva para somar
        verificações de conteúdo (incidência, qualidade, fora de distribuição),
        chamando ``super().applicability(inputs)`` primeiro.
        """
        return static_applicability(self.manifest, inputs)

    @abstractmethod
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        """Roda o modelo real. Levanta em caso de falha; nunca devolve placeholder."""

    def explain(self, inputs: ModelInputs, labels: Sequence[str]) -> list[ExplanationMap]:
        """Explicações para as chaves de rótulo pedidas. Padrão: nenhuma disponível."""
        del inputs, labels  # padrão do contrato: modelo sem explicação
        return []


def static_applicability(manifest: ModelManifest, inputs: ModelInputs) -> QAReport:
    """Portão de aplicabilidade do manifesto, compartilhado por todos os adaptadores."""
    spec = manifest.applicability
    checks: list[QACheck] = []
    reasons: list[AbstentionReason] = []

    def reason(r: AbstentionReason) -> None:
        if r not in reasons:
            reasons.append(r)

    ok_exam = inputs.exam_type in spec.exam_types
    checks.append(QACheck(name="exam_type", passed=ok_exam, value=inputs.exam_type.value))
    if not ok_exam:
        reason(AbstentionReason.EXAM_TYPE)

    if not inputs.images:
        checks.append(QACheck(name="has_images", passed=False))
        reason(AbstentionReason.MISSING_INPUT)
        return QAReport(applicable=False, checks=tuple(checks), reasons=tuple(reasons))

    for idx, image in enumerate(inputs.images):
        ok_mod = image.modality in spec.modalities
        ok_kind = image.kind in spec.image_kinds
        checks.append(QACheck(name=f"modality[{idx}]", passed=ok_mod, value=image.modality.value))
        checks.append(QACheck(name=f"image_kind[{idx}]", passed=ok_kind, value=image.kind.value))
        if not ok_mod:
            reason(AbstentionReason.MODALITY)
        if not ok_kind:
            reason(AbstentionReason.INPUT)
        if spec.min_pixels is not None:
            shortest = min(image.plane_shape)
            ok_px = shortest >= spec.min_pixels
            checks.append(
                QACheck(
                    name=f"min_pixels[{idx}]",
                    passed=ok_px,
                    value=float(shortest),
                    threshold=float(spec.min_pixels),
                )
            )
            if not ok_px:
                reason(AbstentionReason.QUALITY)
        if spec.views:
            view = image.meta.view_position
            ok_view = view is not None and view.upper() in {v.upper() for v in spec.views}
            checks.append(QACheck(name=f"view[{idx}]", passed=ok_view, value=view))
            if not ok_view:
                reason(AbstentionReason.VIEW)
        if spec.adult_only:
            age = image.meta.patient_age_years
            ok_age = age is not None and age >= 18
            checks.append(
                QACheck(
                    name=f"adult[{idx}]",
                    passed=ok_age,
                    value=age,
                    threshold=18.0,
                    detail=None if age is not None else "idade ausente: população não verificável",
                )
            )
            if not ok_age:
                reason(AbstentionReason.POPULATION)

    if manifest.inputs.mode == "multi":
        missing = [role for role in manifest.inputs.roles if role not in inputs.roles]
        checks.append(
            QACheck(name="required_roles", passed=not missing, value=",".join(missing) or None)
        )
        if missing:
            reason(AbstentionReason.MISSING_INPUT)
    if manifest.inputs.prompts == "required" and not inputs.prompts:
        checks.append(QACheck(name="prompt", passed=False))
        reason(AbstentionReason.MISSING_INPUT)

    return QAReport(applicable=not reasons, checks=tuple(checks), reasons=tuple(reasons))
