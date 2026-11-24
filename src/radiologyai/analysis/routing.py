"""Inferência do tipo de exame e roteamento de modelos.

CONTRATO (assinaturas e formato do ``routing.yaml``).

O roteamento é explícito. Por padrão (``derive_from_manifests: true``) cada
manifesto declara ``route`` (main / research / interactive / dependency / none)
e ``applicability.exam_types``; o ``routing.yaml`` só acrescenta ou exclui
modelos por tipo de exame. Assim um engenheiro de adaptador não precisa editar
um arquivo central para o seu modelo ser roteado.

Quando o tipo de exame não pode ser estabelecido pelos metadados, a resposta é
``ExamType.UNKNOWN`` e um humano escolhe — o sistema nunca adivinha em silêncio.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from radiologyai.analysis.types import ImageData, ImageKind, Modality
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import ConfigurationError

if TYPE_CHECKING:
    from radiologyai.analysis.registry import Registry

__all__ = [
    "ROUTING_FILE",
    "Route",
    "RoutingFile",
    "RoutingTable",
    "infer_exam_type",
    "load_routing",
    "role_of",
]

ROUTING_FILE = Path(__file__).parent / "routing.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Route(_Strict):
    """Modelos de um tipo de exame, por papel no produto."""

    main: tuple[str, ...] = ()  # condições podem atingir níveis A-C
    research: tuple[str, ...] = ()  # só quando o modo pesquisa é pedido
    interactive: tuple[str, ...] = ()  # só com prompt do usuário (ex.: caixa)


class RoutingTable(_Strict):
    """Tabela final de roteamento por tipo de exame."""

    exam_types: dict[ExamType, Route] = Field(default_factory=dict)

    def models_for(self, exam_type: ExamType, *, include_research: bool = False) -> list[str]:
        route = self.exam_types.get(exam_type, Route())
        return [*route.main, *(route.research if include_research else ())]

    def interactive_for(self, exam_type: ExamType) -> list[str]:
        return list(self.exam_types.get(exam_type, Route()).interactive)

    def all_model_ids(self) -> set[str]:
        ids: set[str] = set()
        for route in self.exam_types.values():
            ids.update(route.main, route.research, route.interactive)
        return ids

    def validate_against(self, registry: Registry) -> None:
        """Confere que todo modelo roteado existe no registro e se aplica ao exame.

        Raises:
            ConfigurationError: modelo desconhecido ou tipo de exame incompatível.
        """
        for exam_type, route in self.exam_types.items():
            for model_id in (*route.main, *route.research, *route.interactive):
                if not registry.has(model_id):
                    raise ConfigurationError(f"roteamento cita modelo desconhecido {model_id!r}")
                manifest = registry.manifest(model_id)
                if exam_type not in manifest.applicability.exam_types:
                    raise ConfigurationError(
                        f"{model_id} roteado para {exam_type.value}, mas o manifesto "
                        "não declara esse tipo de exame"
                    )


class RoutingFile(_Strict):
    """Formato do ``routing.yaml``."""

    derive_from_manifests: bool = True
    include: dict[ExamType, Route] = Field(default_factory=dict)  # somado ao derivado
    exclude: tuple[str, ...] = ()  # ids nunca roteados automaticamente


def derive_routing(registry: Registry, spec: RoutingFile | None = None) -> RoutingTable:
    """Tabela a partir dos manifestos disponíveis (``route`` + ``exam_types``)."""
    spec = spec or RoutingFile()
    table: dict[ExamType, dict[str, list[str]]] = {}

    def add(exam_type: ExamType, bucket: str, model_id: str) -> None:
        lists = table.setdefault(exam_type, {"main": [], "research": [], "interactive": []})
        if model_id not in lists[bucket]:
            lists[bucket].append(model_id)

    if spec.derive_from_manifests:
        for manifest in registry.manifests():
            route = manifest.effective_route
            if manifest.status != "available" or manifest.id in spec.exclude:
                continue
            if route not in ("main", "research", "interactive"):
                continue
            for exam_type in manifest.applicability.exam_types:
                add(exam_type, route, manifest.id)
    for exam_type, extra in spec.include.items():
        for bucket, ids in (
            ("main", extra.main),
            ("research", extra.research),
            ("interactive", extra.interactive),
        ):
            for model_id in ids:
                add(exam_type, bucket, model_id)
    routing = RoutingTable(
        exam_types={
            exam: Route(
                main=tuple(v["main"]),
                research=tuple(v["research"]),
                interactive=tuple(v["interactive"]),
            )
            for exam, v in table.items()
        }
    )
    routing.validate_against(registry)
    return routing


def load_routing(registry: Registry, path: Path = ROUTING_FILE) -> RoutingTable:
    """Carrega ``routing.yaml`` e monta a tabela sobre o registro.

    Raises:
        ConfigurationError: arquivo inválido ou roteamento inconsistente.
    """
    import yaml

    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = RoutingFile.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
    return derive_routing(registry, spec)


# --------------------------------------------------------------------------- #
# Inferência do tipo de exame a partir dos metadados de aquisição
# --------------------------------------------------------------------------- #
_CHEST = ("chest", "thorax", "torax", "tórax", "lung", "pulm", "pecho")
_HEAD = ("head", "brain", "skull", "cranio", "crânio", "cranium", "neuro", "encefalo", "encéfalo")
_ABDOMEN = (
    "abdom", "pelvis", "pelve", "liver", "figado", "fígado", "kidney", "renal", "urogra",
)  # fmt: skip
_MSK = (
    "hand", "wrist", "finger", "elbow", "forearm", "humerus", "shoulder", "clavicle",
    "knee", "ankle", "foot", "toe", "leg", "tibia", "fibula", "femur", "hip", "extremity",
    "mao", "mão", "punho", "dedo", "cotovelo", "antebraco", "antebraço", "ombro", "joelho",
    "tornozelo", "pé", "perna", "quadril", "bacia",
)  # fmt: skip
_BREAST = ("breast", "mama", "mamm")
_CARDIAC = ("echo", "cardiac", "heart", "tte", "ecocardio", "coração", "coracao")


def _text(image: ImageData) -> str:
    meta = image.meta
    parts = [meta.body_part, meta.study_description, meta.series_description]
    return " " + " ".join(p for p in parts if p).lower() + " "


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def infer_exam_type(images: Sequence[ImageData]) -> ExamType:
    """Tipo de exame pelos metadados DICOM; ``UNKNOWN`` quando não há certeza.

    Conservador por projeto: modalidades misturadas, texto ambíguo (tórax e
    abdome ao mesmo tempo numa radiografia) ou ausência de descrição dão
    ``UNKNOWN``, e a escolha passa a um humano.
    """
    if not images:
        return ExamType.UNKNOWN
    modalities = {image.modality for image in images}
    text = " ".join(_text(image) for image in images)

    if modalities <= {Modality.CR, Modality.DX}:
        chest, msk = _has(text, _CHEST), _has(text, _MSK)
        if chest and not msk:
            return ExamType.CXR
        if msk and not chest:
            return ExamType.MSK_XR
        return ExamType.UNKNOWN
    if modalities == {Modality.MG}:
        return ExamType.MG
    if modalities == {Modality.CT}:
        head, chest, abdomen = _has(text, _HEAD), _has(text, _CHEST), _has(text, _ABDOMEN)
        if head and not (chest or abdomen):
            return ExamType.CT_HEAD
        if chest and not (head or abdomen):
            return ExamType.CT_CHEST
        if abdomen and not head:
            return ExamType.CT_ABDOMEN
        if text.strip():
            return ExamType.CT_OTHER
        return ExamType.UNKNOWN
    if modalities == {Modality.MR}:
        if _has(text, _HEAD):
            return ExamType.MR_BRAIN
        return ExamType.MR_BODY if text.strip() else ExamType.UNKNOWN
    if modalities == {Modality.US}:
        cardiac, breast, abdomen = _has(text, _CARDIAC), _has(text, _BREAST), _has(text, _ABDOMEN)
        if sum((cardiac, breast, abdomen)) > 1:
            return ExamType.UNKNOWN
        if cardiac:
            return ExamType.ECHO
        if breast:
            return ExamType.US_BREAST
        if abdomen:
            return ExamType.US_ABDOMEN
        # Clipe sem descrição pode ser eco ou não: não se adivinha.
        if any(image.kind is ImageKind.CINE for image in images) or not text.strip():
            return ExamType.UNKNOWN
        return ExamType.US_OTHER
    return ExamType.UNKNOWN


def role_of(image: ImageData) -> str | None:
    """Papel de uma imagem para modelos de entrada múltipla.

    * Mamografia: ``"L-CC"``, ``"R-CC"``, ``"L-MLO"``, ``"R-MLO"``.
    * RM: ``meta.sequence_role`` em minúsculas (``"t1"``, ``"t1c"``, ``"flair"`` ...).
    * Eco/US: ``meta.view_position`` em minúsculas (``"a4c"``, ``"a2c"`` ...).
    """
    meta = image.meta
    if image.modality is Modality.MG:
        view = (meta.view_position or "").upper().replace(" ", "")
        if meta.laterality in ("L", "R") and view in ("CC", "MLO"):
            return f"{meta.laterality}-{view}"
        return None
    if meta.sequence_role:
        return meta.sequence_role.lower()
    if meta.view_position and image.modality is Modality.US:
        return meta.view_position.lower()
    return None
