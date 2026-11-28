"""Esquema dos manifestos de modelo (um YAML por modelo em ``analysis/manifests/``).

CONTRATO. O manifesto é a fonte única sobre um modelo: de onde vêm os pesos e
o sha256 fixado, licença (código e pesos; uso comercial), a que exames se
aplica, que rótulos emite e a que condição do catálogo cada rótulo corresponde.

Deliberadamente NÃO carrega números de desempenho: evidência vive em
``artifacts/eval/`` e no catálogo, e o nível mostrado ao usuário é calculado a
partir dela. Chaves como ``accuracy``/``auc``/``sensitivity``/``specificity``
são rejeitadas em qualquer profundidade.

Regras estruturais:

* ``status: planned`` exige ``planned_reason`` e nunca é carregado; é a forma
  honesta de listar um modelo que ainda não roda de verdade (sem stub).
* ``status: available`` exige adaptador sob ``radiologyai.analysis.adapters.``
  e todos os pesos com sha256 fixado a partir dos bytes reais (``pin``).
* Códigos de rótulo: somente ``pydicom.sr.codedict`` (com o significado
  oficial) ou o esquema privado ``99MEDAI``.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from radiologyai.analysis.types import PRIVATE_SCHEME, ImageKind, Modality, ScoreKind
from radiologyai.calibration.decision import OperatingPoint
from radiologyai.catalog.schema import CodedConcept, ExamType, Visibility
from radiologyai.errors import ManifestError

__all__ = [
    "ALLOWED_ADAPTER_PREFIX",
    "FORBIDDEN_PERFORMANCE_KEYS",
    "Applicability",
    "InputSpec",
    "LabelOperatingPoint",
    "LabelSpec",
    "License",
    "ModelManifest",
    "Reference",
    "WeightFile",
    "load_manifest",
    "validate_code",
]

ALLOWED_ADAPTER_PREFIX = "radiologyai.analysis.adapters."
FORBIDDEN_PERFORMANCE_KEYS = frozenset(
    {"accuracy", "auc", "auroc", "sensitivity", "specificity", "ppv", "npv", "f1", "dice"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}[a-z0-9]$")
_ADAPTER = re.compile(r"^radiologyai\.analysis\.adapters\.[A-Za-z0-9_.]+:[A-Za-z_][A-Za-z0-9_]*$")


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


@lru_cache(maxsize=32)
def _scheme_index(scheme: str) -> dict[str, str] | None:
    """``{valor: significado}`` de um esquema do ``pydicom.sr.codedict``; None se não existe."""
    from pydicom.sr.codedict import Collection, codes

    if scheme not in codes.schemes():
        return None
    return {code.value: code.meaning for code in Collection(scheme).concepts.values()}


def validate_code(concept: CodedConcept) -> CodedConcept:
    """Confere um código contra ``pydicom.sr.codedict`` ou aceita o esquema ``99MEDAI``.

    Raises:
        ValueError: esquema desconhecido, valor inexistente no esquema ou
            significado diferente do oficial (código trocado é erro clínico).
    """
    if concept.scheme == PRIVATE_SCHEME:
        if not concept.value.strip() or not concept.meaning.strip():
            raise ValueError("código 99MEDAI exige value e meaning não vazios")
        return concept
    index = _scheme_index(concept.scheme)
    if index is None:
        raise ValueError(
            f"esquema {concept.scheme!r} não está em pydicom.sr.codedict; "
            f"use um código oficial ou o esquema privado {PRIVATE_SCHEME}"
        )
    meaning = index.get(concept.value)
    if meaning is None:
        raise ValueError(
            f"código {concept.scheme}:{concept.value} não existe em pydicom.sr.codedict"
        )
    if meaning.casefold() != concept.meaning.casefold():
        raise ValueError(
            f"significado de {concept.scheme}:{concept.value} é {meaning!r}, "
            f"não {concept.meaning!r}"
        )
    return concept


def _find_forbidden_keys(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key).lower() in FORBIDDEN_PERFORMANCE_KEYS:
                found.append(here)
            found.extend(_find_forbidden_keys(sub, here))
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            found.extend(_find_forbidden_keys(sub, f"{path}[{i}]"))
    return found


class WeightFile(_Strict):
    """Um arquivo de pesos. ``sha256`` é escrito por ``pin`` a partir dos bytes reais."""

    key: str
    url: str
    filename: str
    sha256: str = ""  # vazio somente em modelo planejado
    bytes: int = Field(default=0, ge=0)
    extract: Literal["none", "zip", "tar"] = "none"
    unsafe_pickle: bool = False  # checkpoint exige unpickling completo (weights_only=False)
    note: str | None = None

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if v and not _SHA256.match(v):
            raise ValueError("sha256 deve ter 64 hex minúsculos (ou vazio em modelo planejado)")
        return v

    @field_validator("filename")
    @classmethod
    def _plain_name(cls, v: str) -> str:
        if "/" in v or "\\" in v or v in ("", ".", "..") or v.startswith("."):
            raise ValueError("filename deve ser um nome de arquivo simples")
        return v

    @field_validator("url")
    @classmethod
    def _https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("url de pesos deve ser https://")
        return v


class License(_Strict):
    """Licença do código e dos pesos. Pesos não comerciais são permitidos e sinalizados."""

    code: str
    weights: str
    training_data_terms: str | None = None
    commercial_use: Literal["yes", "no", "unclear"]
    evidence: str | None = None  # onde a licença foi lida (URL / arquivo + citação)

    @model_validator(mode="after")
    def _no_copyleft_code(self) -> Self:
        if re.search(r"\b(A?GPL)", self.code, re.I) and not re.search(r"\bLGPL", self.code, re.I):
            raise ValueError("código GPL/AGPL é excluído por política do projeto")
        return self


class Applicability(_Strict):
    """Portão estático: exames, modalidades, tipos de imagem, incidências, população."""

    exam_types: tuple[ExamType, ...]
    modalities: tuple[Modality, ...]
    image_kinds: tuple[ImageKind, ...]
    views: tuple[str, ...] = ()  # vazio = qualquer
    min_pixels: int | None = Field(default=None, ge=1)  # menor lado no plano
    adult_only: bool = False
    acquisition: tuple[str, ...] = ()  # ex.: "non_contrast", "slice_thickness<=3"


class InputSpec(_Strict):
    """Entrada única ou múltipla (4 incidências de mamografia, sequências de RM, janelas de eco)."""

    mode: Literal["single", "multi"] = "single"
    roles: tuple[str, ...] = ()  # papéis obrigatórios para modo multi
    optional_roles: tuple[str, ...] = ()
    prompts: Literal["none", "optional", "required"] = "none"

    @model_validator(mode="after")
    def _roles(self) -> Self:
        if self.mode == "multi" and not self.roles:
            raise ValueError("entrada multi exige ao menos um papel obrigatório (roles)")
        if self.mode == "single" and (self.roles or self.optional_roles):
            raise ValueError("papéis só fazem sentido com mode: multi")
        return self


class LabelSpec(_Strict):
    """Um rótulo emitido pelo adaptador e a condição do catálogo a que corresponde."""

    key: str  # chave que o adaptador emite
    source_label: str  # nome usado pelo modelo / biblioteca de origem
    condition_id: str  # id no catálogo de condições
    kind: Literal["binary", "categorical", "derived"] = "binary"
    score_kind: ScoreKind = "sigmoid"
    code: CodedConcept
    measurement: str | None = None  # nome da medida, para rótulos derivados
    classes: tuple[str, ...] | None = None  # para rótulos categóricos

    @field_validator("code")
    @classmethod
    def _code(cls, v: CodedConcept) -> CodedConcept:
        return validate_code(v)

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.kind == "derived" and not self.measurement:
            raise ValueError("rótulo derivado deve nomear a medida que limiariza")
        if self.kind == "derived" and self.score_kind != "measurement":
            raise ValueError("rótulo derivado tem score_kind 'measurement'")
        if self.kind == "categorical" and not self.classes:
            raise ValueError("rótulo categórico deve listar as classes")
        if self.kind != "categorical" and self.classes:
            raise ValueError("classes só valem para rótulo categórico")
        return self


class LabelOperatingPoint(_Strict):
    """Ponto de operação publicado/da biblioteca para um rótulo (com origem)."""

    label: str
    point: OperatingPoint


class Reference(_Strict):
    title: str
    identifier: str | None = None  # DOI / arXiv / PMID
    url: str | None = None


Task = Literal[
    "classification",
    "segmentation",
    "measurement",
    "gate",
    "detection",
    "zero_shot",
    "encoder",
]
Capability = Literal["classify", "segment", "measure", "explain", "prompt", "gate", "detect"]
Route = Literal["main", "research", "interactive", "dependency", "none"]


class ModelManifest(_Strict):
    """Manifesto completo de um modelo."""

    id: str
    version: str
    name: str
    status: Literal["available", "planned", "retired"]
    adapter: str | None = None  # "radiologyai.analysis.adapters.<mod>:<Classe>"
    task: Task
    capabilities: tuple[Capability, ...]
    summary: str
    intended_use: Literal["research_use_only"] = "research_use_only"
    visibility_default: Visibility = "main"  # zero-shot / sem validação: "research"
    route: Route | None = None  # None => "research" se zero_shot, senão "main"

    source: dict[str, str] = Field(default_factory=dict)  # biblioteca, nome, revisão ...
    weights: tuple[WeightFile, ...] = ()
    license: License
    applicability: Applicability
    inputs: InputSpec = Field(default_factory=InputSpec)
    depends_on: tuple[str, ...] = ()  # ids de modelos cuja saída este consome

    training_data: tuple[str, ...] = ()  # chaves de dataset, para controle de vazamento
    labels: tuple[LabelSpec, ...] = ()
    operating_points: tuple[LabelOperatingPoint, ...] = ()
    limitations: tuple[str, ...] = ()
    references: tuple[Reference, ...] = ()
    planned_reason: str | None = None  # por que o modelo ainda não está disponível
    config: dict[str, Any] = Field(default_factory=dict)  # ajustes do adaptador

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError("id deve ser minúsculo [a-z0-9._-], de 3 a 64 caracteres")
        return v

    @model_validator(mode="after")
    def _rules(self) -> Self:
        forbidden = _find_forbidden_keys(self.config) + _find_forbidden_keys(self.source)
        if forbidden:
            raise ValueError(
                f"manifesto não carrega métricas de desempenho: {forbidden}. "
                "Métrica vive em artifacts/eval/ e no catálogo de evidência."
            )
        label_keys = [label.key for label in self.labels]
        if len(set(label_keys)) != len(label_keys):
            raise ValueError("chaves de rótulo duplicadas")
        for op in self.operating_points:
            if op.label not in label_keys:
                raise ValueError(f"ponto de operação para rótulo desconhecido {op.label!r}")
        weight_keys = [w.key for w in self.weights]
        if len(set(weight_keys)) != len(weight_keys):
            raise ValueError("chaves de pesos duplicadas")
        filenames = [w.filename for w in self.weights]
        if len(set(filenames)) != len(filenames):
            raise ValueError("nomes de arquivo de pesos duplicados")
        if self.id in self.depends_on:
            raise ValueError("modelo não pode depender de si mesmo")
        if self.task == "zero_shot" and (self.visibility_default == "main" or self.route == "main"):
            raise ValueError("modelos zero-shot são exclusivos do modo pesquisa")

        if self.status == "planned":
            if not self.planned_reason or not self.planned_reason.strip():
                raise ValueError("modelo planejado deve declarar planned_reason")
            return self
        if not self.adapter:
            raise ValueError("modelo disponível deve nomear um adaptador")
        if not _ADAPTER.match(self.adapter):
            raise ValueError(
                f"adapter deve ser 'módulo:Classe' sob {ALLOWED_ADAPTER_PREFIX!r}; "
                f"recebido {self.adapter!r}"
            )
        if self.status == "available":
            unpinned = [w.key for w in self.weights if not w.sha256]
            if unpinned:
                raise ValueError(
                    f"modelo disponível com pesos sem sha256 fixado: {unpinned}. "
                    "Rode o pin a partir dos bytes reais (radiologyai.analysis.registry.pin)."
                )
        return self

    # -- auxiliares ------------------------------------------------------- #
    @property
    def effective_route(self) -> Route:
        """Roteamento efetivo: explícito, ou derivado da tarefa."""
        if self.route is not None:
            return self.route
        return "research" if self.task == "zero_shot" else "main"

    def label(self, key: str) -> LabelSpec:
        """Especificação de um rótulo pela chave.

        Raises:
            KeyError: rótulo não declarado.
        """
        for spec in self.labels:
            if spec.key == key:
                return spec
        raise KeyError(key)

    def points_for(self, label_key: str) -> list[OperatingPoint]:
        """Pontos de operação declarados no manifesto para um rótulo."""
        return [op.point for op in self.operating_points if op.label == label_key]

    @property
    def is_pinned(self) -> bool:
        return all(w.sha256 for w in self.weights)


def parse_manifest(data: Any, *, origin: str = "<dict>") -> ModelManifest:
    """Valida um manifesto já carregado (dict).

    Raises:
        ManifestError: estrutura inválida.
    """
    if not isinstance(data, dict):
        raise ManifestError(f"{origin}: o manifesto deve ser um mapeamento YAML")
    forbidden = [k for k in data if str(k).lower() in FORBIDDEN_PERFORMANCE_KEYS]
    if forbidden:
        raise ManifestError(f"{origin}: campos de desempenho no manifesto: {forbidden}")
    try:
        return ModelManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestError(f"{origin}: {exc}") from exc


def load_manifest(path: Path) -> tuple[ModelManifest, str]:
    """Lê um arquivo de manifesto. Devolve o manifesto e o sha256 dos bytes do arquivo.

    Raises:
        ManifestError: YAML inválido, esquema violado ou nome de arquivo != id.
    """
    import yaml

    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path}: YAML inválido: {exc}") from exc
    manifest = parse_manifest(data, origin=str(path))
    if path.stem != manifest.id:
        raise ManifestError(f"{path}: o nome do arquivo deve ser o id do modelo {manifest.id!r}")
    return manifest, hashlib.sha256(raw).hexdigest()
