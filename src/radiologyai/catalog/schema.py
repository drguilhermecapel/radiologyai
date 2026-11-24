"""Esquema do catálogo de condições e dos registros de evidência.

O catálogo responde a uma única pergunta para o produto inteiro: *para esta
condição e este modelo, qual é a força da evidência?* A resposta (níveis A-D)
nunca é digitada: :func:`compute_level` a deriva dos registros de evidência, e
``levels.json`` é um arquivo gerado.

Três regras estruturais sustentam isso:

1. Nenhum modelo de entrada (``Condition``, ``EvidenceRecord``) possui campo de
   nível, e todos rejeitam chaves desconhecidas. Não há onde digitar um nível.
2. ``LevelEntry`` recusa combinações incoerentes: nível D fora do modo pesquisa,
   ou "refutado por medição" com nível diferente de D.
3. Uma medição própria limpa e reprovada rebaixa a condição a D mesmo que exista
   publicação favorável. Medição nossa prevalece sobre alegação de terceiros.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from radiologyai.errors import EvidenceError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EvidenceKind = Literal[
    "measured_external",  # medição nossa; modelo não treinado neste dataset
    "measured_internal",  # medição nossa; hold-out por paciente do dataset de treino
    "published",  # validação de terceiros revisada por pares, citação conferida na fonte
    "guideline_threshold",  # limiar de diretriz aplicado a uma medida validada
]
Leakage = Literal["none", "possible", "known"]
ReferenceStandard = Literal[
    "radiologist_adjudicated",
    "radiologist_single_read",
    "report_nlp",
    "histopathology",
    "expert_segmentation",
    "clinical_reference_measurement",
    "other",
]
Endpoint = Literal["discrimination", "detection", "agreement", "segmentation", "threshold"]
Visibility = Literal["main", "research"]


class EvidenceLevel(StrEnum):
    """Nível de evidência de um trio (condição, modelo, rótulo). Sempre calculado."""

    A = "A"  # validado externamente por medição nossa
    B = "B"  # validação publicada revisada por pares e/ou limiar de diretriz
    C = "C"  # validado internamente por medição nossa
    D = "D"  # exploratório (zero-shot ou sem validação por condição)


class ExamType(StrEnum):
    """Tipo de exame ao qual uma condição se aplica (chave de roteamento)."""

    CXR = "cxr"
    MSK_XR = "msk_xr"
    CT_HEAD = "ct_head"
    CT_CHEST = "ct_chest"
    CT_ABDOMEN = "ct_abdomen"
    CT_OTHER = "ct_other"
    MR_BRAIN = "mr_brain"
    MR_BODY = "mr_body"
    MG = "mg"
    US_BREAST = "us_breast"
    US_ABDOMEN = "us_abdomen"
    US_OTHER = "us_other"
    ECHO = "echo"
    UNKNOWN = "unknown"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


def _require_sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.match(value):
        raise ValueError(
            f"{field_name} deve ser 64 caracteres hexadecimais minúsculos; "
            f"recebido {value!r}. Placeholders são rejeitados por projeto."
        )
    return value


class CodedConcept(_Strict):
    """Termo codificado: esquema de codificação, valor e significado."""

    scheme: str
    value: str
    meaning: str


# --------------------------------------------------------------------------- #
# Protocolo pré-registrado
# --------------------------------------------------------------------------- #
class Protocol(_Strict):
    """Critérios de aceitação fixados ANTES de qualquer medição (pré-registro)."""

    protocol_id: str
    frozen_on: str  # data ISO do commit que congelou o protocolo

    # elegibilidade de um registro medido
    min_positive: int = 30
    min_negative: int = 30
    # barra de discriminação para os níveis A e C
    auroc_min: float = 0.80
    auroc_ci_low_min: float = 0.75
    # desfechos de detecção avaliados em limiar fixo e pré-especificado
    sens_spec_ci_low_min: float = 0.80
    # barra de concordância para medidas contínuas (ex.: fração de ejeção)
    icc_min: float = 0.75
    # estatística
    ci_level: float = 0.95
    bootstrap_iterations: int = 2000
    bootstrap_cluster: Literal["patient"] = "patient"
    random_seed: int = 20261124
    # pontos de operação ajustados na metade de calibração, reportados na de teste
    calibration_fraction: float = 0.5
    rule_out_sensitivity_target: float = 0.90
    rule_in_specificity_target: float = 0.90
    # barra para evidência publicada por terceiros (nível B)
    published_auroc_min: float = 0.80
    published_sens_spec_min: float = 0.80
    published_min_positive: int = 30
    segmentation_dice_min: float = 0.85
    subgroups: tuple[str, ...] = ("sex", "age_band", "view")


# --------------------------------------------------------------------------- #
# Entradas do catálogo
# --------------------------------------------------------------------------- #
class ConditionSource(_Strict):
    """Modelo e rótulo que produzem escore para uma condição."""

    model_id: str
    label: str


class Condition(_Strict):
    """Uma condição do catálogo. Deliberadamente **sem** campo de nível."""

    id: str  # "<grupo>.<slug>", ex.: "cxr.pneumothorax"
    name_pt: str
    name_en: str
    group: Literal["cxr", "msk_xr", "ct", "mr", "mg", "us", "echo"]
    exam_types: tuple[ExamType, ...]
    system: str  # sistema orgânico / área clínica, para agrupar na interface
    code: CodedConcept
    icd10: str | None = None
    description_pt: str | None = None
    sources: tuple[ConditionSource, ...] = ()


class Metric(_Strict):
    """Valor de uma métrica com intervalo de confiança opcional."""

    value: float
    ci_low: float | None = None
    ci_high: float | None = None


class EvidenceSource(_Strict):
    """Origem de um número publicado. ``quote`` deve conter literalmente o valor."""

    title: str
    identifier: str | None = None  # DOI / PMID / id da diretriz
    url: str
    quote: str
    peer_reviewed: bool
    verified_at_source: bool = False  # ligado somente pela ferramenta de conferência


class MeasurementProvenance(_Strict):
    """O que é preciso para reproduzir uma medição nossa."""

    git_sha: str
    protocol_id: str
    dataset_id: str
    dataset_sha256: str  # hash do manifesto do dataset (lista de arquivos + hashes)
    weights_sha256: dict[str, str]
    command: str
    created_at: datetime

    @field_validator("dataset_sha256")
    @classmethod
    def _validate_dataset_sha256(cls, v: str) -> str:
        return _require_sha256(v, "dataset_sha256")

    @field_validator("weights_sha256")
    @classmethod
    def _validate_weights_sha256(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("weights_sha256 não pode ser vazio: medição exige pesos identificados")
        for name, digest in v.items():
            _require_sha256(digest, f"weights_sha256[{name!r}]")
        return v


class EvidenceRecord(_Strict):
    """Um registro de evidência sobre um trio (condição, modelo, rótulo).

    Não possui campo de nível: o nível é consequência dos registros, calculada
    por :func:`compute_level`.
    """

    id: str
    condition_id: str
    model_id: str
    label: str
    kind: EvidenceKind
    endpoint: Endpoint = "discrimination"
    dataset: str | None = None
    population: str | None = None
    reference_standard: ReferenceStandard | None = None
    n_total: int | None = None
    n_positive: int | None = None
    metrics: dict[str, Metric] = Field(default_factory=dict)  # auroc, auprc, dice ...
    leakage: Leakage = "none"
    leakage_note: str | None = None
    passes_protocol: bool | None = None  # decidido pelo avaliador, nunca à mão
    failure_reasons: tuple[str, ...] = ()
    source: EvidenceSource | None = None  # obrigatório para publicado / diretriz
    provenance: MeasurementProvenance | None = None  # obrigatório para medido
    caveats: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> Self:
        measured = self.kind in ("measured_external", "measured_internal")
        if measured and self.provenance is None:
            raise ValueError(
                "evidência medida exige proveniência (git sha, hashes do dataset e dos pesos)"
            )
        if measured and (self.n_total is None or self.n_positive is None):
            raise ValueError("evidência medida exige n_total e n_positive")
        if not measured and self.source is None:
            raise ValueError("evidência publicada / de diretriz exige fonte com citação conferida")
        if self.passes_protocol is False and not self.failure_reasons:
            raise ValueError("registro reprovado deve declarar failure_reasons")
        return self


class LevelEntry(_Strict):
    """Nível calculado de um trio (condição, modelo, rótulo)."""

    condition_id: str
    model_id: str
    label: str
    level: EvidenceLevel
    visibility: Visibility
    basis: tuple[str, ...]  # ids dos registros de evidência que justificam o nível
    refuted_by_measurement: bool = False
    headline: str | None = None  # resumo curto gerado a partir dos registros

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        expected = visibility_for(self.level)
        if self.visibility != expected:
            raise ValueError(
                f"nível {self.level.value} exige visibilidade {expected!r}; "
                f"recebido {self.visibility!r}. Achado exploratório não vai à visão principal."
            )
        if self.refuted_by_measurement and self.level is not EvidenceLevel.D:
            raise ValueError("condição refutada por medição própria só pode ter nível D")
        return self


class LevelsFile(_Strict):
    """Arquivo gerado ``levels.json``. Nunca editado à mão."""

    protocol_id: str
    evidence_digest: str  # sha256 sobre os registros de evidência usados, ordenados
    entries: tuple[LevelEntry, ...]

    @field_validator("evidence_digest")
    @classmethod
    def _validate_digest(cls, v: str) -> str:
        return _require_sha256(v, "evidence_digest")


# --------------------------------------------------------------------------- #
# A regra de nível
# --------------------------------------------------------------------------- #
def compute_level(records: list[EvidenceRecord]) -> tuple[EvidenceLevel, list[str], bool]:
    """Deriva o nível de evidência de um trio (condição, modelo, rótulo).

    Devolve ``(nível, ids_dos_registros_de_base, refutado_por_medição)``.

    Regras (pré-registradas):

    * **A** - ao menos um registro ``measured_external`` com ``leakage == "none"``
      passa no protocolo.
    * **refutado** - existem medições externas limpas e *todas* reprovam: a
      condição cai para **D** independentemente de evidência publicada. Nossa
      medição reprovada prevalece sobre alegação de terceiros.
    * **B** - senão, um registro ``published`` ou ``guideline_threshold``
      conferido na fonte e aprovado na barra de publicação.
    * **C** - senão, um ``measured_internal`` aprovado, ou medição externa
      aprovada cujo vazamento é "possible"/"known".
    * **D** - todo o resto.

    Raises:
        EvidenceError: os registros não pertencem todos ao mesmo trio. Misturar
            trios permitiria que a evidência de uma condição promovesse outra.
    """
    triples = {(r.condition_id, r.model_id, r.label) for r in records}
    if len(triples) > 1:
        raise EvidenceError(
            "compute_level recebe registros de um único trio (condição, modelo, rótulo); "
            f"recebidos {len(triples)}: {sorted(triples)}"
        )

    clean_external = [r for r in records if r.kind == "measured_external" and r.leakage == "none"]
    passing_clean = [r for r in clean_external if r.passes_protocol is True]
    if passing_clean:
        return EvidenceLevel.A, [r.id for r in passing_clean], False

    decided_clean = [r for r in clean_external if r.passes_protocol is not None]
    if decided_clean and all(r.passes_protocol is False for r in decided_clean):
        return EvidenceLevel.D, [r.id for r in decided_clean], True

    published = [
        r
        for r in records
        if r.kind in ("published", "guideline_threshold")
        and r.passes_protocol is True
        and r.source is not None
        and r.source.verified_at_source
    ]
    if published:
        return EvidenceLevel.B, [r.id for r in published], False

    internal = [
        r
        for r in records
        if r.passes_protocol is True
        and (
            r.kind == "measured_internal" or (r.kind == "measured_external" and r.leakage != "none")
        )
    ]
    if internal:
        return EvidenceLevel.C, [r.id for r in internal], False

    return EvidenceLevel.D, [], False


def visibility_for(level: EvidenceLevel) -> Visibility:
    """Nível D só aparece em modo pesquisa; A-C vão à visão principal."""
    return "research" if level is EvidenceLevel.D else "main"
