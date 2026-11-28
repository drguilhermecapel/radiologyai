"""Decisão a partir de pontos de operação com origem declarada.

Complementa :mod:`radiologyai.calibration.abstention` sem alterá-lo. Lá a banda
vem de limiares por rótulo; aqui cada limiar é um :class:`OperatingPoint` que
carrega o papel clínico e **de onde veio** (ajuste do protocolo, diretriz,
padrão de biblioteca).

Regra estrutural: um escore só vira "positivo"/"negativo" em relação a um ponto
de operação explícito e com origem. Sem ponto utilizável, a decisão é
``"score_only"`` — o escore é mostrado sem veredito, nunca com limiar inventado.

O ponto de operação deliberadamente **não** carrega métricas de desempenho.
Métrica vive em ``artifacts/eval/``; o ponto apenas aponta para ela via
``evidence_ref``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from radiologyai.errors import EvaluationError
from radiologyai.inference.types import Band

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

OperatingScale = Literal["logit", "sigmoid", "softmax", "calibrated", "measurement", "similarity"]
OperatingRole = Literal["rule_out", "rule_in", "balanced", "guideline"]
Decision = Literal[
    "positive",  # no ponto de inclusão (rule-in) ou além dele
    "negative",  # aquém do ponto de exclusão (rule-out)
    "indeterminate",  # entre exclusão e inclusão (banda de baixa confiança)
    "score_only",  # não há ponto de operação com origem; escore sem veredito
]

_BOUNDED_SCALES: frozenset[str] = frozenset({"sigmoid", "softmax", "calibrated"})

_BANDS: dict[Decision, Band] = {
    "positive": Band.LIKELY,
    "negative": Band.UNLIKELY,
    "indeterminate": Band.INDETERMINATE,
    "score_only": Band.INDETERMINATE,
}


class OperatingPoint(BaseModel):
    """Limiar de decisão com a sua origem. ``role`` dá o significado clínico."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    role: OperatingRole
    threshold: float
    on: OperatingScale
    direction: Literal["ge", "le"] = Field(
        default="ge", description="Positivo quando escore >= limiar ('ge') ou <= ('le')"
    )
    source: str = Field(
        description="Origem legível do limiar: ajuste do protocolo, DOI de diretriz, "
        "padrão de biblioteca"
    )
    evidence_ref: str | None = Field(
        default=None, description="Caminho#chave do artefato de avaliação, quando medido por nós"
    )

    @field_validator("threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"threshold deve ser finito; recebido {v!r}")
        return v

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "ponto de operação exige origem declarada (source); "
                "limiar sem origem não é política de risco"
            )
        return v


def _score_on(scores: Mapping[str, float], point: OperatingPoint) -> float:
    score = scores[point.on]
    if not math.isfinite(score):
        raise EvaluationError(
            f"escore não finito na escala {point.on!r}: {score!r}. "
            "Nenhuma decisão é emitida sobre escore inválido."
        )
    return score


def _passes(score: float, point: OperatingPoint) -> bool:
    return score >= point.threshold if point.direction == "ge" else score <= point.threshold


def decide(
    scores: Mapping[str, float], points: Sequence[OperatingPoint]
) -> tuple[Decision, list[OperatingPoint]]:
    """Transforma escores em decisão usando apenas pontos de operação com origem.

    ``scores`` mapeia o nome da escala ("sigmoid", "calibrated", "measurement", ...)
    ao valor nessa escala. Pontos cuja escala não está disponível são ignorados.

    * exclusão (rule_out) e inclusão (rule_in) presentes -> decisão em três vias;
      a zona entre os dois é ``"indeterminate"`` (banda de baixa confiança).
    * um único ponto utilizável -> decisão em duas vias.
    * nenhum ponto utilizável -> ``"score_only"``: escore exibido sem veredito.

    Devolve a decisão e os pontos efetivamente usados (vazio em ``"score_only"``).

    Raises:
        EvaluationError: escore não finito numa escala usada para decidir.
    """
    usable = [p for p in points if p.on in scores]
    if not usable:
        return "score_only", []
    rule_out = next((p for p in usable if p.role == "rule_out"), None)
    rule_in = next((p for p in usable if p.role == "rule_in"), None)
    single = next((p for p in usable if p.role in ("balanced", "guideline")), None)

    if rule_out is not None and rule_in is not None:
        if _passes(_score_on(scores, rule_in), rule_in):
            return "positive", [rule_out, rule_in]
        if not _passes(_score_on(scores, rule_out), rule_out):
            return "negative", [rule_out, rule_in]
        return "indeterminate", [rule_out, rule_in]

    point = single or rule_in or rule_out or usable[0]
    return ("positive" if _passes(_score_on(scores, point), point) else "negative"), [point]


def op_normalize(scores: Mapping[str, float], points: Sequence[OperatingPoint]) -> float | None:
    """Reescala um escore limitado para que 0.5 caia no ponto de operação primário.

    Devolve ``None`` quando não há ponto, quando a escala do ponto primário não é
    limitada a [0, 1] ou não está disponível, ou quando o limiar não está em (0, 1).
    O resultado é apenas uma reescala para exibição: NÃO é probabilidade de doença.

    Raises:
        EvaluationError: escore não finito na escala do ponto primário.
    """
    if not points:
        return None
    primary = next((p for p in points if p.role in ("balanced", "guideline")), None)
    primary = primary or next((p for p in points if p.role == "rule_in"), points[0])
    if primary.on not in _BOUNDED_SCALES or primary.on not in scores:
        return None
    score, threshold = _score_on(scores, primary), primary.threshold
    if primary.direction == "le":
        score, threshold = 1.0 - score, 1.0 - threshold
    if not 0.0 < threshold < 1.0:
        return None
    score = min(1.0, max(0.0, score))
    if score < threshold:
        return score / (2.0 * threshold)
    return 1.0 - (1.0 - score) / (2.0 * (1.0 - threshold))


def band_for(decision: Decision) -> Band:
    """Mapeia a decisão para a :class:`Band` usada no restante do pacote.

    ``"score_only"`` vira *não avaliável*, coerente com
    ``AbstentionPolicy.classify``: rótulo sem limiar medido é sempre indeterminado.
    """
    return _BANDS[decision]
