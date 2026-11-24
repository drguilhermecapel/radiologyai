"""Política de abstenção — o controle de risco primário do sistema.

Perigos H-01 (falso negativo em achado crítico) e H-02 (viés de automação),
ver ROADMAP §6.3 e REG-01 §4.

O sistema legado sempre emitia um rótulo. Sem pesos, emitia um rótulo derivado
de heurística OpenCV. **Emitir sempre é o problema**: um leitor humano exposto a
uma saída confiante e errada perde achados que veria sozinho.

Aqui a saída tem três bandas, e a banda do meio — *não avaliável* — é uma
resposta legítima, não uma falha. Duas regras estruturais adicionais:

1. O sistema **nunca afirma normalidade**. "Achado improvável" não é "exame
   normal": a ausência de evidência de um achado não é evidência de ausência.
2. Um achado crítico exige limiar de abstenção mais largo, porque o custo de um
   falso negativo domina o de um falso positivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.errors import EvaluationError
from radiologyai.inference.types import Band, Finding

if TYPE_CHECKING:
    from collections.abc import Sequence

# Achados cuja perda pode levar a deterioração grave ou morte. Recebem banda de
# indeterminação mais larga: na dúvida, o sistema se abstém em vez de negar.
CRITICAL_FINDINGS: frozenset[str] = frozenset(
    {"Pneumothorax", "Mass", "Nodule", "Consolidation", "Edema", "Effusion"}
)


@dataclass(frozen=True)
class BandThresholds:
    """Limiares de uma banda de decisão para um rótulo.

    Args:
        lower: abaixo disso, *achado improvável*.
        upper: acima disso, *achado provável*.
        entre os dois: *não avaliável*.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError(
                f"limiares inválidos: exige 0 <= lower ({self.lower}) <= upper ({self.upper}) <= 1"
            )

    @property
    def indeterminate_width(self) -> float:
        return self.upper - self.lower

    def classify(self, score: float) -> Band:
        if score >= self.upper:
            return Band.LIKELY
        if score <= self.lower:
            return Band.UNLIKELY
        return Band.INDETERMINATE


@dataclass(frozen=True)
class AbstentionPolicy:
    """Política de abstenção por rótulo.

    Args:
        thresholds: limiares por rótulo.
        calibrated: se os escores de entrada foram calibrados. Quando False, o
            resultado marca ``calibrated=False`` em todo achado, e a IFU deve
            declarar que os escores não são probabilidade de doença.
    """

    thresholds: dict[str, BandThresholds]
    calibrated: bool = False

    @classmethod
    def from_operating_points(
        cls,
        operating_points: dict[str, float],
        *,
        margin: float = 0.10,
        critical_margin: float = 0.20,
        calibrated: bool = False,
    ) -> AbstentionPolicy:
        """Constrói a política a partir dos pontos de operação medidos.

        A banda indeterminada é centrada no limiar que atinge a sensibilidade-alvo
        medida em ``artifacts/eval/``. Achados críticos recebem margem maior.

        Args:
            operating_points: ``rótulo -> limiar`` vindo de uma avaliação real.
            margin: meia-largura da banda para achados não críticos.
            critical_margin: meia-largura para achados críticos.
        """
        if not operating_points:
            raise EvaluationError(
                "política de abstenção exige pontos de operação medidos. "
                "Limiares inventados não são política de risco."
            )

        thresholds: dict[str, BandThresholds] = {}
        for label, threshold in operating_points.items():
            m = critical_margin if label in CRITICAL_FINDINGS else margin
            thresholds[label] = BandThresholds(
                lower=max(0.0, threshold - m), upper=min(1.0, threshold + m)
            )
        return cls(thresholds=thresholds, calibrated=calibrated)

    def classify(self, label: str, score: float) -> Band:
        """Banda de um escore. Rótulo sem limiar medido é sempre indeterminado."""
        if label not in self.thresholds:
            return Band.INDETERMINATE
        return self.thresholds[label].classify(score)

    def apply(
        self,
        labels: Sequence[str],
        scores: Sequence[float],
        *,
        evaluated: Sequence[bool] | None = None,
    ) -> tuple[Finding, ...]:
        """Converte escores em achados com banda.

        Args:
            evaluated: por rótulo, se existe medição de desempenho válida.
                Um rótulo sem medição é forçado a *não avaliável*, mesmo que o
                escore seja alto — não há base para afirmar nada sobre ele.
        """
        if len(labels) != len(scores):
            raise EvaluationError(f"{len(labels)} rótulos para {len(scores)} escores")
        flags = list(evaluated) if evaluated is not None else [True] * len(labels)

        findings = []
        for label, score, is_evaluated in zip(labels, scores, flags, strict=True):
            band = self.classify(label, float(score)) if is_evaluated else Band.INDETERMINATE
            findings.append(
                Finding(
                    label=label,
                    score=float(score),
                    band=band,
                    calibrated=self.calibrated,
                    evaluated=bool(is_evaluated),
                )
            )
        return tuple(findings)

    def abstention_rate(self, findings: Sequence[Finding]) -> float:
        """Fração de achados em banda indeterminada."""
        if not findings:
            return 0.0
        n = sum(1 for f in findings if f.band is Band.INDETERMINATE)
        return n / len(findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "critical_findings": sorted(CRITICAL_FINDINGS),
            "thresholds": {
                label: {
                    "lower": round(t.lower, 4),
                    "upper": round(t.upper, 4),
                    "critical": label in CRITICAL_FINDINGS,
                }
                for label, t in sorted(self.thresholds.items())
            },
        }


def summarize(findings: Sequence[Finding]) -> str:
    """Resumo textual das bandas, para o laudo.

    Nunca produz a palavra "normal". A ausência de achado provável não é exame
    normal, e o texto do laudo não pode sugerir que seja (REG-01 §4).
    """
    likely = [f.label for f in findings if f.band is Band.LIKELY]
    indeterminate = [f.label for f in findings if f.band is Band.INDETERMINATE]

    parts: list[str] = []
    if likely:
        parts.append(f"Achados prováveis: {', '.join(sorted(likely))}.")
    if indeterminate:
        parts.append(f"Não avaliável por baixa confiança: {', '.join(sorted(indeterminate))}.")
    if not likely:
        parts.append(
            "Nenhum achado atingiu o limiar de probabilidade. "
            "Isto NÃO constitui afirmação de exame normal."
        )
    parts.append("Rascunho gerado por IA. Requer revisão e validação médica.")
    return " ".join(parts)
