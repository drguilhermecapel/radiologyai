"""Interface de política de segurança usada pelo pipeline.

CONTRATO. A implementação lastreada em artefatos (calibração ajustada e pontos
de operação do protocolo, em ``artifacts/eval/``) é de outro módulo; o
pipeline depende só deste protocolo.
"""

from __future__ import annotations

from typing import Protocol

from radiologyai.analysis.types import ScoreKind
from radiologyai.calibration.decision import OperatingPoint

__all__ = ["ManifestOnlyPolicy", "SafetyPolicy"]


class SafetyPolicy(Protocol):
    """Fonte de calibração e de pontos de operação ajustados sob o protocolo."""

    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: ScoreKind
    ) -> tuple[float, str] | None:
        """``(probabilidade_calibrada, calibration_id)`` ou ``None``.

        ``None`` significa que não existe artefato de calibração ajustado para
        este modelo e rótulo; o pipeline então não reporta probabilidade alguma.
        """
        ...

    def operating_points(self, model_id: str, label: str) -> list[OperatingPoint]:
        """Pontos de operação ajustados sob o protocolo de evidência.

        Lista vazia faz o pipeline recorrer aos pontos publicados no manifesto
        (por exemplo, os limiares da própria biblioteca, com origem).
        """
        ...


class ManifestOnlyPolicy:
    """Política sem artefatos ajustados: sem calibração; só pontos do manifesto."""

    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: ScoreKind
    ) -> tuple[float, str] | None:
        del model_id, label, raw_score, score_kind  # sem artefato de calibração
        return None

    def operating_points(self, model_id: str, label: str) -> list[OperatingPoint]:
        del model_id, label  # sem pontos ajustados: o pipeline usa os do manifesto
        return []
