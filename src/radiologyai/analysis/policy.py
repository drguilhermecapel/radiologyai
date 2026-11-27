"""Interface de política de segurança usada pelo pipeline.

CONTRATO. O pipeline depende só deste protocolo. A implementação lastreada em
evidência medida (pontos de operação do protocolo guardados nos registros de
``evidence/measured/``) é :class:`radiologyai.analysis.evidence_policy.EvidencePolicy`;
:class:`ManifestOnlyPolicy` é o padrão conservador sem evidência carregada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from radiologyai.analysis.types import ScoreKind
from radiologyai.calibration.decision import OperatingPoint
from radiologyai.catalog.schema import EvidenceLevel

__all__ = [
    "MEASURED_LEVELS",
    "ManifestOnlyPolicy",
    "PointsRequest",
    "PointsResolution",
    "SafetyPolicy",
]

# Níveis medidos por este projeto sob o protocolo: decidir com os pontos do
# manifesto atribuiria à decisão um desempenho medido em outro limiar.
MEASURED_LEVELS: frozenset[EvidenceLevel] = frozenset({EvidenceLevel.A, EvidenceLevel.C})


@dataclass(frozen=True)
class PointsRequest:
    """O que o pipeline sabe de um achado binário ao pedir pontos de operação.

    ``level`` é o nível EFETIVO (já rebaixado a D pelo manifesto quando for o
    caso); ``score_kind`` é a escala em que o adaptador emitiu o escore;
    ``weights_sha256`` são os hashes dos pesos declarados no manifesto.
    """

    condition_id: str
    model_id: str
    label: str
    level: EvidenceLevel
    score_kind: ScoreKind
    manifest_points: tuple[OperatingPoint, ...] = ()
    weights_sha256: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PointsResolution:
    """Pontos de operação escolhidos pela política e avisos para o resultado.

    ``points`` vazio faz o achado ficar ``score_only`` (escore sem veredito).
    """

    points: tuple[OperatingPoint, ...] = ()
    warnings: tuple[str, ...] = ()


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

    def resolve_points(self, request: PointsRequest) -> PointsResolution:
        """Pontos de operação com que o achado é decidido (vazio = ``score_only``).

        O pipeline ainda recusa, para os níveis A e C, qualquer ponto sem
        ``evidence_ref`` (ponto que não veio de um registro de evidência).
        """
        ...


class ManifestOnlyPolicy:
    """Política sem evidência carregada: sem calibração; pontos do manifesto só em B e D.

    Níveis A e C ficam ``score_only``: foram medidos em pontos do protocolo que
    esta política não conhece, e os pontos do manifesto nunca foram avaliados.
    """

    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: ScoreKind
    ) -> tuple[float, str] | None:
        del model_id, label, raw_score, score_kind  # sem artefato de calibração
        return None

    def resolve_points(self, request: PointsRequest) -> PointsResolution:
        if request.level in MEASURED_LEVELS:
            if not request.manifest_points:
                return PointsResolution()
            return PointsResolution(
                warnings=(
                    f"nível {request.level.value} foi medido em pontos de operação do protocolo "
                    "que a política não fornece; os pontos do manifesto não foram avaliados — "
                    "somente escore, sem decisão",
                )
            )
        return PointsResolution(points=request.manifest_points)
