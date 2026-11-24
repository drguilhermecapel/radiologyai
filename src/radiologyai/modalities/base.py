"""Contrato de plugin de modalidade — o ponto de extensão da plataforma.

Ver ROADMAP §4.1. Cada modalidade (RX, TC, RM, US) é uma unidade versionada e
validada separadamente, porque a ANVISA escopa registro por uso pretendido e
uso pretendido é por modalidade (REG-01 §2).

``validate()`` é um **controle de risco**, não conveniência. O maior modo de
falha silenciosa de IA radiológica em produção é entrada fora de distribuição:
incidência lateral num modelo PA, exame pediátrico, AP de leito, um scout de TC
confundido com radiografia. Tornar a rejeição um método obrigatório do plugin
dá ao arquivo de risco um gancho verificável (perigo H-03).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from radiologyai.errors import OutOfScopeError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.io.metadata import StudyMetadata

Dimensionality = Literal["2d", "3d", "cine"]


@dataclass(frozen=True)
class ValidationReport:
    """Resultado do gate de escopo de uma modalidade."""

    accepted: bool
    modality: str
    reasons: tuple[str, ...] = field(default=())

    def raise_if_rejected(self) -> None:
        """Levanta :class:`OutOfScopeError` quando a entrada foi recusada."""
        if not self.accepted:
            raise OutOfScopeError(
                f"entrada fora do uso pretendido para {self.modality}: " + "; ".join(self.reasons)
            )


class ModalityPlugin(ABC):
    """Base de todo adaptador de modalidade.

    Subclasses declaram ``code``, ``sop_classes`` e ``dimensionality`` como
    atributos de classe, e implementam o gate de escopo e o pré-processamento.
    """

    code: ClassVar[str]
    """DICOM (0008,0060) Modality — ex.: 'DX', 'CR', 'CT', 'MR', 'US'."""

    accepted_modalities: ClassVar[frozenset[str]]
    """Conjunto de valores de Modality que este plugin aceita."""

    dimensionality: ClassVar[Dimensionality]
    """Forma do dado: 2d, volume 3d, ou cine (2d + tempo)."""

    implemented: ClassVar[bool] = False
    """False enquanto a modalidade ainda não foi treinada e validada."""

    @abstractmethod
    def validate(self, metadata: StudyMetadata) -> ValidationReport:
        """Decide se o exame está dentro do uso pretendido declarado."""

    @abstractmethod
    def preprocess(
        self,
        array: npt.NDArray[np.floating[Any]],
        metadata: StudyMetadata,  # noqa: ARG002 - parte do contrato do plugin
    ) -> npt.NDArray[np.float32]:
        """Converte pixels na escala física para o tensor de entrada do modelo."""

    def describe(self) -> dict[str, Any]:
        """Descrição declarativa, usada pela API e pela CLI."""
        return {
            "code": self.code,
            "accepted_modalities": sorted(self.accepted_modalities),
            "dimensionality": self.dimensionality,
            "implemented": self.implemented,
        }


class UnimplementedModality(ModalityPlugin):
    """Base para modalidades declaradas mas ainda não validadas.

    A interface existe desde a Fase 1 para que o desenho seja provado multimodal
    e para que o caminho 3D exista no sistema de tipos — impedindo que a
    implementação de RX assente premissas 2D no código comum. Mas nenhuma
    predição é produzida: qualquer tentativa levanta ``NotImplementedError``.
    """

    implemented: ClassVar[bool] = False

    def validate(
        self,
        metadata: StudyMetadata,  # noqa: ARG002 - parte do contrato do plugin
    ) -> ValidationReport:
        return ValidationReport(
            accepted=False,
            modality=self.code,
            reasons=(
                f"modalidade {self.code} declarada mas não implementada nesta versão; "
                "nenhum modelo foi treinado ou validado para ela",
            ),
        )

    def preprocess(
        self,
        array: npt.NDArray[np.floating[Any]],
        metadata: StudyMetadata,  # noqa: ARG002 - parte do contrato do plugin
    ) -> npt.NDArray[np.float32]:
        raise NotImplementedError(
            f"pré-processamento de {self.code} não implementado. Ver ROADMAP.md §5, Fases 4 e 5."
        )
