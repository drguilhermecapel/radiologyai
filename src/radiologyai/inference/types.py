"""Tipos de resultado de inferência — imutáveis e explícitos."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Band(StrEnum):
    """Banda de decisão em três níveis (ROADMAP §5, Fase 2).

    A banda intermediária é uma resposta legítima, não uma falha. É o controle
    de risco primário para os perigos H-01 (falso negativo em achado crítico) e
    H-02 (viés de automação).
    """

    LIKELY = "achado_provavel"
    INDETERMINATE = "nao_avaliavel"
    UNLIKELY = "achado_improvavel"


class Finding(BaseModel):
    """Um achado, com escore calibrado e banda de decisão."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    score: float = Field(ge=0.0, le=1.0)
    band: Band
    calibrated: bool = Field(
        description="False significa que o escore NÃO é probabilidade de doença"
    )
    evaluated: bool = Field(
        default=True,
        description="False quando o achado não tem medição de desempenho válida",
    )


class StudyResult(BaseModel):
    """Resultado completo para um exame.

    Carrega tudo o que a trilha de auditoria precisa para reconstruir a decisão:
    hash da entrada, identidade do modelo, sha256 dos pesos e a versão do código.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    card_id: str
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_version: str
    modality: str
    findings: tuple[Finding, ...]
    abstained: bool = Field(
        default=False, description="True quando o sistema recusou emitir predição"
    )
    abstention_reason: str | None = None

    @property
    def normality_asserted(self) -> Literal[False]:
        """O sistema nunca afirma normalidade (REG-01 §4).

        Ausência de achado provável não é exame normal. Esta propriedade existe
        para tornar a regra explícita e testável.
        """
        return False
