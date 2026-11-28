"""Modo silencioso (shadow) do piloto prospectivo — REQ-195, portão G4.

Com ``MEDAI_SHADOW_MODE=true`` a implantação inteira fica silenciosa: as
análises rodam, são gravadas imutáveis e auditadas exatamente como no modo
normal, mas a saída de IA só é servida a quem tem ``shadow:view`` (papel
``researcher``, a equipe do piloto). ``research:view`` NÃO é o portão: o
médico assistente o tem.

Regra única, aplicada rota a rota:

* recurso que é só saída de IA (quadro de segmentação, mapa de explicação ou
  máscara em ``/artifacts``, revisões, laudos e os seus PDF/SR/SEG/FHIR,
  criação e assinatura de laudo, ``/analyze`` legado) => **403** com
  ``error: "ShadowModeError"``, auditado como ``shadow.blocked``;
* recurso misto (estudo, lista de trabalho, estado da análise) => servido sem a
  saída de IA e com ``shadow: true`` (contagens ``None``, nunca ``0`` — vazio
  não é "exame normal");
* envio a sistema de cuidado (STOW-RS ao PACS) => 403 para TODOS, inclusive a
  equipe de pesquisa: nada sai em direção ao cuidado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from radiologyai.errors import RadiologyAIError
from radiologyai.persistence.security import Permission

if TYPE_CHECKING:
    from radiologyai.api.deps import Principal
    from radiologyai.api.schemas import AnalysisOut, AnalysisSummary
    from radiologyai.worker.runtime import Platform

__all__ = [
    "SHADOW_ERROR",
    "SHADOW_OFF_ERROR",
    "ShadowModeError",
    "ShadowModeOffError",
    "ai_visible",
    "audit_redaction",
    "block_ai",
    "block_care_export",
    "redact_analysis",
    "redact_summary",
    "shadow_on",
]

#: Código estável do erro (campo ``error`` de ``ErrorOut``).
SHADOW_ERROR = "ShadowModeError"

#: Tipos de artefato que são saída de IA (``input`` é o exame, não é).
AI_ARTIFACT_KINDS = frozenset({"labelmap", "explanation"})


#: Código estável do erro da exportação do piloto com o modo silencioso desligado.
SHADOW_OFF_ERROR = "ShadowModeOffError"


class ShadowModeError(RadiologyAIError):
    """Saída de IA retida pelo modo silencioso (HTTP 403, ``error: ShadowModeError``)."""


class ShadowModeOffError(RadiologyAIError):
    """Operação do piloto com o modo silencioso desligado (HTTP 409, ``ShadowModeOffError``)."""


def shadow_on(platform: Platform) -> bool:
    return bool(getattr(platform.settings, "shadow_mode", False))


def ai_visible(platform: Platform, principal: Principal) -> bool:
    """O usuário pode ver a saída de IA? Sempre no modo normal; só ``shadow:view`` no silencioso."""
    return not shadow_on(platform) or principal.can(Permission.SHADOW_VIEW)


def _audit(
    platform: Platform,
    event: str,
    principal: Principal,
    *,
    action: str,
    subject_type: str | None,
    subject_id: str | None,
    care_export: bool = False,
) -> None:
    from radiologyai.persistence.audit import append_event

    append_event(
        platform.db,
        event,
        actor_id=principal.user_id,
        actor_username=principal.username,
        subject_type=subject_type,
        subject_id=subject_id,
        payload={"action": action, "role": principal.role, "care_export": care_export},
    )


def block_ai(
    platform: Platform,
    principal: Principal,
    *,
    action: str,
    subject_type: str | None = None,
    subject_id: str | None = None,
) -> None:
    """403 auditado quando o modo silencioso retém a saída de IA deste usuário; senão, nada."""
    if ai_visible(platform, principal):
        return
    _audit(
        platform,
        "shadow.blocked",
        principal,
        action=action,
        subject_type=subject_type,
        subject_id=subject_id,
    )
    raise ShadowModeError(
        "modo silencioso (piloto): a saída de IA não é exibida ao cuidado; "
        "acesso restrito à equipe de pesquisa (shadow:view)"
    )


def block_care_export(
    platform: Platform,
    principal: Principal,
    *,
    action: str,
    subject_type: str | None = None,
    subject_id: str | None = None,
) -> None:
    """No modo silencioso nada sai para sistemas de cuidado (PACS/DICOMweb), para ninguém."""
    if not shadow_on(platform):
        return
    _audit(
        platform,
        "shadow.blocked",
        principal,
        action=action,
        subject_type=subject_type,
        subject_id=subject_id,
        care_export=True,
    )
    raise ShadowModeError(
        "modo silencioso (piloto): nenhum envio a sistemas de cuidado (PACS/DICOMweb)"
    )


def audit_redaction(platform: Platform, principal: Principal, *, analysis_id: str) -> None:
    """Registra que um resultado concluído foi servido sem a saída de IA."""
    _audit(
        platform,
        "shadow.redacted",
        principal,
        action="analysis.read",
        subject_type="analysis",
        subject_id=analysis_id,
    )


def redact_analysis(out: AnalysisOut) -> AnalysisOut:
    """Mesma análise sem nada que revele a saída de IA; ``shadow=True``."""
    return out.model_copy(
        update={
            "result": None,
            "result_sha256": None,
            "result_status": None,
            # A mensagem de erro do pipeline pode citar abstenção ou saída de modelo.
            "error": None,
            "research_view": False,
            "hidden_research_findings": 0,
            "finding_indices": None,
            "segmentation_indices": None,
            "hidden_research_artifacts": 0,
            "research_segmentation_indices": (),
            "research_measurement_names": (),
            "reviews": (),
            "shadow": True,
        }
    )


def redact_summary(summary: AnalysisSummary) -> AnalysisSummary:
    return summary.model_copy(
        update={
            "result_status": None,
            "n_main_findings": None,
            "n_research_findings": None,
            "shadow": True,
        }
    )
