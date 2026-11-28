"""Criação de análises (enfileiramento) — usada pela API e pela CLI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from radiologyai import __version__
from radiologyai.errors import ExamTypeUnknownError, ModelUnavailableError, RadiologyAIError

if TYPE_CHECKING:
    from radiologyai.persistence.models import Analysis
    from radiologyai.worker.runtime import Platform

__all__ = ["NotFoundError", "enqueue_analysis", "research_models"]


class NotFoundError(RadiologyAIError):
    """Recurso inexistente."""


def research_models(platform: Platform, model_ids: Sequence[str]) -> list[str]:
    """Modelos exploratórios entre ``model_ids`` e suas dependências.

    Exploratório: rota efetiva ``research`` no manifesto, ``visibility_default:
    research`` ou listado como pesquisa na tabela de roteamento. Pedir um deles
    explicitamente é rodar modo pesquisa — exige a mesma permissão.

    Raises:
        ModelUnavailableError: modelo inexistente no registro.
    """
    from radiologyai.analysis.routing import load_routing

    registry = platform.engine.registry
    for model_id in model_ids:
        if not registry.has(model_id):
            raise ModelUnavailableError(model_id, "modelo inexistente no registro")
    routed_research: set[str] = set()
    try:
        table = load_routing(registry)
        for route in table.exam_types.values():
            routed_research.update(route.research)
    except RadiologyAIError:
        routed_research = set()
    out: list[str] = []
    for model_id in registry.execution_order(model_ids):
        manifest = registry.manifest(model_id)
        if (
            manifest.effective_route == "research"
            or manifest.visibility_default == "research"
            or model_id in routed_research
        ):
            out.append(model_id)
    return out


def enqueue_analysis(
    platform: Platform,
    study_id: str,
    *,
    user_id: str,
    username: str,
    model_ids: Sequence[str] | None = None,
    include_research: bool = False,
    prompts: Sequence[dict[str, Any]] = (),
) -> Analysis:
    """Cria a análise ``queued`` (uma linha nova a cada pedido; reexecução = nova linha).

    Raises:
        NotFoundError: estudo inexistente.
        ExamTypeUnknownError: tipo de exame ``unknown`` — um humano deve escolher antes.
        ModelUnavailableError: ``model_ids`` cita modelo que não existe no registro.
    """
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import Analysis, Study

    if model_ids and research_models(platform, model_ids):
        # Modelo exploratório pedido explicitamente: a análise É de modo pesquisa
        # (quem chama confere a permissão; ver ``research_models``).
        include_research = True
    with platform.db.session() as session:
        study = session.get(Study, study_id)
        if study is None:
            raise NotFoundError(f"estudo {study_id} inexistente")
        if study.exam_type == "unknown":
            raise ExamTypeUnknownError(
                "tipo de exame desconhecido; um humano deve escolhê-lo antes da análise"
            )
        inputs = [
            {"image_idx": i, "series_id": s.id, "content_sha256": s.content_sha256}
            for i, s in enumerate(study.series)
        ]
        row = Analysis(
            study_id=study_id,
            state="queued",
            exam_type=study.exam_type,
            include_research=include_research,
            model_ids=list(model_ids) if model_ids else None,
            prompts=[dict(p) for p in prompts],
            inputs=inputs,
            requested_by=user_id,
            code_version=__version__,
        )
        session.add(row)
        session.flush()
        append_event(
            None,
            "analysis.requested",
            actor_id=user_id,
            actor_username=username,
            subject_type="analysis",
            subject_id=row.id,
            payload={
                "study_id": study_id,
                "exam_type": row.exam_type,
                "include_research": include_research,
                "model_ids": list(model_ids) if model_ids else None,
                "input_sha256": [i["content_sha256"] for i in inputs],
                "n_prompts": len(prompts),
            },
            session=session,
        )
    return row
