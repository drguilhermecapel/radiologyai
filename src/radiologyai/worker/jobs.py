"""Criação de análises (enfileiramento) — usada pela API e pela CLI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from radiologyai import __version__
from radiologyai.errors import ExamTypeUnknownError, ModelUnavailableError, RadiologyAIError

if TYPE_CHECKING:
    from radiologyai.persistence.models import Analysis
    from radiologyai.worker.runtime import Platform

__all__ = ["NotFoundError", "enqueue_analysis"]


class NotFoundError(RadiologyAIError):
    """Recurso inexistente."""


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

    if model_ids:
        registry = platform.engine.registry
        for model_id in model_ids:
            if not registry.has(model_id):
                raise ModelUnavailableError(model_id, "modelo inexistente no registro")
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
    append_event(
        platform.db,
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
    )
    return row
