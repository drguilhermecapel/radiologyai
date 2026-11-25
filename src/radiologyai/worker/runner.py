"""Execução de UMA análise: imagens guardadas -> ``analyze`` -> resultado imutável.

O resultado é gravado exatamente como o pipeline o produziu
(``AnalysisResult.model_dump_json()``) junto do sha256 desses bytes. Nada é
acrescentado, filtrado ou substituído: modelo indisponível, planejado, não
aplicável ou que falhou aparece em ``runs`` com o motivo. Falha da própria
análise (ex.: imagem que não decodifica) marca a linha como ``failed`` com a
mensagem — nunca um resultado parcial inventado.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from radiologyai.errors import RadiologyAIError

if TYPE_CHECKING:
    from radiologyai.analysis.types import AnalysisResult, ImageData
    from radiologyai.worker.runtime import Platform

__all__ = ["AnalysisJobError", "execute", "persist_result"]

log = logging.getLogger("radiologyai.worker.runner")


class AnalysisJobError(RadiologyAIError):
    """A análise não pôde ser executada (entrada, configuração, estado)."""


def _load_images(platform: Platform, inputs: list[dict[str, Any]]) -> list[ImageData]:
    from radiologyai.interop.images import load_series_image
    from radiologyai.persistence.models import Series

    images: list[ImageData] = []
    with platform.db.session() as session:
        for item in inputs:
            series = session.get(Series, item["series_id"])
            if series is None:
                raise AnalysisJobError(f"série {item['series_id']} não existe mais")
            cached = platform.cache.get(series.id)
            if cached is None:
                _ = list(series.instances)  # carrega a relação dentro da sessão
                cached = load_series_image(series, platform.blobs)
                platform.cache.put(series.id, cached)
            images.append(cached)
    return images


def execute(platform: Platform, analysis_id: str) -> AnalysisResult:
    """Roda o pipeline para uma análise já reivindicada (estado ``running``).

    Raises:
        AnalysisJobError: análise inexistente ou fora do estado ``running``.
        RadiologyAIError: entrada inválida, tipo de exame desconhecido etc.
    """
    from radiologyai.analysis.pipeline import analyze
    from radiologyai.analysis.types import ExamInputs, Primitive
    from radiologyai.catalog.schema import ExamType
    from radiologyai.persistence.models import Analysis

    with platform.db.session() as session:
        row = session.get(Analysis, analysis_id)
        if row is None:
            raise AnalysisJobError(f"análise {analysis_id} inexistente")
        if row.state != "running":
            raise AnalysisJobError(f"análise {analysis_id} está {row.state!r}, não 'running'")
        inputs = list(row.inputs)
        exam_type = ExamType(row.exam_type)
        prompts = tuple(Primitive.model_validate(p) for p in row.prompts or [])
        model_ids = list(row.model_ids) if row.model_ids else None
        include_research = bool(row.include_research)

    images = _load_images(platform, inputs)
    engine = platform.engine
    exam = ExamInputs(images=tuple(images), exam_type=exam_type, prompts=prompts)
    return analyze(
        exam,
        registry=engine.registry,
        catalog=engine.catalog,
        store=platform.blobs.store,
        policy=engine.policy,
        model_ids=model_ids,
        include_research=include_research,
    )


def persist_result(platform: Platform, analysis_id: str, result: AnalysisResult) -> str:
    """Grava resultado, achados desnormalizados e índice de artefatos; audita. Devolve o sha."""
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import Analysis, FindingRow, utcnow

    payload = result.model_dump_json()
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with platform.db.session() as session:
        row = session.get(Analysis, analysis_id)
        if row is None or row.result_json is not None:
            raise AnalysisJobError(f"análise {analysis_id} inexistente ou já com resultado")
        for idx, f in enumerate(result.findings):
            prov = result.provenance[f.provenance_idx]
            session.add(
                FindingRow(
                    analysis_id=analysis_id,
                    idx=idx,
                    model_id=prov.model_id,
                    condition_id=f.condition_id,
                    label=f.label,
                    kind=f.kind,
                    decision=f.decision,
                    evidence_level=f.evidence_level.value,
                    visibility=f.visibility,
                    raw_score=f.raw_score,
                    op_normalized=f.op_normalized,
                    probability=f.probability,
                    category=f.category,
                    abstention_reason=f.abstention_reason.value if f.abstention_reason else None,
                    image_idx=f.image_idx,
                )
            )
        for seg in result.segmentations:
            platform.blobs.register(session, seg.mask, kind="labelmap")
        for exp in result.explanations:
            platform.blobs.register(session, exp.map, kind="explanation")
        row.result_json = payload
        row.result_sha256 = digest
        row.result_status = result.status
        row.state = "completed"
        row.finished_at = utcnow()
        requested_by = row.requested_by
    append_event(
        platform.db,
        "analysis.completed",
        actor_username="worker",
        subject_type="analysis",
        subject_id=analysis_id,
        payload={
            "requested_by": requested_by,
            "result_sha256": digest,
            "status": result.status,
            "input_sha256": list(result.input_sha256),
            "models": [
                {
                    "model_id": p.model_id,
                    "model_version": p.model_version,
                    "weights_sha256": p.weights_sha256,
                    "manifest_sha256": p.manifest_sha256,
                    "code_version": p.code_version,
                }
                for p in result.provenance
            ],
            "runs": {r.model_id: r.status for r in result.runs},
            "n_findings": len(result.findings),
        },
    )
    return digest
