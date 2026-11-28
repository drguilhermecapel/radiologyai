"""Análise legada de UM DICOM por UM model card — gate de escopo primeiro, falha fechada.

Obsoleta: o fluxo da plataforma é ``POST /studies`` + ``POST /studies/{id}/analyses``
(pipeline multimodelo com níveis de evidência). Mantida, autenticada, porque
cobre o gate de escopo por HTTP (REQ-031) e os limiares vindos de artefato
medido (REQ-062).

Esta rota NÃO aplica o catálogo de níveis: seus rótulos não têm nível A-C e
são, portanto, exploratórios (nível D). Escores e faixas só são devolvidos a
papel com modo pesquisa; os demais recebem a contagem do que foi omitido.
Toda chamada é auditada na cadeia do banco (``legacy.analyze``).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict

from radiologyai import __version__
from radiologyai.api.app import DISCLAIMER
from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.errors import (
    BackendUnavailableError,
    IngestionError,
    ModelNotFoundError,
    OutOfScopeError,
)
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(tags=["analyze (legado)"])

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class LegacyAnalyzeOut(BaseModel):
    """Resposta do endpoint legado. Sem ``clinical_metrics`` — o v1 as fabricava."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    input_sha256: str
    code_version: str
    card_id: str
    weights_sha256: str
    modality: str
    metadata: dict[str, Any]
    findings: list[dict[str, Any]]
    research_view: bool = False
    hidden_research_findings: int = 0
    evidence_note: str = (
        "rota legada sem catálogo de níveis: todo rótulo é exploratório (nível D) e só "
        "aparece no modo pesquisa"
    )
    normality_asserted: bool
    requires_physician_review: bool
    disclaimer: str


@router.post(
    "/analyze",
    response_model=LegacyAnalyzeOut,
    deprecated=True,
    openapi_extra=perm(Permission.LEGACY_ANALYZE),
)
async def analyze(
    principal: Annotated[Principal, Depends(require(Permission.LEGACY_ANALYZE))],
    platform: Annotated[Platform, Depends(get_platform)],
    request: Request,
    file: Annotated[UploadFile, File(description="Arquivo DICOM")],
    card_id: Annotated[str, Form()] = "xrv-densenet121-pc",
) -> LegacyAnalyzeOut:
    """Analisa um exame DICOM.

    Ordem obrigatória: ler cabeçalho, aplicar o gate de escopo da modalidade,
    e só então processar pixels. Entrada fora do uso pretendido é recusada com
    HTTP 422 — nunca recebe uma predição degradada (perigo H-03).

    A resposta **não** contém métricas de validação clínica. O v1 devolvia
    ``clinical_metrics`` calculadas a partir de um array inventado de um elemento.
    """
    from radiologyai.api.shadow import block_ai
    from radiologyai.io.metadata import extract_metadata
    from radiologyai.io.reader import read_dicom
    from radiologyai.modalities import resolve
    from radiologyai.models import get_card

    block_ai(platform, principal, action="legacy.analyze")
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="arquivo excede 200 MB")

    input_sha256 = hashlib.sha256(payload).hexdigest()

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=True) as tmp:
        tmp.write(payload)
        tmp.flush()
        path = Path(tmp.name)

        try:
            dataset = read_dicom(path, stop_before_pixels=True)
        except IngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        metadata = extract_metadata(dataset)

        try:
            plugin = resolve(metadata.modality)
        except OutOfScopeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        report = plugin.validate(metadata)
        if not report.accepted:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "entrada fora do uso pretendido",
                    "modality": plugin.code,
                    "reasons": list(report.reasons),
                    "intended_use": "docs/regulatory/01-intended-use.md",
                },
            )

        try:
            card = get_card(card_id)
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            findings = _run_inference(
                path, card, plugin, metadata, artifacts_dir=request.app.state.artifacts_dir
            )
        except BackendUnavailableError as exc:
            # Falha fechada: sem backend não há predição. Nunca um substituto.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "inferência indisponível",
                    "reason": str(exc),
                    "note": "Nenhuma predição é fabricada quando o modelo não pode rodar.",
                },
            ) from exc
    from radiologyai.persistence.audit import append_event

    research_view = principal.can(Permission.RESEARCH_MODE)
    findings = [{**f, "evidence_level": "D", "visibility": "research"} for f in findings]
    abstained = any(f.get("band") in {"inconclusivo", "nao_avaliavel"} for f in findings)
    append_event(
        platform.db,
        "legacy.analyze",
        actor_id=principal.user_id,
        actor_username=principal.username,
        subject_type="input",
        subject_id=input_sha256,
        payload={
            "input_sha256": input_sha256,
            "card_id": card.card_id,
            "weights_sha256": card.weights_sha256,
            "code_version": __version__,
            "modality": plugin.code,
            "n_findings": len(findings),
            "abstained": abstained,
            "research_view": research_view,
        },
    )
    audit_trail = getattr(request.app.state, "audit_trail", None)
    if audit_trail is not None:
        audit_trail.record(
            event="analyze",
            input_sha256=input_sha256,
            card_id=card.card_id,
            weights_sha256=card.weights_sha256,
            code_version=__version__,
            modality=plugin.code,
            user_id=principal.username,
            findings=findings,
            abstained=abstained,
        )

    shown = findings if research_view else []
    return LegacyAnalyzeOut(
        input_sha256=input_sha256,
        code_version=__version__,
        card_id=card.card_id,
        weights_sha256=card.weights_sha256,
        modality=plugin.code,
        metadata=metadata.model_dump(),
        findings=shown,
        research_view=research_view,
        hidden_research_findings=len(findings) - len(shown),
        normality_asserted=False,
        requires_physician_review=True,
        disclaimer=DISCLAIMER,
    )


def _run_inference(
    path: Path, card: Any, plugin: Any, metadata: Any, *, artifacts_dir: str | Path
) -> list[dict[str, Any]]:
    """Executa o modelo. Levanta ``BackendUnavailableError`` quando não pode.

    Os limiares da política de abstenção vêm EXCLUSIVAMENTE de um artefato de
    avaliação referenciado pelo card. Sem artefato disponível neste checkout,
    todo achado sai como *não avaliável* — nunca com um limiar inventado.
    """
    from radiologyai.calibration import AbstentionPolicy
    from radiologyai.evaluation.artifacts import (
        evaluated_labels,
        latest_run_for_card,
        load_metrics,
        operating_points,
    )
    from radiologyai.io.reader import read_dicom, to_pixel_array
    from radiologyai.models.backends import build_backend

    backend = build_backend(card)
    array = to_pixel_array(read_dicom(path), scale="modality")
    image = plugin.preprocess(array, metadata)
    scores = backend.predict(image)

    trained = set(getattr(backend, "trained_labels", backend.labels))
    labels = [label for label in backend.labels if label in trained]
    values = [
        float(score)
        for label, score in zip(backend.labels, scores, strict=True)
        if label in trained
    ]

    run_dir = latest_run_for_card(artifacts_dir, tuple(card.evaluation_runs))
    if run_dir is None:
        reason = (
            "nenhum desempenho medido para este modelo"
            if not card.evaluation_runs
            else "artefato de avaliação referenciado pelo card não está neste checkout"
        )
        return [
            {
                "label": label,
                "score": round(score, 6),
                "band": "nao_avaliavel",
                "calibrated": False,
                "evaluated": False,
                "reason": reason,
            }
            for label, score in zip(labels, values, strict=True)
        ]

    metrics = load_metrics(artifacts_dir, run_dir.name)
    policy = AbstentionPolicy.from_operating_points(operating_points(metrics), calibrated=False)
    measured = evaluated_labels(metrics)
    findings = policy.apply(labels, values, evaluated=[label in measured for label in labels])
    return [
        {
            "label": f.label,
            "score": round(f.score, 6),
            "band": f.band.value,
            "calibrated": f.calibrated,
            "evaluated": f.evaluated,
            "evaluation_run": run_dir.name,
        }
        for f in findings
    ]
