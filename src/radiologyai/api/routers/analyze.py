"""Análise de exame — gate de escopo primeiro, falha fechada sempre."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from radiologyai import __version__
from radiologyai.api.app import DISCLAIMER
from radiologyai.errors import (
    BackendUnavailableError,
    IngestionError,
    ModelNotFoundError,
    OutOfScopeError,
)

router = APIRouter(tags=["analyze"])

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@router.post("/analyze")
async def analyze(
    file: Annotated[UploadFile, File(description="Arquivo DICOM")],
    card_id: Annotated[str, Form()] = "xrv-densenet121-pc",
) -> dict[str, Any]:
    """Analisa um exame DICOM.

    Ordem obrigatória: ler cabeçalho, aplicar o gate de escopo da modalidade,
    e só então processar pixels. Entrada fora do uso pretendido é recusada com
    HTTP 422 — nunca recebe uma predição degradada (perigo H-03).

    A resposta **não** contém métricas de validação clínica. O v1 devolvia
    ``clinical_metrics`` calculadas a partir de um array inventado de um elemento.
    """
    from radiologyai.io.metadata import extract_metadata
    from radiologyai.io.reader import read_dicom
    from radiologyai.modalities import resolve
    from radiologyai.models import get_card

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
            findings = _run_inference(path, card, plugin, metadata)
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

    return {
        "input_sha256": input_sha256,
        "code_version": __version__,
        "card_id": card.card_id,
        "weights_sha256": card.weights_sha256,
        "modality": plugin.code,
        "metadata": metadata.model_dump(),
        "findings": findings,
        "normality_asserted": False,
        "requires_physician_review": True,
        "disclaimer": DISCLAIMER,
    }


def _run_inference(path: Path, card: Any, plugin: Any, metadata: Any) -> list[dict[str, Any]]:
    """Executa o modelo. Levanta ``BackendUnavailableError`` quando não pode."""
    from radiologyai.calibration import AbstentionPolicy
    from radiologyai.io.reader import read_dicom, to_pixel_array
    from radiologyai.models.backends import build_backend

    backend = build_backend(card)
    array = to_pixel_array(read_dicom(path), scale="modality")
    image = plugin.preprocess(array, metadata)
    scores = backend.predict(image)

    trained = set(getattr(backend, "trained_labels", backend.labels))

    # Sem artefato de avaliação não há ponto de operação medido, portanto não há
    # política de abstenção legítima: tudo fica indeterminado até haver medição.
    if not card.evaluation_runs:
        return [
            {
                "label": label,
                "score": round(float(score), 6),
                "band": "nao_avaliavel",
                "calibrated": False,
                "evaluated": False,
                "reason": "nenhum desempenho medido para este modelo",
            }
            for label, score in zip(backend.labels, scores, strict=True)
            if label in trained
        ]

    policy = AbstentionPolicy.from_operating_points(
        {label: 0.5 for label in backend.labels if label in trained}, calibrated=False
    )
    labels = [label for label in backend.labels if label in trained]
    values = [
        float(score)
        for label, score in zip(backend.labels, scores, strict=True)
        if label in trained
    ]
    return [
        {
            "label": f.label,
            "score": round(f.score, 6),
            "band": f.band.value,
            "calibrated": f.calibrated,
            "evaluated": f.evaluated,
        }
        for f in policy.apply(labels, values)
    ]
