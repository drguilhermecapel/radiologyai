"""Estudos: envio, lista de trabalho, detalhe, tipo de exame, séries e quadros."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from radiologyai.analysis.types import ImageData
from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.schemas import (
    ErrorOut,
    ExamTypeIn,
    SeriesMetaOut,
    StudyListItem,
    StudyListOut,
    StudyOut,
)
from radiologyai.api.shadow import ai_visible, redact_summary
from radiologyai.api.views import analysis_summary, series_out, study_out
from radiologyai.catalog.schema import ExamType
from radiologyai.interop.images import FrameData
from radiologyai.persistence.models import Series, Study
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(tags=["studies"])

_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorOut},
    403: {"model": ErrorOut},
    404: {"model": ErrorOut},
    409: {"model": ErrorOut},
    413: {"model": ErrorOut},
    422: {"model": ErrorOut},
    503: {"model": ErrorOut},
}
FRAME_HEADERS = (
    "Corpo binário little-endian, linha a linha. Cabeçalhos: X-Frame-Rows, X-Frame-Columns, "
    "X-Frame-Channels (1 ou 3), X-Frame-Dtype (uint16|int16|uint8), X-Frame-Scale (fator inteiro "
    "de redução: coordenada no quadro = coordenada de origem / escala), X-Frame-Index, "
    "X-Frame-Count, X-Rescale-Slope, X-Rescale-Intercept (valor físico = armazenado * slope + "
    "intercept; HU em TC). O cliente aplica janela/LUT."
)


def _get_study(session: Session, study_id: str) -> Study:
    study = session.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="estudo inexistente")
    return study


@router.post(
    "/studies",
    response_model=StudyOut,
    status_code=status.HTTP_201_CREATED,
    responses=_ERRORS,
    openapi_extra=perm(Permission.STUDIES_UPLOAD),
)
def upload_study(
    principal: Annotated[Principal, Depends(require(Permission.STUDIES_UPLOAD))],
    platform: Annotated[Platform, Depends(get_platform)],
    files: Annotated[
        list[UploadFile], File(description="DICOM (ou zip de DICOM), NIfTI, PNG ou JPEG")
    ],
    exam_type: Annotated[ExamType | None, Form(description="Tipo de exame declarado")] = None,
    modality: Annotated[
        Literal["CR", "DX", "CT", "MR", "MG", "US", "OT"] | None,
        Form(description="Obrigatória para NIfTI; PNG/JPEG usam OT se ausente"),
    ] = None,
    body_part: Annotated[str | None, Form(max_length=64)] = None,
    view_position: Annotated[str | None, Form(max_length=32)] = None,
    laterality: Annotated[Literal["L", "R", "B"] | None, Form()] = None,
    patient_age_years: Annotated[float | None, Form(ge=0, le=130)] = None,
    patient_sex: Annotated[Literal["F", "M", "O"] | None, Form()] = None,
    sequence_role: Annotated[str | None, Form(max_length=32)] = None,
) -> StudyOut:
    """Envia UM estudo. DICOM é des-identificado na entrada (``MEDAI_DEIDENT_KEY``).

    Metadados de formulário são declarações humanas: preenchem campos ausentes
    (ex.: ``ViewPosition`` vazio), nunca sobrescrevem o DICOM, e ficam registrados.
    ``exam_type=unknown`` (ou ausente e não inferível) exige escolha humana antes da análise.
    """
    from radiologyai.interop.ingest import HumanMeta, ingest
    from radiologyai.persistence.models import Study

    if exam_type is ExamType.UNKNOWN:
        exam_type = None
    human = HumanMeta(
        modality=modality,
        body_part=body_part,
        view_position=view_position,
        laterality=laterality,
        patient_age_years=patient_age_years,
        patient_sex=patient_sex,
        sequence_role=sequence_role,
    )
    tmp = Path(tempfile.mkdtemp(prefix="radiologyai-upload-"))
    try:
        items: list[tuple[str, Path]] = []
        for n, upload in enumerate(files):
            target = tmp / f"upload-{n:05d}"
            with target.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh, length=1 << 20)
            items.append((upload.filename or target.name, target))
        result = ingest(
            items,
            db=platform.db,
            blobs=platform.blobs,
            settings=platform.settings,
            user_id=principal.user_id,
            username=principal.username,
            human=human,
            exam_type=exam_type,
            on_decoded=platform.cache.put,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with platform.db.session() as session:
        study = session.get(Study, result.study_id)
        assert study is not None
        return study_out(session, study, redact=not ai_visible(platform, principal))


@router.get(
    "/studies",
    response_model=StudyListOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.STUDIES_READ),
)
def worklist(
    principal: Annotated[Principal, Depends(require(Permission.STUDIES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    state: Annotated[
        Literal["no_analysis", "queued", "running", "completed", "failed"] | None,
        Query(description="Estado da análise mais recente"),
    ] = None,
    exam_type: ExamType | None = None,
    modality: Annotated[str | None, Query(max_length=8)] = None,
    review_status: Literal["no_analysis", "pending_review", "in_review", "signed"] | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> StudyListOut:
    """Lista de trabalho: estudos com o estado da análise mais recente e da revisão.

    Modo silencioso (REQ-195) sem ``shadow:view``: o resumo da análise vem com
    ``shadow: true`` e sem contagens de achados nem estado do resultado.
    """
    from sqlalchemy import func, select

    from radiologyai.persistence.models import Analysis, Report, Review, Series, Study

    visible = ai_visible(platform, principal)
    latest_id = (
        select(Analysis.id)
        .where(Analysis.study_id == Study.id)
        .order_by(Analysis.requested_at.desc(), Analysis.id.desc())
        .limit(1)
        .correlate(Study)
        .scalar_subquery()
    )
    latest_state = (
        select(Analysis.state).where(Analysis.id == latest_id).correlate(Study).scalar_subquery()
    )
    signed = (
        select(func.count(Report.id))
        .where(Report.analysis_id == latest_id, Report.signed_at.is_not(None))
        .correlate(Study)
        .scalar_subquery()
    )
    reviewed = (
        select(func.count(Review.id))
        .where(Review.analysis_id == latest_id)
        .correlate(Study)
        .scalar_subquery()
    )
    from sqlalchemy import case

    review_expr = case(
        (latest_id.is_(None), "no_analysis"),
        (signed > 0, "signed"),
        (reviewed > 0, "in_review"),
        else_="pending_review",
    )
    query = select(Study, func.coalesce(latest_state, "no_analysis"), review_expr)
    if state is not None:
        query = query.where(func.coalesce(latest_state, "no_analysis") == state)
    if exam_type is not None:
        query = query.where(Study.exam_type == exam_type.value)
    if modality:
        query = query.where(Study.modalities.contains(modality.upper()))
    if review_status is not None:
        query = query.where(review_expr == review_status)
    with platform.db.session() as session:
        total = session.execute(select(func.count()).select_from(query.subquery())).scalar_one()
        rows = session.execute(
            query.order_by(Study.created_at.desc(), Study.id).limit(limit).offset(offset)
        ).all()
        items = []
        for study, _state, review in rows:
            latest = session.execute(
                select(Analysis)
                .where(Analysis.study_id == study.id)
                .order_by(Analysis.requested_at.desc(), Analysis.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            n_series = session.execute(
                select(func.count(Series.id)).where(Series.study_id == study.id)
            ).scalar_one()
            items.append(
                StudyListItem(
                    id=study.id,
                    modalities=tuple(m for m in study.modalities.split(",") if m),
                    description=study.description,
                    exam_type=study.exam_type,
                    exam_type_source=study.exam_type_source,
                    created_at=study.created_at,
                    n_series=int(n_series),
                    latest_analysis=(
                        None
                        if latest is None
                        else analysis_summary(session, latest)
                        if visible
                        else redact_summary(analysis_summary(session, latest))
                    ),
                    review_status=review,
                )
            )
    return StudyListOut(items=tuple(items), total=int(total), limit=limit, offset=offset)


@router.get(
    "/studies/{study_id}",
    response_model=StudyOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.STUDIES_READ),
)
def get_study(
    principal: Annotated[Principal, Depends(require(Permission.STUDIES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    study_id: str,
) -> StudyOut:
    """Estudo com séries e análises (resumos com ``shadow: true`` no modo silencioso)."""
    with platform.db.session() as session:
        return study_out(
            session, _get_study(session, study_id), redact=not ai_visible(platform, principal)
        )


@router.put(
    "/studies/{study_id}/exam-type",
    response_model=StudyOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.STUDIES_SET_EXAM_TYPE),
)
def set_exam_type(
    principal: Annotated[Principal, Depends(require(Permission.STUDIES_SET_EXAM_TYPE))],
    platform: Annotated[Platform, Depends(get_platform)],
    study_id: str,
    body: ExamTypeIn,
) -> StudyOut:
    """Define o tipo de exame (decisão humana, auditada). ``unknown`` não é aceito."""
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import utcnow

    if body.exam_type is ExamType.UNKNOWN:
        raise HTTPException(status_code=422, detail="escolha um tipo de exame concreto")
    with platform.db.session() as session:
        study = _get_study(session, study_id)
        previous = study.exam_type
        study.exam_type = body.exam_type.value
        study.exam_type_source = "human"
        study.exam_type_set_by = principal.user_id
        study.exam_type_set_at = utcnow()
        session.flush()
        append_event(
            None,
            "study.exam_type_set",
            actor_id=principal.user_id,
            actor_username=principal.username,
            subject_type="study",
            subject_id=study_id,
            payload={"from": previous, "to": body.exam_type.value},
            session=session,
        )
    with platform.db.session() as session:
        return study_out(
            session, _get_study(session, study_id), redact=not ai_visible(platform, principal)
        )


def _series(session: Session, series_id: str) -> Series:
    series = session.get(Series, series_id)
    if series is None:
        raise HTTPException(status_code=404, detail="série inexistente")
    return series


def decoded(platform: Platform, series_id: str) -> ImageData:
    """``ImageData`` da série (cache LRU; senão decodifica dos arquivos guardados)."""
    from radiologyai.interop.images import load_series_image

    cached = platform.cache.get(series_id)
    if cached is not None:
        return cached
    with platform.db.session() as session:
        series = _series(session, series_id)
        _ = list(series.instances)
        image = load_series_image(series, platform.blobs)
    platform.cache.put(series_id, image)
    return image


@router.get(
    "/series/{series_id}",
    response_model=SeriesMetaOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.STUDIES_READ),
)
def series_meta(
    principal: Annotated[Principal, Depends(require(Permission.STUDIES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    series_id: str,
) -> SeriesMetaOut:
    """Metadados e geometria para o visualizador (sem decodificar pixels)."""
    del principal
    with platform.db.session() as session:
        s = _series(session, series_id)
        meta = dict(s.meta)
        lo, hi = meta["value_range"]
        color = bool(meta.get("color"))
        if color:
            dtype, slope, intercept = "uint8", 1.0, 0.0
        elif lo >= 0 and hi <= 65535:
            dtype, slope, intercept = "uint16", 1.0, 0.0
        elif lo >= -32768 and hi <= 32767:
            dtype, slope, intercept = "int16", 1.0, 0.0
        else:
            dtype, slope, intercept = "uint16", (hi - lo) / 65535.0, float(lo)
        return SeriesMetaOut(
            series=series_out(s),
            image_meta=meta["image_meta"],
            spacing_mm=tuple(meta["spacing_mm"]) if meta.get("spacing_mm") else None,
            window=tuple(meta["window"]) if meta.get("window") else None,
            value_range=(float(lo), float(hi)),
            color=color,
            affine_lps=meta.get("affine_lps"),
            frame_dtype=dtype,  # type: ignore[arg-type]
            rescale_slope=slope,
            rescale_intercept=intercept,
            default_max_dim=platform.settings.frames_max_dim,
            max_dim_cap=platform.settings.frames_hard_cap,
        )


@router.get(
    "/series/{series_id}/frames/{index}",
    response_class=Response,
    responses={
        200: {"content": {"application/octet-stream": {}}, "description": FRAME_HEADERS},
        **_ERRORS,
    },
    openapi_extra=perm(Permission.STUDIES_READ),
)
def frame(
    principal: Annotated[Principal, Depends(require(Permission.STUDIES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    series_id: str,
    index: int,
    max_dim: Annotated[int | None, Query(ge=16)] = None,
) -> Response:
    """Quadro decodificado no servidor (corte de volume, quadro de cine ou a projeção)."""
    from radiologyai.interop.images import encode_frame

    del principal
    limit = min(max_dim or platform.settings.frames_max_dim, platform.settings.frames_hard_cap)
    image = decoded(platform, series_id)
    data = encode_frame(image, index, limit)
    count = 1 if image.kind.value == "projection" else int(image.pixels.shape[0])
    return Response(
        content=data.data,
        media_type="application/octet-stream",
        headers=frame_headers(data, index, count),
    )


def frame_headers(data: FrameData, index: int, count: int) -> dict[str, str]:
    return {
        "X-Frame-Rows": str(data.rows),
        "X-Frame-Columns": str(data.columns),
        "X-Frame-Channels": str(data.channels),
        "X-Frame-Dtype": data.dtype,
        "X-Frame-Scale": str(data.scale),
        "X-Frame-Index": str(index),
        "X-Frame-Count": str(count),
        "X-Rescale-Slope": repr(data.rescale_slope),
        "X-Rescale-Intercept": repr(data.rescale_intercept),
    }
