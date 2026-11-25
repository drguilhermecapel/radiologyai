"""DICOMweb: busca (QIDO) e importação (WADO) de estudos do PACS (Orthanc).

A importação passa pela mesma ingestão do envio de arquivos: des-identificação
na entrada, decodificação real e tipo de exame por metadados ou humano.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS
from radiologyai.api.schemas import DicomwebImportIn, DicomwebSearchOut, StudyOut
from radiologyai.api.views import study_out
from radiologyai.persistence.models import Study
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(prefix="/dicomweb", tags=["dicomweb"])
_P = Permission.DICOMWEB_IMPORT


@router.get("/studies", response_model=DicomwebSearchOut, responses=_ERRORS, openapi_extra=perm(_P))
def search(
    principal: Annotated[Principal, Depends(require(_P))],
    platform: Annotated[Platform, Depends(get_platform)],
    modality: Annotated[str | None, Query(max_length=8)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DicomwebSearchOut:
    """QIDO-RS no PACS configurado; devolve só atributos NÃO identificadores."""
    from radiologyai.interop.dicomweb import DicomwebGateway

    del principal
    gateway = DicomwebGateway.from_settings(platform.settings)
    found = gateway.search_studies(modality=modality, limit=limit, offset=offset)
    return DicomwebSearchOut(items=tuple(s.model_dump() for s in found))


@router.post(
    "/import",
    response_model=StudyOut,
    status_code=status.HTTP_201_CREATED,
    responses=_ERRORS,
    openapi_extra=perm(_P),
)
def import_study(
    principal: Annotated[Principal, Depends(require(_P))],
    platform: Annotated[Platform, Depends(get_platform)],
    body: DicomwebImportIn,
) -> StudyOut:
    """WADO-RS do estudo inteiro -> ingestão (des-identificação na entrada)."""
    import io

    from radiologyai.interop.dicomweb import DicomwebGateway
    from radiologyai.interop.ingest import ingest

    gateway = DicomwebGateway.from_settings(platform.settings)
    datasets = gateway.retrieve_study(body.study_instance_uid)
    with tempfile.TemporaryDirectory(prefix="radiologyai-wado-") as tmp:
        items: list[tuple[str, Path]] = []
        for n, ds in enumerate(datasets):
            buf = io.BytesIO()
            ds.save_as(buf, enforce_file_format=True)
            target = Path(tmp) / f"wado-{n:06d}.dcm"
            target.write_bytes(buf.getvalue())
            items.append((target.name, target))
        result = ingest(
            items,
            db=platform.db,
            blobs=platform.blobs,
            settings=platform.settings,
            user_id=principal.user_id,
            username=principal.username,
            exam_type=body.exam_type,
            source="dicomweb",
            on_decoded=platform.cache.put,
        )
    with platform.db.session() as session:
        study = session.get(Study, result.study_id)
        assert study is not None
        return study_out(session, study)
