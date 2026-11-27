"""Modelos: manifesto (cartão) com estado dos pesos e nível de evidência por rótulo.

Nenhum número de desempenho aparece aqui: o manifesto não os carrega, e o
nível vem do catálogo de evidência (calculado, nunca digitado). O ``/models``
do v1 devolvia ``accuracy: 0.92`` (número fabricado) de modelos que nunca existiram.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS
from radiologyai.api.schemas import LabelOut, ModelCardOut, WeightOut
from radiologyai.catalog.schema import visibility_for
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(tags=["models"])


def _states(platform: Platform, model_id: str, verify: bool) -> dict[str, str]:
    """Estado dos pesos. Sem ``verify`` não recalcula sha256 (pesos podem ter GB):
    arquivo presente vira ``unverified`` — a conferência real acontece na carga."""
    registry = platform.engine.registry
    if verify:
        return {s.key: s.state for s in registry.weight_states(model_id)}
    out: dict[str, str] = {}
    for w in registry.manifest(model_id).weights:
        path = registry.model_dir(model_id) / w.filename
        if not w.sha256:
            out[w.key] = "unpinned"
        elif not path.is_file():
            out[w.key] = "missing"
        else:
            out[w.key] = "unverified"
    return out


def card(platform: Platform, model_id: str, *, verify: bool = False) -> ModelCardOut:
    engine = platform.engine
    registry, catalog = engine.registry, engine.catalog
    m = registry.manifest(model_id)
    states = _states(platform, model_id, verify)
    labels = []
    for spec in m.labels:
        level = catalog.level(spec.condition_id, m.id, spec.key)
        labels.append(
            LabelOut(
                key=spec.key,
                source_label=spec.source_label,
                condition_id=spec.condition_id,
                kind=spec.kind,
                score_kind=spec.score_kind,
                code=spec.code,
                evidence_level=level.value,
                visibility=visibility_for(level),
                in_catalog=catalog.has(spec.condition_id),
            )
        )
    return ModelCardOut(
        id=m.id,
        name=m.name,
        version=m.version,
        status=m.status,
        planned_reason=m.planned_reason,
        task=m.task,
        route=m.effective_route,
        summary=m.summary,
        intended_use=m.intended_use,
        capabilities=tuple(m.capabilities),
        exam_types=m.applicability.exam_types,
        modalities=tuple(x.value for x in m.applicability.modalities),
        license_code=m.license.code,
        license_weights=m.license.weights,
        commercial_use=m.license.commercial_use,
        training_data_terms=m.license.training_data_terms,
        training_data=m.training_data,
        depends_on=m.depends_on,
        weights=tuple(
            WeightOut(
                key=w.key,
                filename=w.filename,
                sha256=w.sha256,
                bytes=w.bytes,
                state=states.get(w.key, "unpinned"),  # type: ignore[arg-type]
            )
            for w in m.weights
        ),
        labels=tuple(labels),
        limitations=m.limitations,
        references=tuple(r.model_dump() for r in m.references),
        manifest_sha256=registry.manifest_sha256(m.id),
    )


@router.get(
    "/models",
    response_model=list[ModelCardOut],
    responses=_ERRORS,
    openapi_extra=perm(Permission.MODELS_READ),
)
def models(
    principal: Annotated[Principal, Depends(require(Permission.MODELS_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
) -> list[ModelCardOut]:
    """Todos os modelos do registro, inclusive os planejados (com o motivo)."""
    del principal
    return [card(platform, m.id) for m in platform.engine.registry.manifests()]


@router.get(
    "/models/{model_id}",
    response_model=ModelCardOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.MODELS_READ),
)
def model(
    principal: Annotated[Principal, Depends(require(Permission.MODELS_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    model_id: str,
    verify_weights: Annotated[
        bool, Query(description="Recalcula o sha256 de cada peso (lento para pesos grandes)")
    ] = False,
) -> ModelCardOut:
    """Cartão de um modelo; ``verify_weights`` confere os pesos agora."""
    del principal
    if not platform.engine.registry.has(model_id):
        raise HTTPException(status_code=404, detail="modelo inexistente")
    return card(platform, model_id, verify=verify_weights)
