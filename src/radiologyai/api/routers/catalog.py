"""Catálogo de condições com o nível de evidência por (modelo, rótulo)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS
from radiologyai.api.schemas import (
    CatalogConditionOut,
    CatalogEntryOut,
    CatalogOut,
    UncataloguedLabelOut,
)
from radiologyai.catalog.schema import EvidenceLevel
from radiologyai.persistence.security import Permission
from radiologyai.reporting.content import LEVEL_MEANING_PT
from radiologyai.worker.runtime import Platform

router = APIRouter(tags=["catalog"])

LEVEL_MEANINGS = {
    **LEVEL_MEANING_PT,
    "D": "Nível D — exploratório: só no modo pesquisa, com marca d'água; nunca no laudo.",
}


@router.get(
    "/catalog",
    response_model=CatalogOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.CATALOG_READ),
)
def catalog(
    principal: Annotated[Principal, Depends(require(Permission.CATALOG_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
) -> CatalogOut:
    """Condições do catálogo com o nível calculado de cada modelo que as cobre.

    Entrada cujo manifesto declara ``visibility_default: research`` aparece com
    o nível medido, mas visibilidade ``research`` (``restricted_by_manifest``).

    Rótulos de manifesto cuja condição não está no catálogo aparecem em
    ``uncatalogued`` — sempre nível D (modo pesquisa).
    """
    del principal
    engine = platform.engine
    cat, registry = engine.catalog, engine.registry
    by_condition: dict[str, list[CatalogEntryOut]] = {}
    uncatalogued: list[UncataloguedLabelOut] = []
    for m in registry.manifests():
        for spec in m.labels:
            if not cat.has(spec.condition_id):
                uncatalogued.append(
                    UncataloguedLabelOut(
                        model_id=m.id, label=spec.key, condition_id=spec.condition_id
                    )
                )
                continue
            entry = cat.entry(spec.condition_id, m.id, spec.key)
            level = cat.level(spec.condition_id, m.id, spec.key)
            # O manifesto pode restringir o modelo ao modo pesquisa (licença, termo
            # de uso): vence o mais restritivo, como no pipeline e no laudo.
            restricted = level is not EvidenceLevel.D and m.visibility_default == "research"
            by_condition.setdefault(spec.condition_id, []).append(
                CatalogEntryOut(
                    model_id=m.id,
                    label=spec.key,
                    level=level.value,
                    visibility="research" if level is EvidenceLevel.D or restricted else "main",
                    restricted_by_manifest=restricted,
                    headline=entry.headline if entry else None,
                    basis=entry.basis if entry else (),
                )
            )
    order = "ABCD"
    conditions = []
    for c in cat.conditions():
        entries = tuple(by_condition.get(c.id, ()))
        best = min((e.level for e in entries), key=order.index, default="D")
        conditions.append(
            CatalogConditionOut(
                id=c.id,
                name_pt=c.name_pt,
                name_en=c.name_en,
                group=c.group,
                exam_types=c.exam_types,
                system=c.system,
                code=c.code,
                entries=entries,
                best_level=best,  # type: ignore[arg-type]
                research_only=all(e.visibility == "research" for e in entries),
            )
        )
    levels_file = cat.levels_file
    return CatalogOut(
        protocol_id=levels_file.protocol_id if levels_file else None,
        level_meanings=LEVEL_MEANINGS,
        conditions=tuple(conditions),
        uncatalogued=tuple(uncatalogued),
    )
