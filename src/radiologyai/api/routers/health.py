"""Estado do serviço."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from radiologyai import __version__, selftest
from radiologyai.api.app import DISCLAIMER

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, Any]:
    """Estado do serviço e disponibilidade dos subsistemas opcionais."""
    return {
        "status": "ok",
        "version": __version__,
        "disclaimer": DISCLAIMER,
        "subsystems": selftest()["optional"],
    }


@router.get("/status")
def status() -> dict[str, Any]:
    """Estado detalhado, incluindo modalidades e sua maturidade."""
    from radiologyai.modalities import available_modalities

    seen = {p.code: p.describe() for p in available_modalities().values()}
    return {
        "version": __version__,
        "modalities": [seen[c] for c in sorted(seen)],
        "clinically_validated": False,
        "regulatory_status": "não registrado em nenhuma agência",
        "disclaimer": DISCLAIMER,
    }
