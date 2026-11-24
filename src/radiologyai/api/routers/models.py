"""Modelos registrados."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from radiologyai.models import list_cards

router = APIRouter(tags=["models"])


@router.get("/models")
def models() -> list[dict[str, Any]]:
    """Model cards registrados.

    ``has_measured_performance: false`` significa exatamente isso: nenhum
    desempenho foi medido, e nenhum pode ser alegado. O v1 devolvia
    ``accuracy: 0.92`` para modelos que nunca existiram.
    """
    return [
        {
            "card_id": card.card_id,
            "display_name": card.display_name,
            "version": card.version,
            "modality": card.modality,
            "backend": card.backend,
            "labels": list(card.labels),
            "trained_on": list(card.trained_on),
            "weights_sha256": card.weights_sha256,
            "has_measured_performance": card.has_measured_performance,
            "evaluation_runs": list(card.evaluation_runs),
            "intended_use_ref": card.intended_use_ref,
        }
        for card in list_cards()
    ]
