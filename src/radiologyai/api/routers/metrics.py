"""Métricas de desempenho — servidas de artefatos medidos, nunca fabricadas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    """Métricas medidas, lidas de ``artifacts/eval/``.

    Quando não há artefato, devolve ``measured: false`` — e não um número.
    O ``/api/v1/metrics`` do v1 fabricava ``clinical_metrics`` a partir de
    ``y_true = np.array([1])  # Mock ground truth`` a cada requisição.
    """
    root = Path(request.app.state.artifacts_dir)
    runs = sorted(root.glob("*/metrics.json")) if root.is_dir() else []

    if not runs:
        return {
            "measured": False,
            "reason": (
                "Nenhum artefato de avaliação em "
                f"{root}. Nenhum desempenho foi medido, portanto nenhum é reportado."
            ),
            "how_to_produce": (
                "radiologyai evaluate --card <id> --manifest <csv> --data-root <dir>"
            ),
        }

    latest = json.loads(runs[-1].read_text(encoding="utf-8"))
    return {
        "measured": True,
        "run_id": latest["run_id"],
        "model": latest["model"]["card_id"],
        "dataset": latest["dataset"]["name"],
        "external_to_training_data": latest["dataset"]["external_to_training_data"],
        "macro_auroc": latest["macro_auroc"],
        "per_label": latest["per_label"],
        "not_evaluated": latest["not_evaluated"],
        "subgroups": latest["subgroups"],
        "limitations": latest["limitations"],
        "provenance": latest["environment"],
    }


@router.get("/metrics/runs")
def runs(request: Request) -> list[str]:
    """Todos os run_ids disponíveis."""
    root = Path(request.app.state.artifacts_dir)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"{root} não existe")
    return sorted(p.parent.name for p in root.glob("*/metrics.json"))
