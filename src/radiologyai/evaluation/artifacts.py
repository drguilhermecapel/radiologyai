"""Leitura de artefatos de avaliação — a única fonte de limiares e métricas.

Um model card aponta para ``evaluation_runs``; este módulo localiza o artefato
correspondente em ``artifacts/eval/`` e extrai o que o resto do sistema pode
usar: pontos de operação medidos e o conjunto de rótulos efetivamente
avaliados. Nada aqui inventa valor: sem artefato, devolve vazio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from radiologyai.errors import EvaluationError


def find_run(artifacts_dir: str | Path, run_id: str) -> Path:
    """Diretório de um run. Levanta se não existir — nunca devolve um substituto."""
    run_dir = Path(artifacts_dir) / run_id
    if not (run_dir / "metrics.json").is_file():
        raise EvaluationError(
            f"artefato {run_id!r} não encontrado em {artifacts_dir}. "
            "O model card referencia uma avaliação que não está neste checkout."
        )
    return run_dir


def load_metrics(artifacts_dir: str | Path, run_id: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (find_run(artifacts_dir, run_id) / "metrics.json").read_text(encoding="utf-8")
    )
    return data


def operating_points(metrics: dict[str, Any]) -> dict[str, float]:
    """``rótulo -> limiar`` medido em sensibilidade-alvo. Só rótulos avaliados.

    Um rótulo sem ponto de operação (suporte insuficiente, ou erro registrado)
    fica de fora — e a política de abstenção o trata como não avaliável.
    """
    points: dict[str, float] = {}
    for label, entry in metrics.get("per_label", {}).items():
        op = entry.get("operating_point") or {}
        threshold = op.get("threshold")
        if isinstance(threshold, int | float):
            points[label] = float(threshold)
    return points


def evaluated_labels(metrics: dict[str, Any]) -> frozenset[str]:
    return frozenset(metrics.get("per_label", {}))


def latest_run_for_card(artifacts_dir: str | Path, run_ids: tuple[str, ...]) -> Path | None:
    """O run mais recente entre os referenciados pelo card, ou ``None`` se nenhum existir."""
    existing = []
    for run_id in run_ids:
        candidate = Path(artifacts_dir) / run_id / "metrics.json"
        if candidate.is_file():
            existing.append(candidate.parent)
    if not existing:
        return None
    return sorted(existing, key=lambda p: p.name)[-1]
