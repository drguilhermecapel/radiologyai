"""Executor de avaliação — produz o artefato que sustenta qualquer alegação.

Regra estrutural do projeto: nenhum número de desempenho aparece em documento
algum sem um ``artifacts/eval/<run_id>/metrics.json`` que o sustente, produzido
por este módulo. O verificador ``scripts/check_honesty.py`` impõe isso no CI.

Todo artefato registra ``git_sha``, ``weights_sha256``, ``manifest_sha256``,
``seed`` e as versões das bibliotecas — o conjunto necessário para demonstrar
que uma execução de verificação é repetível (IEC 62304 §5.7).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai import __version__
from radiologyai.errors import EvaluationError
from radiologyai.evaluation.metrics import (
    auroc_with_ci,
    expected_calibration_error,
    operating_point_at_sensitivity,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.data.manifest import Manifest
    from radiologyai.models.backends.base import InferenceBackend
    from radiologyai.models.card import ModelCard

MIN_SUPPORT = 10
"""Mínimo de positivos e negativos para reportar AUROC de um rótulo.

Abaixo disso o intervalo de confiança é largo demais para ter significado. O
sistema legado reportou ``mean_auc: NaN`` sobre ``test_samples: 1``; aqui o
rótulo é marcado como não avaliado, com o motivo declarado.
"""


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "desconhecido"


def environment() -> dict[str, Any]:
    """Ambiente de execução, gravado no artefato para reprodutibilidade."""
    versions: dict[str, str] = {}
    for name in ("numpy", "torch", "torchxrayvision", "pydicom"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:  # noqa: BLE001
            versions[name] = "ausente"

    return {
        "git_sha": git_sha(),
        "code_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": versions,
    }


@dataclass(frozen=True)
class EvaluationConfig:
    """Parâmetros de uma execução. Todos gravados no artefato."""

    seed: int = 20260101
    n_bootstrap: int = 2000
    target_sensitivity: float = 0.90
    min_support: int = MIN_SUPPORT


def check_leakage(card: ModelCard, manifest: Manifest) -> str:
    """Determina se o dataset de avaliação foi usado no treino do modelo.

    Esta é a checagem que separa validação externa de teatro. Avaliar
    ``densenet121-res224-all`` no NIH é *in-distribution* — os pesos ``-all``
    foram treinados em NIH entre outros — e produziria um número inflado que
    parece medição mas não é.

    ``ModelCard.trained_on`` é obrigatório e não-vazio, então esta checagem
    sempre tem base — não existe estado "indeterminado".

    Returns:
        ``"externo"`` ou ``"in-distribution"``.
    """
    dataset = manifest.name.lower()
    if any(dataset.startswith(t.lower()) or t.lower() in dataset for t in card.trained_on):
        return "in-distribution"
    return "externo"


def evaluate_labelwise(
    y_true: npt.NDArray[np.integer[Any]],
    y_score: npt.NDArray[np.floating[Any]],
    label_names: tuple[str, ...],
    config: EvaluationConfig,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Calcula métricas por rótulo. Devolve ``(avaliados, não_avaliados)``."""
    import numpy as np

    evaluated: dict[str, Any] = {}
    not_evaluated: list[dict[str, str]] = []

    for i, name in enumerate(label_names):
        truth = np.asarray(y_true[:, i], dtype=int)
        score = np.asarray(y_score[:, i], dtype=float)
        n_pos, n_neg = int((truth == 1).sum()), int((truth == 0).sum())

        if n_pos < config.min_support or n_neg < config.min_support:
            not_evaluated.append(
                {
                    "label": name,
                    "reason": (
                        f"suporte insuficiente: {n_pos} positivos, {n_neg} negativos "
                        f"(mínimo {config.min_support} de cada)"
                    ),
                }
            )
            continue

        result = auroc_with_ci(truth, score, n_bootstrap=config.n_bootstrap, seed=config.seed)
        entry: dict[str, Any] = {
            "auroc": round(result.auroc, 4),
            "auroc_ci95": [round(result.ci95_low, 4), round(result.ci95_high, 4)],
            "ci_method": result.ci_method,
            "n_pos": result.n_pos,
            "n_neg": result.n_neg,
            "prevalence": round(n_pos / (n_pos + n_neg), 4),
        }
        try:
            op = operating_point_at_sensitivity(truth, score, config.target_sensitivity)
            entry["operating_point"] = {
                "target_sensitivity": op.target_sensitivity,
                "threshold": round(op.threshold, 6),
                "sensitivity": round(op.sensitivity, 4),
                "specificity": round(op.specificity, 4),
                "ppv": round(op.ppv, 4),
                "npv": round(op.npv, 4),
            }
        except EvaluationError as exc:
            entry["operating_point"] = {"error": str(exc)}

        # ECE exige escore em [0, 1]. As saídas do xrv já são sigmoides.
        if score.min() >= 0.0 and score.max() <= 1.0:
            entry["ece"] = round(expected_calibration_error(truth, score), 4)
            entry["calibrated"] = False  # nenhuma calibração foi aplicada ainda

        evaluated[name] = entry

    return evaluated, not_evaluated


def subgroup_analysis(
    manifest: Manifest,
    y_true: npt.NDArray[np.integer[Any]],
    y_score: npt.NDArray[np.floating[Any]],
    label_names: tuple[str, ...],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """AUROC macro por sexo, faixa etária e incidência.

    Exigido para avaliar equidade. Uma lacuna grande entre subgrupos é achado
    a reportar, não a esconder.
    """
    import numpy as np

    def macro_auroc(mask: npt.NDArray[np.bool_]) -> dict[str, Any]:
        if int(mask.sum()) < config.min_support * 2:
            return {"n": int(mask.sum()), "macro_auroc": None, "reason": "suporte insuficiente"}
        values = []
        for i in range(len(label_names)):
            truth = np.asarray(y_true[mask, i], dtype=int)
            if int((truth == 1).sum()) < config.min_support:
                continue
            if int((truth == 0).sum()) < config.min_support:
                continue
            values.append(
                auroc_with_ci(
                    truth,
                    np.asarray(y_score[mask, i], dtype=float),
                    n_bootstrap=200,
                    seed=config.seed,
                ).auroc
            )
        return {
            "n": int(mask.sum()),
            "n_labels": len(values),
            "macro_auroc": round(float(np.mean(values)), 4) if values else None,
        }

    rows = manifest.rows
    groups: dict[str, dict[str, Any]] = {}

    for key, values in (
        ("sex", sorted({r.patient_sex for r in rows if r.patient_sex})),
        ("age_band", ["18-39", "40-59", "60-74", "75+"]),
        ("view_position", sorted({r.view_position for r in rows if r.view_position})),
    ):
        groups[key] = {}
        for value in values:
            if key == "sex":
                mask = np.array([r.patient_sex == value for r in rows])
            elif key == "age_band":
                mask = np.array([r.age_band() == value for r in rows])
            else:
                mask = np.array([r.view_position == value for r in rows])
            groups[key][str(value)] = macro_auroc(mask)

    return groups


def run_evaluation(
    *,
    card: ModelCard,
    backend: InferenceBackend,
    manifest: Manifest,
    y_score: npt.NDArray[np.floating[Any]],
    output_dir: str | Path,
    config: EvaluationConfig | None = None,
    dataset_limitations: tuple[str, ...] = (),
) -> Path:
    """Escreve ``artifacts/eval/<run_id>/metrics.json`` e devolve o diretório.

    Args:
        y_score: matriz ``n_imagens x n_rótulos_do_modelo`` já inferida.
        dataset_limitations: limitações declaradas do dataset, propagadas ao artefato.
    """
    import numpy as np

    cfg = config or EvaluationConfig()
    model_labels = backend.labels

    # Avalia apenas os rótulos presentes nos dois lados. Os demais são
    # declarados como não avaliados — nunca descartados em silêncio.
    shared = [name for name in model_labels if name in manifest.label_names]
    model_index = {name: i for i, name in enumerate(model_labels)}

    truth_matrix = np.array(manifest.label_matrix(), dtype=int)
    manifest_index = {name: i for i, name in enumerate(manifest.label_names)}

    y_true = np.stack([truth_matrix[:, manifest_index[n]] for n in shared], axis=1)
    y_pred = np.stack([np.asarray(y_score)[:, model_index[n]] for n in shared], axis=1)

    evaluated, not_evaluated = evaluate_labelwise(y_true, y_pred, tuple(shared), cfg)

    for name in model_labels:
        if name in manifest.label_names:
            continue
        if name.startswith("__untrained_"):
            not_evaluated.append(
                {
                    "label": name,
                    "reason": (
                        "cabeça de saída não treinada neste conjunto de pesos; "
                        "o valor produzido não tem significado e não é reportado"
                    ),
                }
            )
        else:
            not_evaluated.append(
                {"label": name, "reason": f"sem rótulo correspondente em {manifest.name}"}
            )

    macro = (
        round(float(np.mean([e["auroc"] for e in evaluated.values()])), 4) if evaluated else None
    )

    leakage = check_leakage(card, manifest)
    run_id = f"{card.card_id}__{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": environment(),
        "config": {
            "seed": cfg.seed,
            "n_bootstrap": cfg.n_bootstrap,
            "target_sensitivity": cfg.target_sensitivity,
            "min_support": cfg.min_support,
        },
        "model": {
            "card_id": card.card_id,
            "version": card.version,
            "weights_sha256": card.weights_sha256,
            "trained_on": list(card.trained_on),
            "backend": backend.describe(),
        },
        "dataset": {
            "name": manifest.name,
            "split": manifest.split,
            "manifest_sha256": manifest.sha256(),
            "n_images": len(manifest),
            "n_patients": len(manifest.patient_ids),
            "external_to_training_data": leakage == "externo",
            "leakage_status": leakage,
        },
        "per_label": evaluated,
        "macro_auroc": macro,
        "n_labels_evaluated": len(evaluated),
        "not_evaluated": not_evaluated,
        "subgroups": subgroup_analysis(manifest, y_true, y_pred, tuple(shared), cfg),
        "limitations": [
            *dataset_limitations,
            "Nenhuma calibração foi aplicada: os escores NÃO são probabilidade de doença.",
            "Medição retrospectiva de desempenho de algoritmo isolado. "
            "NÃO constitui validação clínica nem evidência de utilidade clínica.",
        ],
    }

    if leakage == "in-distribution":
        payload["limitations"].insert(
            0,
            "ATENÇÃO: o dataset de avaliação consta entre os dados de treino deste "
            "modelo. Este resultado é in-distribution e NÃO é validação externa.",
        )

    out = Path(output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest.to_csv(out / "manifest_used.csv")
    return out
