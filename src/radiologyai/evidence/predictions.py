"""Leitura do parquet de predições — a interface entre inferência e medição.

Formato por rótulo (``<raiz>/<modelo>/<dataset>/<rótulo>.parquet``):

========== ======== ==========================================================
coluna     tipo     significado
========== ======== ==========================================================
patient_id str      obrigatória; cluster do bootstrap e do split
image_id   str      obrigatória; única no arquivo
y_true     int      obrigatória; 0/1, ou nulo/-1 = rótulo ausente (excluído)
y_score    float    obrigatória; escore bruto do modelo (sem NaN)
sex        str      opcional (subgrupo)
age        float    opcional, anos (subgrupo ``age_band``)
view       str      opcional (subgrupo)
========== ======== ==========================================================

Formato multirrótulo (um arquivo para vários rótulos): colunas fixas acima sem
``y_true``/``y_score`` e, para cada rótulo ``L``, ``y_true__L`` e ``y_score__L``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.errors import EvidenceError
from radiologyai.evaluation.dataset import canonical_rows_sha256

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

REQUIRED = ("patient_id", "image_id")
OPTIONAL = ("sex", "age", "view")
TRUE_PREFIX = "y_true__"
SCORE_PREFIX = "y_score__"


@dataclass(frozen=True)
class LabelPredictions:
    """Predições de um rótulo, já sem as imagens de rótulo ausente."""

    label: str
    patient_ids: list[str]
    image_ids: list[str]
    y_true: npt.NDArray[np.int64]
    y_score: npt.NDArray[np.float64]
    subgroups: dict[str, list[Any]]
    n_missing_label: int
    dataset_sha256: str  # manifesto: todas as colunas exceto o escore
    predictions_sha256: str  # manifesto + escore


def _require_pandas() -> Any:
    import importlib.util

    if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("pyarrow") is None:
        from radiologyai.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "leitura de parquet exige pandas e pyarrow: pip install 'radiologyai[eval]'"
        )
    import pandas as pd

    return pd


def labels_in(path: Path) -> list[str]:
    """Rótulos presentes num parquet (``[""]`` para o formato por rótulo)."""
    import pyarrow.parquet as pq

    _require_pandas()
    names = pq.read_schema(path).names
    multi = sorted(n[len(TRUE_PREFIX) :] for n in names if n.startswith(TRUE_PREFIX))
    if multi:
        return multi
    if "y_true" in names and "y_score" in names:
        return [""]
    raise EvidenceError(
        f"{path}: sem colunas y_true/y_score nem y_true__<rótulo>/y_score__<rótulo>"
    )


def read_label(path: str | Path, label: str) -> LabelPredictions:
    """Lê as predições de ``label``; ``label`` vazio = formato por rótulo.

    Raises:
        EvidenceError: colunas ausentes, ids duplicados, escore não finito,
            rótulo fora de {0, 1, ausente}.
    """
    import numpy as np

    pd = _require_pandas()
    p = Path(path)
    if not p.is_file():
        raise EvidenceError(f"parquet de predições não encontrado: {p}")
    df = pd.read_parquet(p)
    true_col, score_col = (
        ("y_true", "y_score")
        if not label
        else (
            TRUE_PREFIX + label,
            SCORE_PREFIX + label,
        )
    )
    missing = [c for c in (*REQUIRED, true_col, score_col) if c not in df.columns]
    if missing:
        raise EvidenceError(f"{p}: colunas obrigatórias ausentes: {missing}")
    if df["image_id"].duplicated().any():
        raise EvidenceError(f"{p}: image_id duplicado")
    if df["patient_id"].isna().any():
        raise EvidenceError(f"{p}: patient_id ausente em alguma linha")

    present = [c for c in OPTIONAL if c in df.columns]
    rows = []
    for rec in df[[*REQUIRED, *present, true_col, score_col]].to_dict(orient="records"):
        yt = rec[true_col]
        yt = None if yt is None or (isinstance(yt, float) and np.isnan(yt)) else int(yt)
        row = {
            "image_id": str(rec["image_id"]),
            "patient_id": str(rec["patient_id"]),
            "y_true": None if yt == -1 else yt,
            **{c: rec[c] for c in present},
        }
        row["y_score"] = float(rec[score_col])
        rows.append(row)
    manifest_cols = ["image_id", "patient_id", *present, "y_true"]
    dataset_sha = canonical_rows_sha256(rows, manifest_cols)
    predictions_sha = canonical_rows_sha256(rows, [*manifest_cols, "y_score"])

    kept = [r for r in rows if r["y_true"] is not None]
    bad = sorted({r["y_true"] for r in kept} - {0, 1})
    if bad:
        raise EvidenceError(f"{p}: {true_col} fora de {{0, 1, ausente}}: {bad}")
    scores = np.array([r["y_score"] for r in kept], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise EvidenceError(f"{p}: {score_col} com NaN/inf; nenhuma métrica é calculada")

    def norm(v: Any) -> Any:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return v

    return LabelPredictions(
        label=label,
        patient_ids=[r["patient_id"] for r in kept],
        image_ids=[r["image_id"] for r in kept],
        y_true=np.array([r["y_true"] for r in kept], dtype=np.int64),
        y_score=scores,
        subgroups={c: [norm(r[c]) for r in kept] for c in present},
        n_missing_label=len(rows) - len(kept),
        dataset_sha256=dataset_sha,
        predictions_sha256=predictions_sha,
    )


@dataclass(frozen=True)
class ContinuousPredictions:
    """Medida contínua do modelo contra uma medida de referência, uma linha por caso."""

    patient_ids: list[str]
    image_ids: list[str]
    reference: npt.NDArray[np.float64]
    measured: npt.NDArray[np.float64]
    n_abstained: int  # casos sem medida do modelo (abstenção declarada ou nula)
    n_filtered_out: int  # linhas fora do filtro ``where``
    dataset_sha256: str  # manifesto: ids + referência (+ colunas do filtro)
    predictions_sha256: str  # manifesto + medida do modelo


def read_continuous(
    path: str | Path,
    *,
    reference_column: str,
    measured_column: str,
    where: dict[str, str] | None = None,
) -> ContinuousPredictions:
    """Lê uma medida contínua (ex.: FE do modelo) e a sua referência.

    ``where`` seleciona linhas por igualdade textual de colunas (ex.: ``{"mode":
    "prompt_free"}``), para arquivos que guardam várias variantes do mesmo caso.
    Linhas sem medida do modelo são **contadas** como abstenção e excluídas da
    concordância — nunca imputadas.

    Raises:
        EvidenceError: colunas ausentes, ``image_id`` duplicado após o filtro,
            referência ausente/não finita, filtro que não seleciona nenhuma linha.
    """
    import numpy as np

    pd = _require_pandas()
    p = Path(path)
    if not p.is_file():
        raise EvidenceError(f"parquet de predições não encontrado: {p}")
    df = pd.read_parquet(p)
    conditions = dict(where or {})
    missing = [
        c
        for c in (*REQUIRED, reference_column, measured_column, *conditions)
        if c not in df.columns
    ]
    if missing:
        raise EvidenceError(f"{p}: colunas obrigatórias ausentes: {missing}")
    n_all = len(df)
    for column, value in conditions.items():
        df = df[df[column].astype(str) == value]
    if df.empty:
        raise EvidenceError(f"{p}: o filtro {conditions} não seleciona nenhuma linha")
    if df["image_id"].duplicated().any():
        raise EvidenceError(f"{p}: image_id duplicado após o filtro {conditions}")
    if df["patient_id"].isna().any():
        raise EvidenceError(f"{p}: patient_id ausente em alguma linha")

    rows: list[dict[str, Any]] = []
    for rec in df[[*REQUIRED, *conditions, reference_column, measured_column]].to_dict(
        orient="records"
    ):
        ref = rec[reference_column]
        if ref is None or not np.isfinite(float(ref)):
            raise EvidenceError(
                f"{p}: {reference_column} ausente ou não finito em {rec['image_id']!r}; "
                "caso sem referência não entra na concordância"
            )
        meas = rec[measured_column]
        meas_f = None if meas is None or not np.isfinite(float(meas)) else float(meas)
        rows.append(
            {
                "image_id": str(rec["image_id"]),
                "patient_id": str(rec["patient_id"]),
                **{c: str(rec[c]) for c in conditions},
                "reference": float(ref),
                "measured": meas_f,
            }
        )
    manifest_cols = ["image_id", "patient_id", *conditions, "reference"]
    dataset_sha = canonical_rows_sha256(rows, manifest_cols)
    predictions_sha = canonical_rows_sha256(rows, [*manifest_cols, "measured"])
    kept = [r for r in rows if r["measured"] is not None]
    if not kept:
        raise EvidenceError(f"{p}: nenhum caso com medida do modelo ({measured_column})")
    return ContinuousPredictions(
        patient_ids=[r["patient_id"] for r in kept],
        image_ids=[r["image_id"] for r in kept],
        reference=np.array([r["reference"] for r in kept], dtype=np.float64),
        measured=np.array([r["measured"] for r in kept], dtype=np.float64),
        n_abstained=len(rows) - len(kept),
        n_filtered_out=n_all - len(rows),
        dataset_sha256=dataset_sha,
        predictions_sha256=predictions_sha,
    )
