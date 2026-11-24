#!/usr/bin/env python
"""Escolha do codificador das sondas de RX por medição (viabilidade, não evidência).

Reusa os *embeddings* já extraídos no spike (``/root/medai-cache/spikes/encoders``)
para comparar codificadores com a MESMA receita de sonda do produto
(:func:`radiologyai.analysis.probes.fit.fit_probe`):

* TB: treina em Shenzhen, testa em Montgomery (externo) e o inverso;
* fratura: treina em FracAtlas (2 mil imagens com embedding de todos os
  codificadores), testa na amostra do GRAZPEDWRI-DX (externo, pediátrico).

Grava ``<eval-preds>/_encoder_selection/compare.json``. Não é medição de
desempenho do produto: os embeddings do spike usaram um pré-processamento
canônico próprio; as sondas publicadas são reajustadas com o pré-processamento
do adaptador (``probe_fit_xr_probes.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from predict_xr_common import EVAL_PREDS, write_json  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from radiologyai.analysis.probes.fit import fit_probe  # noqa: E402

SPIKE = Path("/root/medai-cache/spikes/encoders")
ENCODERS = ("xrv", "raddino", "biomedclip")


def emb(dataset: str, encoder: str) -> np.ndarray | None:
    path = SPIKE / "emb" / f"{dataset}__{encoder}.npy"
    return np.load(path) if path.is_file() else None


def index(dataset: str) -> pd.DataFrame:
    return pd.read_csv(SPIKE / "cache" / f"{dataset}_index.tsv", sep="\t", dtype={"image_id": str})


def transfer(xa, ya, ga, xb, yb) -> dict[str, float]:
    fit = fit_probe(xa, ya, ga)
    score = ((xb - fit.mean) / fit.scale) @ fit.coef + fit.intercept
    from sklearn.metrics import roc_auc_score as auc

    return {
        "C": fit.C,
        "train_oof_auroc": round(float(auc(ya, fit.oof_logit)), 4),
        "external_auroc": round(float(auc(yb, score)), 4),
        "n_train": int(len(ya)),
        "n_test": int(len(yb)),
        "n_test_pos": int(yb.sum()),
    }


def main() -> None:
    out: dict[str, dict[str, object]] = {}
    sz, mc = index("shenzhen"), index("montgomery")
    for enc in ENCODERS:
        a, b = emb("shenzhen", enc), emb("montgomery", enc)
        if a is None or b is None:
            out.setdefault("tb", {})[enc] = "embeddings ausentes"
            continue
        out.setdefault("tb", {})[enc] = {
            "train_shenzhen_test_montgomery": transfer(
                a, sz.label.values, sz.case_number.values, b, mc.label.values
            ),
            "train_montgomery_test_shenzhen": transfer(
                b, mc.label.values, mc.case_number.values, a, sz.label.values
            ),
        }
    fa_full, fa2k, gz = index("fracatlas"), index("fracatlas2k"), index("grazpedwri")
    rows = fa2k["full_row"].to_numpy()
    for enc in ENCODERS:
        a = emb("fracatlas2k", enc)
        if a is None:
            full = emb("fracatlas", enc)
            a = None if full is None else full[rows]
        b = emb("grazpedwri", enc)
        if a is None or b is None:
            out.setdefault("fracture", {})[enc] = "embeddings ausentes"
            continue
        groups = fa2k["image_id"].to_numpy()  # FracAtlas não publica id de paciente
        gy = gz["fracture_visible"].fillna(0).astype(int).to_numpy()
        out.setdefault("fracture", {})[enc] = {
            "train_fracatlas2k_test_grazpedwri_sample": transfer(
                a, fa2k.label.values, groups, b, gy
            )
        }
    del fa_full
    print(roc_auc_score.__module__)
    write_json(EVAL_PREDS / "_encoder_selection" / "compare.json", out)
    import json

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
