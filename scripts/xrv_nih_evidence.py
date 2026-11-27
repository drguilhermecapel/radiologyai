"""Converte as predições TorchXRayVision do NIH para o formato da fábrica de evidência.

Entrada (gerada por ``predict_xr_xrv.py --merge``)::

    <preds_root>/<model>/nih-cxr14-test/predictions.parquet          rótulos NLP do NIH
    <preds_root>/<model>/nih-google-adjudicated/predictions.parquet  rótulos adjudicados (Google)

Colunas de entrada: ``y_<rótulo>`` (0/1) e ``logit_<rótulo>`` (logit bruto do modelo).

Saída: ``evidence.parquet`` ao lado de cada ``predictions.parquet``, com
``y_true__<chave>`` e ``y_score__<chave>`` para cada rótulo do manifesto que tem
rótulo de referência correspondente no conjunto. Nenhum valor é inventado: rótulo sem
correspondência semântica direta não entra (ex.: ``lung_lesion`` x "Nodule or mass"
entra; ``nodule`` isolado x "Nodule or mass" não entra).

``plan`` imprime os itens ``measure`` do plano (um por modelo x conjunto x rótulo).

Uso::

    python scripts/xrv_nih_evidence.py convert
    python scripts/xrv_nih_evidence.py plan > /tmp/jobs.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "src" / "radiologyai" / "analysis" / "manifests"
PREDS_ROOT = Path(os.environ.get("MEDAI_CACHE", "/root/medai-cache")) / "eval-preds"
WEIGHTS_DIR = Path(os.environ.get("MEDAI_CACHE", "/root/medai-cache")) / "weights" / "xrv"

MODELS = (
    "xrv-densenet121-res224-pc",
    "xrv-densenet121-res224-chex",
    "xrv-densenet121-res224-mimic_nb",
    "xrv-densenet121-res224-mimic_ch",
    "xrv-densenet121-res224-all",
    "xrv-densenet121-res224-nih",
)

# chave do manifesto -> coluna de rótulo no conjunto (sem o prefixo y_)
NIH_TEST_MAP = {
    k: k
    for k in (
        "atelectasis",
        "cardiomegaly",
        "pleural_effusion",
        "infiltration",
        "mass",
        "nodule",
        "pneumonia",
        "pneumothorax",
        "consolidation",
        "pulmonary_edema",
        "emphysema",
        "fibrosis",
        "pleural_thickening",
        "hernia",
    )
}
GOOGLE_MAP = {
    "pneumothorax": "pneumothorax",
    "fracture": "fracture",
    "lung_opacity": "lung_opacity",  # "Airspace opacity"
    "lung_lesion": "nodule_or_mass",  # "Nodule or mass"
}

DATASETS: dict[str, dict[str, Any]] = {
    "nih-cxr14-test": {
        "dataset_id": "nih-cxr14",
        "map": NIH_TEST_MAP,
        "reference_standard": "report_nlp",
        "population": (
            "NIH Clinical Center (Bethesda, EUA); split oficial de teste do ChestX-ray14 "
            "(test_list.txt, 25596 imagens, pacientes disjuntos do split oficial de treino)"
        ),
        "caveats": [
            "Rótulos do NIH minerados de laudos por NLP; erro de rótulo documentado na literatura.",
            "O split oficial de teste é enriquecido em pacientes com seguimento (várias imagens "
            "por paciente); IC por bootstrap agrupado por paciente.",
        ],
    },
    "nih-google-adjudicated": {
        "dataset_id": "nih-cxr14-google",
        "map": GOOGLE_MAP,
        "reference_standard": "radiologist_adjudicated",
        "population": (
            "Subconjunto do teste oficial do NIH ChestX-ray14 com rótulos adjudicados por "
            "radiologistas (Majkowska et al., Radiology 2020; google_labels_joined.csv)"
        ),
        "caveats": [
            "Rótulos do Google: 4 classes; 'Nodule or mass' comparado a lung_lesion e "
            "'Airspace opacity' a lung_opacity (definições próximas, não idênticas).",
        ],
    },
}

KEEP = ("image_id", "patient_id", "sex", "age", "view")


def _manifest(mid: str) -> dict[str, Any]:
    data = yaml.safe_load((MANIFESTS / f"{mid}.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _labels(man: dict[str, Any]) -> dict[str, str]:
    """chave -> condition_id para os rótulos binários com condição de catálogo."""
    out = {}
    for lab in man.get("labels", []):
        if lab.get("condition_id") and lab.get("kind") == "binary":
            out[str(lab["key"])] = str(lab["condition_id"])
    return out


def _pairs(mid: str, ds: str) -> list[tuple[str, str, str]]:
    """(chave, coluna do conjunto, condition_id) para o par modelo x conjunto."""
    labels = _labels(_manifest(mid))
    m = DATASETS[ds]["map"]
    return [(k, m[k], c) for k, c in sorted(labels.items()) if k in m]


def convert() -> int:
    import pandas as pd

    for mid in MODELS:
        for ds in DATASETS:
            src = PREDS_ROOT / mid / ds / "predictions.parquet"
            if not src.is_file():
                print(f"AUSENTE: {src}", file=sys.stderr)
                return 1
            df = pd.read_parquet(src)
            out = df[[c for c in KEEP if c in df.columns]].copy()
            out["patient_id"] = out["patient_id"].astype(str)
            for key, ycol, _ in _pairs(mid, ds):
                out[f"y_true__{key}"] = df[f"y_{ycol}"]
                out[f"y_score__{key}"] = df[f"logit_{key}"]
            dst = src.with_name("evidence.parquet")
            out.to_parquet(dst, index=False)
            print(f"{mid}/{ds}: {len(out)} linhas, {len(_pairs(mid, ds))} rótulos -> {dst}")
    return 0


def plan() -> int:
    jobs = []
    for mid in MODELS:
        man = _manifest(mid)
        fname = man["weights"][0]["filename"]
        for ds, info in DATASETS.items():
            for key, _ycol, cond in _pairs(mid, ds):
                jobs.append(
                    {
                        "id": f"{mid}@{ds}#{key}",
                        "status": "measure",
                        "model_id": mid,
                        "dataset_id": info["dataset_id"],
                        "label": key,
                        "condition_id": cond,
                        "predictions": f"{mid}/{ds}/evidence.parquet",
                        "label_column": key,
                        "score_scale": "logit",
                        "training_data": list(man.get("training_data", [])),
                        "reference_standard": info["reference_standard"],
                        "population": info["population"],
                        "weights": {"model": str(WEIGHTS_DIR / fname)},
                        "manifest": mid,
                        "caveats": [
                            *info["caveats"],
                            "Rota principal dos modelos xrv definida antes desta medição "
                            "(commit 9961292), sem comparação de desempenho no NIH registrada.",
                        ],
                    }
                )
    yaml.safe_dump(jobs, sys.stdout, allow_unicode=True, sort_keys=False, width=100)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("convert", "plan"))
    args = ap.parse_args(argv)
    return convert() if args.command == "convert" else plan()


if __name__ == "__main__":
    raise SystemExit(main())
