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
OPENI_MAP = {
    k: k
    for k in (
        "atelectasis",
        "cardiomegaly",
        "pulmonary_edema",
        "pleural_effusion",
        "emphysema",
        "fibrosis",
        "fracture",
        "hernia",
        "infiltration",
        "lung_lesion",
        "mass",
        "nodule",
        "lung_opacity",
        "pleural_thickening",
        "pneumonia",
        "pneumothorax",
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
            "Pesos pc escolhidos como modelo de validação externa em c80c22e (2026-09-09), "
            "antes de qualquer medição; o mesmo teste oficial do NIH foi medido com pc em "
            "2026-09-12 (artifacts/eval/xrv-densenet121-pc__20260912T215742Z, cardiomegalia "
            "0,796), antes de o protocolo ser escrito; não houve comparação entre modelos no NIH.",
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
            "Subconjunto enriquecido (Majkowska 2020): prevalência de opacidade ~58%; limiares, "
            "VPP e VPN ajustados nesta prevalência não se transferem à prática clínica.",
            "Pesos pc escolhidos como modelo de validação externa em c80c22e (2026-09-09), "
            "antes de qualquer medição; o mesmo teste oficial do NIH foi medido com pc em "
            "2026-09-12 (artifacts/eval/xrv-densenet121-pc__20260912T215742Z, cardiomegalia "
            "0,796), antes de o protocolo ser escrito; não houve comparação entre modelos no NIH.",
        ],
    },
    "openi-frontal": {
        "dataset_id": "openi",
        "map": OPENI_MAP,
        "reference_standard": "other",
        "population": (
            "Indiana University / Open-i (NLMCXR, EUA); todas as radiografias frontais (PA/AP) "
            "com PNG do índice local: 4014 imagens de 2561 laudos"
        ),
        "caveats": [
            "Rótulos: códigos MeSH atribuídos por curadores a partir do laudo (via "
            "torchxrayvision); não é leitura adjudicada da imagem.",
            "O Open-i não publica id de paciente: o agrupamento (divisão calibração/teste e "
            "bootstrap) é por laudo (uid); um paciente com mais de um laudo pode cair nas duas "
            "metades, o que tende a otimizar a estimativa.",
            "Pesos pc escolhidos como modelo de validação externa em c80c22e (2026-09-09), antes "
            "de qualquer medição no Open-i; não houve comparação entre modelos no Open-i.",
            "138 frontais do Open-i entraram na calibração do limiar do portão OOD "
            "(eval-preds/_gates/summary.json); o portão não é aplicado nesta medição.",
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


def convert(datasets: list[str]) -> int:
    import pandas as pd

    for mid in MODELS:
        for ds in datasets:
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


def plan(datasets: list[str], status: str, reason: str | None) -> int:
    jobs = []
    for mid in MODELS:
        man = _manifest(mid)
        fname = man["weights"][0]["filename"]
        for ds in datasets:
            info = DATASETS[ds]
            for key, _ycol, cond in _pairs(mid, ds):
                head: dict[str, Any] = {"id": f"{mid}@{ds}#{key}", "status": status}
                if reason:
                    head["reason"] = reason
                jobs.append(
                    {
                        **head,
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
                            "Inferência feita em fragmentos ao longo de vários commits (código de "
                            "inferência e adaptador sem mudança funcional entre eles); o checkout "
                            "limpo vale para o cálculo das métricas.",
                        ],
                    }
                )
    yaml.safe_dump(jobs, sys.stdout, allow_unicode=True, sort_keys=False, width=100)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("convert", "plan"))
    ap.add_argument("--datasets", default=",".join(DATASETS), help="separados por vírgula")
    ap.add_argument("--status", choices=("measure", "pending"), default="measure")
    ap.add_argument("--reason", help="obrigatório com --status pending")
    args = ap.parse_args(argv)
    datasets = [d for d in args.datasets.split(",") if d]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        ap.error(f"conjuntos desconhecidos: {unknown}")
    if args.status == "pending" and not args.reason:
        ap.error("--status pending exige --reason")
    if args.command == "convert":
        return convert(datasets)
    return plan(datasets, args.status, args.reason)


if __name__ == "__main__":
    raise SystemExit(main())
