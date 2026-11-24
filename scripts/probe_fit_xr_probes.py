#!/usr/bin/env python
"""Sondas lineares de RX sobre embeddings congelados do RAD-DINO — extração e ajuste.

Duas etapas:

``embed --dataset NOME`` (pesada; rode sob ``/root/medai-cache/bin/heavy``)
    Extrai o ``pooler_output`` do RAD-DINO com o pré-processamento do adaptador
    (``xr_raddino.raddino_canonical``) e grava ``<eval-preds>/_embeddings/rad-dino/<dataset>.npz``
    (ids, sha256 das imagens, embeddings). ``--reuse-spike`` reaproveita os embeddings do
    spike só para imagens cuja entrada canônica é byte a byte idêntica e após conferir o
    modelo em 8 imagens. Retomável: pula se o arquivo existe.

``fit`` (leve, determinística)
    Ajusta as sondas com :func:`radiologyai.analysis.probes.fit.fit_probe`, grava os
    artefatos (``safetensors`` + JSON) em ``<weights>/<probe-id>/``, gera o manifesto
    ``xr-probe-*.yaml`` com os pontos de operação fora-da-dobra do treino, fixa os
    hashes (``pin``) e grava as predições do conjunto de teste por imagem em
    ``<eval-preds>/<probe-id>/<dataset>/<rótulo>.parquet`` para a fábrica de evidência.

Desenhos (pré-especificados):

* TB: treino Shenzhen (todo), teste externo Montgomery; e o inverso (treino
  Montgomery, teste Shenzhen) como sonda de comparação não distribuída.
* Fratura (MSK): treino FracAtlas (amostra de 2 mil imagens do spike; o dataset não
  publica id de paciente —
  dobras internas por imagem, possível vazamento intrapaciente no ajuste de C e do
  ponto de operação), teste externo pediátrico GRAZPEDWRI-DX (amostra local).
* Osteopenia, metal, gesso: só GRAZPEDWRI-DX (amostra local), divisão por paciente
  70/30 determinística (sha256 do id do paciente); avaliação interna (nível C).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from predict_xr_common import (  # noqa: E402
    EVAL_PREDS,
    REPO,
    WEIGHTS_DIR,
    command_line,
    git_sha,
    raster_image,
    write_json,
)

DATASETS = Path("/root/medai-cache/datasets")
ENCODER_ID = "microsoft/rad-dino@110cbc18d5133582e320b43d53bf5c44e410c936"
ENCODER_DIR = "_encoders/rad-dino"
EMB_DIR = EVAL_PREDS / "_embeddings" / "rad-dino"
MANIFEST_DIR = REPO / "src" / "radiologyai" / "analysis" / "manifests"
PROTOCOL_SENS = 0.90
PROTOCOL_SPEC = 0.90
SEED = 20261124


# --------------------------------------------------------------------------- #
# Índices dos datasets
# --------------------------------------------------------------------------- #
def index(dataset: str) -> Any:
    import pandas as pd

    if dataset in ("tb-shenzhen", "tb-montgomery"):
        d = pd.read_csv(DATASETS / dataset / "index.tsv", sep="\t", dtype={"image_id": str})
        d["path"] = [str(DATASETS / dataset / "raw" / "CXR_png" / f"{i}.png") for i in d.image_id]
        d["patient_id"] = d["case_number"].astype(str)
        d["sex"] = d["sex"].where(d["sex"].isin(["F", "M"]), None)
        d["age"] = pd.to_numeric(d["age_years"], errors="coerce")
        d["view"] = "PA"
        d["y_tuberculosis"] = d["label"].astype(int)
        return d
    if dataset == "fracatlas":
        root = DATASETS / "fracatlas" / "raw" / "FracAtlas"
        d = pd.read_csv(root / "dataset.csv", dtype={"image_id": str})
        sub = d["fractured"].map({1: "Fractured", 0: "Non_fractured"})
        d["path"] = [str(root / "images" / s / i) for s, i in zip(sub, d.image_id, strict=True)]
        d["patient_id"] = d["image_id"]  # FracAtlas não publica id de paciente
        d["sex"], d["age"] = None, np.nan
        d["view"] = np.where(
            d.frontal == 1, "frontal", np.where(d.lateral == 1, "lateral", "oblique")
        )
        d["y_fracture"] = d["fractured"].astype(int)
        return d
    if dataset == "grazpedwri":
        d = pd.read_csv(DATASETS / "grazpedwri" / "dataset.csv")
        files = {p.stem: p for p in (DATASETS / "grazpedwri" / "images_sample").glob("*.png")}
        d = d[d.filestem.isin(files)].copy()
        d["image_id"] = d["filestem"]
        d["path"] = [str(files[s]) for s in d.filestem]
        d["patient_id"] = d["patient_id"].astype(str)
        d["sex"] = d["gender"].where(d["gender"].isin(["F", "M"]), None)
        d["view"] = d["projection"].map({1: "AP", 2: "LATERAL", 3: "OBLIQUE"}).fillna("unknown")
        for col in ("fracture_visible", "osteopenia", "metal", "cast"):
            d[f"y_{col}"] = d[col].fillna(0).astype(int)
        d = d.rename(columns={"y_fracture_visible": "y_fracture"})
        return d.sort_values("image_id").reset_index(drop=True)
    raise SystemExit(f"dataset desconhecido {dataset}")


# --------------------------------------------------------------------------- #
# Etapa 1: embeddings
# --------------------------------------------------------------------------- #
SQUARE = {"tb-shenzhen": "crop", "tb-montgomery": "crop", "fracatlas": "pad", "grazpedwri": "pad"}
SPIKE_NAME = {
    "tb-shenzhen": "shenzhen",
    "tb-montgomery": "montgomery",
    "fracatlas": "fracatlas2k",
    "grazpedwri": "grazpedwri",
}
SPIKE = Path("/root/medai-cache/spikes/encoders")
N_MODEL_CHECK = 8
REUSE_TOL = 1e-3


def _encoder() -> tuple[Any, dict[str, str]]:
    from radiologyai.analysis.adapters.xr_raddino import build_raddino

    files = {
        "encoder_config": WEIGHTS_DIR / ENCODER_DIR / "config.json",
        "encoder_weights": WEIGHTS_DIR / ENCODER_DIR / "model.safetensors",
    }
    data = {k: p.read_bytes() for k, p in files.items()}
    shas = {k: hashlib.sha256(v).hexdigest() for k, v in data.items()}
    return build_raddino(data["encoder_config"], data["encoder_weights"]), shas


def embed(dataset: str, reuse: bool) -> None:
    """Embeddings RAD-DINO com o pré-processamento do adaptador.

    ``reuse``: reaproveita os embeddings do spike SOMENTE para as imagens cuja entrada
    canônica (uint8 518x518) é byte a byte idêntica à do adaptador, e confere que o
    modelo reproduz o embedding em N imagens; imagens divergentes ou ilegíveis saem.
    """
    import torch

    from radiologyai.analysis.adapters.xr_raddino import (
        raddino_canonical,
        raddino_embeddings,
        raddino_tensor,
    )

    out = EMB_DIR / f"{dataset}.npz"
    if out.is_file():
        print(f"{out} já existe")
        return
    torch.set_num_threads(4)
    d = index(dataset)
    model, shas = _encoder()
    started = time.monotonic()
    record: dict[str, Any] = {
        "dataset": dataset,
        "encoder": ENCODER_ID,
        "square": SQUARE[dataset],
        "weights_sha256": shas,
        "git_sha": git_sha(),
        "command": command_line(),
    }
    keep, image_sha, embs = [], [], []
    if reuse:
        spike_idx = pd_read_tsv(SPIKE / "cache" / f"{SPIKE_NAME[dataset]}_index.tsv")
        canon_cache = np.load(SPIKE / "cache" / f"{SPIKE_NAME[dataset]}_518.npy", mmap_mode="r")
        emb_cache = np.load(SPIKE / "emb" / f"{SPIKE_NAME[dataset]}__raddino.npy")
        pos = {str(i): k for k, i in enumerate(spike_idx["image_id"].astype(str))}
        mismatch, unreadable, checked = [], [], []
        for i, (image_id, path) in enumerate(
            zip(d["image_id"].astype(str), d["path"], strict=True)
        ):
            if image_id not in pos:
                continue
            data = Path(path).read_bytes()
            try:
                canon = raddino_canonical(raster_image(data), SQUARE[dataset])
            except Exception as exc:  # noqa: BLE001 - imagem ilegível sai do conjunto
                unreadable.append(f"{image_id}: {type(exc).__name__}")
                continue
            k = pos[image_id]
            if not np.array_equal(canon, canon_cache[k]):
                mismatch.append(image_id)
                continue
            keep.append(i)
            image_sha.append(hashlib.sha256(data).hexdigest())
            embs.append(emb_cache[k])
            if len(checked) < N_MODEL_CHECK and k % max(1, len(spike_idx) // N_MODEL_CHECK) == 0:
                mine = raddino_embeddings(model, raddino_tensor(canon))[0]
                checked.append(float(np.abs(mine - emb_cache[k]).max()))
        if not checked or max(checked) > REUSE_TOL:
            raise SystemExit(
                f"{dataset}: embedding do modelo diverge do spike ({checked}); recalcule"
            )
        record.update(
            {
                "mode": "reused_spike_embeddings_verified",
                "n_canonical_mismatch": len(mismatch),
                "unreadable": unreadable,
                "model_check_max_abs_diff": checked,
            }
        )
    else:
        batch, rows = [], []
        for i, path in enumerate(d["path"]):
            data = Path(path).read_bytes()
            try:
                canon = raddino_canonical(raster_image(data), SQUARE[dataset])
            except Exception as exc:  # noqa: BLE001
                record.setdefault("unreadable", []).append(f"{d['image_id'].iloc[i]}: {exc}")
                continue
            batch.append(canon)
            rows.append(i)
            image_sha.append(hashlib.sha256(data).hexdigest())
            if len(batch) == 4:
                embs.extend(raddino_embeddings(model, raddino_tensor(np.stack(batch))))
                batch = []
            if (i + 1) % 64 == 0:
                print(
                    f"[{dataset}] {i + 1}/{len(d)} {(i + 1) / (time.monotonic() - started):.2f} "
                    "img/s",
                    flush=True,
                )
        if batch:
            embs.extend(raddino_embeddings(model, raddino_tensor(np.stack(batch))))
        keep = rows
        record["mode"] = "computed"
    emb = np.stack(embs).astype(np.float32)
    ids = d["image_id"].astype(str).to_numpy()[keep]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, image_id=ids, image_sha256=np.array(image_sha), emb=emb)
    tmp.replace(out)
    record.update(
        {
            "n": int(len(ids)),
            "dim": int(emb.shape[1]),
            "seconds": round(time.monotonic() - started, 1),
        }
    )
    write_json(out.with_suffix(".json"), record)
    print(f"{dataset}: {emb.shape} -> {out}")


def pd_read_tsv(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path, sep="\t", dtype={"image_id": str})


def load_emb(dataset: str) -> tuple[Any, np.ndarray]:
    d = index(dataset)
    z = np.load(EMB_DIR / f"{dataset}.npz")
    d = d.set_index(d["image_id"].astype(str)).loc[list(z["image_id"])].reset_index(drop=True)
    d["image_sha256"] = z["image_sha256"]
    return d, z["emb"].astype(np.float64)


# --------------------------------------------------------------------------- #
# Etapa 2: ajuste
# --------------------------------------------------------------------------- #
def patient_split(patients: Any, test_fraction: float = 0.3) -> np.ndarray:
    """Máscara de TESTE determinística por paciente (sha256 do id)."""
    h = np.array(
        [int(hashlib.sha256(f"{SEED}:{p}".encode()).hexdigest()[:8], 16) for p in patients]
    )
    return (h / 0xFFFFFFFF) < test_fraction


def save_preds(
    model_id: str, dataset: str, label: str, d: Any, score: np.ndarray, note: str
) -> Path:
    import pandas as pd

    cols = ["patient_id", "image_id", "image_sha256", "sex", "age", "view"]
    frame = pd.DataFrame({c: d[c].to_numpy() for c in cols})
    frame["y_true"] = d[f"y_{label}"].to_numpy().astype(int)
    frame["y_score"] = score.astype(np.float64)
    out = EVAL_PREDS / model_id / dataset / f"{label}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    write_json(
        out.with_suffix(".json"),
        {
            "model_id": model_id,
            "dataset": dataset,
            "label": label,
            "n": len(frame),
            "n_positive": int(frame.y_true.sum()),
            "n_patients": int(frame.patient_id.nunique()),
            "score_kind": "logit",
            "split": note,
            "git_sha": git_sha(),
            "command": command_line(),
            "encoder": ENCODER_ID,
        },
    )
    print(f"  preds -> {out} (n={len(frame)}, pos={int(frame.y_true.sum())})")
    return out


def fit_one(
    model_id: str,
    label: str,
    train: Any,
    x_train: np.ndarray,
    source: str,
    train_desc: str,
    square: str,
) -> tuple[Any, Any]:
    from radiologyai.analysis.probes.artifact import ProbeArtifact, training_data_sha256
    from radiologyai.analysis.probes.fit import fit_probe, oof_operating_points

    y = train[f"y_{label}"].to_numpy().astype(int)
    fit = fit_probe(x_train, y, train["patient_id"].to_numpy(), seed=SEED)
    ops = oof_operating_points(
        fit.oof_logit,
        y,
        sensitivity_target=PROTOCOL_SENS,
        specificity_target=PROTOCOL_SPEC,
        source=f"{source} ({model_id})",
    )
    artifact = ProbeArtifact(
        label=label,
        encoder_id=ENCODER_ID,
        embedding_dim=x_train.shape[1],
        mean=fit.mean,
        scale=fit.scale,
        coef=fit.coef,
        intercept=fit.intercept,
        training_data_sha256=training_data_sha256(
            list(zip(train["image_sha256"].tolist(), y.tolist(), strict=True))
        ),
        operating_points=tuple(ops),
        meta={
            "C": fit.C,
            "n_train": int(len(y)),
            "n_train_positive": int(y.sum()),
            "training_set": train_desc,
            "inner_folds": fit.n_folds,
            "seed": fit.seed,
            "recipe": "radiologyai.analysis.probes.fit.fit_probe",
            "square": square,
        },
    )
    return fit, artifact


RADDINO_REV = "110cbc18d5133582e320b43d53bf5c44e410c936"
RADDINO_FILES = {"encoder_weights": "model.safetensors", "encoder_config": "config.json"}


def write_probe(model_id: str, artifact: Any, manifest: dict[str, Any]) -> None:
    import yaml

    from radiologyai.analysis.probes.artifact import save_probe
    from radiologyai.analysis.registry import pin

    target = WEIGHTS_DIR / model_id
    save_probe(artifact, target / "probe")
    for fname in RADDINO_FILES.values():
        link = target / fname
        if not link.exists():
            link.symlink_to((WEIGHTS_DIR / ENCODER_DIR / fname).resolve())
    path = MANIFEST_DIR / f"{model_id}.yaml"
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    pin(path, WEIGHTS_DIR, download=False)
    print(f"  manifesto -> {path}")


def probe_manifest(
    model_id: str,
    label: str,
    condition: str,
    meaning: str,
    exam: str,
    applic: dict[str, Any],
    train_data: list[str],
    data_terms: str,
    ops: list[dict[str, Any]],
    summary: str,
    limitations: list[str],
    gates: list[str],
    square: str,
    route: str = "main",
) -> dict[str, Any]:
    hf = f"https://huggingface.co/microsoft/rad-dino/resolve/{RADDINO_REV}"
    local = f"https://artifacts.invalid/radiologyai/{model_id}"
    note_local = (
        "produzido localmente por scripts/probe_fit_xr_probes.py; sem URL pública "
        "(domínio .invalid): pull não baixa, só confere o arquivo local"
    )
    return {
        "id": model_id,
        "version": f"probe-{time.strftime('%Y%m%d')}-rad-dino-{RADDINO_REV[:8]}",
        "name": f"Sonda linear ({label}) sobre RAD-DINO congelado",
        "status": "available",
        "adapter": "radiologyai.analysis.adapters.xr_raddino:RadDinoLinearProbeAdapter",
        "task": "classification",
        "capabilities": ["classify"],
        "summary": summary,
        "route": route,
        "source": {
            "encoder": ENCODER_ID,
            "encoder_repository": "https://huggingface.co/microsoft/rad-dino",
            "fit_script": "scripts/probe_fit_xr_probes.py",
            "recipe": "radiologyai.analysis.probes.fit.fit_probe",
            "encoder_selection": "scripts/probe_fit_xr_compare_encoders.py",
        },
        "weights": [
            {
                "key": "encoder_weights",
                "url": f"{hf}/model.safetensors",
                "filename": "model.safetensors",
                "sha256": "",
                "bytes": 0,
                "note": "RAD-DINO (Dinov2Model) congelado",
            },
            {
                "key": "encoder_config",
                "url": f"{hf}/config.json",
                "filename": "config.json",
                "sha256": "",
                "bytes": 0,
            },
            {
                "key": "probe_tensors",
                "url": f"{local}/probe.safetensors",
                "filename": "probe.safetensors",
                "sha256": "",
                "bytes": 0,
                "note": note_local,
            },
            {
                "key": "probe_meta",
                "url": f"{local}/probe.json",
                "filename": "probe.json",
                "sha256": "",
                "bytes": 0,
                "note": note_local,
            },
        ],
        "license": {
            "code": "MIT (RAD-DINO) + Apache-2.0 (este projeto)",
            "weights": "RAD-DINO: MIT; cartão: 'shared for research purposes only ... not "
            "meant to be used for clinical practice'; sonda: este projeto",
            "training_data_terms": data_terms,
            "commercial_use": "no",
            "evidence": "cartão e LICENSE de microsoft/rad-dino no Hugging Face "
            f"(revisão {RADDINO_REV}); DATASET.json dos dados de treino",
        },
        "applicability": applic,
        "inputs": {"mode": "single", "prompts": "none"},
        "depends_on": gates,
        "training_data": train_data,
        "labels": [
            {
                "key": label,
                "source_label": label,
                "condition_id": condition,
                "kind": "binary",
                "score_kind": "logit",
                "code": {"scheme": "99MEDAI", "value": condition, "meaning": meaning},
            }
        ],
        "operating_points": [{"label": label, "point": op} for op in ops],
        "limitations": limitations,
        "references": [
            {
                "title": "Exploring scalable medical image encoders beyond text "
                "supervision (RAD-DINO)",
                "identifier": "doi:10.1038/s42256-024-00965-w",
                "url": "https://doi.org/10.1038/s42256-024-00965-w",
            }
        ],
        "config": {"square": square, "gates": gates, "exam": exam},
    }


CXR_APPLIC = {
    "exam_types": ["cxr"],
    "modalities": ["CR", "DX"],
    "image_kinds": ["projection"],
    "views": ["PA", "AP"],
    "min_pixels": 224,
    "adult_only": True,
}
MSK_APPLIC = {
    "exam_types": ["msk_xr"],
    "modalities": ["CR", "DX"],
    "image_kinds": ["projection"],
    "min_pixels": 224,
}
CXR_GATES = ["xrv-view-xinario", "xrv-ae-ood"]


def score(artifact: Any, x: np.ndarray) -> np.ndarray:
    from radiologyai.analysis.probes.artifact import probe_logit

    return probe_logit(artifact, x)


def fit_all() -> None:
    shz, x_shz = load_emb("tb-shenzhen")
    mcu, x_mcu = load_emb("tb-montgomery")
    # TB: Shenzhen -> Montgomery (distribuída)
    mid = "xr-probe-tb"
    print(mid)
    fit, art = fit_one(
        mid,
        "tuberculosis",
        shz,
        x_shz,
        "pontos fora-da-dobra Shenzhen",
        "NLM Shenzhen (todo)",
        "crop",
    )
    save_preds(
        mid,
        "tb-montgomery",
        "tuberculosis",
        mcu,
        score(art, x_mcu),
        "externo: treino Shenzhen, teste Montgomery inteiro",
    )
    save_preds(
        mid,
        "tb-shenzhen",
        "tuberculosis",
        shz,
        fit.oof_logit,
        "interno: predições fora-da-dobra (CV interna) no próprio treino",
    )
    write_probe(
        mid,
        art,
        probe_manifest(
            mid,
            "tuberculosis",
            "cxr.tuberculosis",
            "pulmonary tuberculosis (suggested)",
            "cxr",
            CXR_APPLIC,
            ["tb-shenzhen"],
            "NLM Shenzhen Hospital CXR Set (termos da NLM)",
            list(art.operating_points),
            "Escore de tuberculose pulmonar em RX de tórax: sonda linear sobre embeddings da "
            "RAD-DINO (treino Shenzhen). Achado sugerido para confirmação médica.",
            [
                "Treinada num único centro (Shenzhen, China); rótulo normal x manifestação de TB.",
                "Não substitui baciloscopia/TRM-TB; não é ferramenta de triagem validada "
                "(WHO TPP).",
                "Pontos de operação das predições fora-da-dobra do treino; não calibrados para "
                "outra prevalência.",
            ],
            CXR_GATES,
            "crop",
        ),
    )
    # TB inverso (comparação; não distribuída)
    mid_r = "xr-probe-tb.trained-montgomery"
    fit_r, art_r = fit_one(
        mid_r,
        "tuberculosis",
        mcu,
        x_mcu,
        "pontos fora-da-dobra Montgomery",
        "NLM Montgomery (todo)",
        "crop",
    )
    save_preds(
        mid_r,
        "tb-shenzhen",
        "tuberculosis",
        shz,
        score(art_r, x_shz),
        "externo: treino Montgomery, teste Shenzhen inteiro",
    )

    # Fratura: FracAtlas -> GRAZPEDWRI (externo pediátrico)
    fa, x_fa = load_emb("fracatlas")
    gz, x_gz = load_emb("grazpedwri")
    mid = "xr-probe-fracture"
    print(mid)
    fit, art = fit_one(
        mid,
        "fracture",
        fa,
        x_fa,
        "pontos fora-da-dobra FracAtlas",
        "FracAtlas (amostra de 2 mil imagens do spike)",
        "pad",
    )
    save_preds(
        mid,
        "fracatlas",
        "fracture",
        fa,
        fit.oof_logit,
        "interno: predições fora-da-dobra; dobras por imagem (sem id de paciente)",
    )
    save_preds(
        mid,
        "grazpedwri-sample",
        "fracture",
        gz,
        score(art, x_gz),
        "externo pediátrico: amostra local do GRAZPEDWRI-DX",
    )
    write_probe(
        mid,
        art,
        probe_manifest(
            mid,
            "fracture",
            "msk_xr.fracture",
            "bone fracture (suggested)",
            "msk_xr",
            MSK_APPLIC,
            ["fracatlas"],
            "FracAtlas (CC BY 4.0)",
            list(art.operating_points),
            "Escore de fratura em radiografia musculoesquelética: sonda linear sobre embeddings da "
            "RAD-DINO treinada no FracAtlas.",
            [
                "Codificador RAD-DINO pré-treinado só em RX de tórax; sonda treinada no FracAtlas.",
                "FracAtlas não publica id de paciente: o ajuste interno pode ter vazamento "
                "intrapaciente.",
                "Não localiza a fratura.",
            ],
            [],
            "pad",
        ),
    )

    # GRAZPEDWRI: osteopenia, metal, gesso (interno, divisão por paciente)
    test_mask = patient_split(gz["patient_id"])
    for label, meaning in (
        ("osteopenia", "osteopenia (suggested)"),
        ("metal", "metal implant or foreign body (suggested)"),
        ("cast", "plaster cast (suggested)"),
    ):
        mid = f"xr-probe-grazpedwri-{label}"
        print(mid)
        train, test = gz[~test_mask], gz[test_mask]
        fit, art = fit_one(
            mid,
            label,
            train,
            x_gz[~test_mask],
            "pontos fora-da-dobra GRAZPEDWRI-DX (treino 70% por paciente)",
            "GRAZPEDWRI-DX amostra local, 70% dos pacientes",
            "pad",
        )
        save_preds(
            mid,
            "grazpedwri-sample-test30",
            label,
            test,
            score(art, x_gz[test_mask]),
            "interno: 30% dos pacientes (sha256 do id) da amostra do GRAZPEDWRI-DX",
        )
        write_probe(
            mid,
            art,
            probe_manifest(
                mid,
                label,
                f"msk_xr.{label}",
                meaning,
                "msk_xr",
                MSK_APPLIC,
                ["grazpedwri-dx"],
                "GRAZPEDWRI-DX (CC BY 4.0)",
                list(art.operating_points),
                f"Escore de {label} em radiografia de punho pediátrica: sonda linear sobre "
                "embeddings do RAD-DINO (GRAZPEDWRI-DX).",
                [
                    "Treinada e avaliada só em punho pediátrico de um centro (Graz); fora "
                    "disso não "
                    "foi medida.",
                    "Avaliação interna (divisão por paciente): no máximo nível C.",
                    "Roteada só no modo pesquisa: o manifesto não consegue restringir região "
                    "anatômica nem faixa etária pediátrica.",
                ],
                [],
                "pad",
                route="research",
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument(
        "--dataset",
        required=True,
        choices=["tb-shenzhen", "tb-montgomery", "fracatlas", "grazpedwri"],
    )
    e.add_argument("--reuse-spike", action="store_true")
    sub.add_parser("fit")
    args = parser.parse_args()
    if args.cmd == "embed":
        embed(args.dataset, args.reuse_spike)
    else:
        fit_all()


if __name__ == "__main__":
    main()
