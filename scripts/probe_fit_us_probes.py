#!/usr/bin/env python3
"""Sondas lineares de US sobre embeddings congelados do BiomedCLIP — extração e ajuste.

Duas etapas:

``embed`` (pesada; rode sob ``/root/medai-cache/bin/heavy``)
    Para cada dataset, aplica o pré-processamento do adaptador
    (:func:`radiologyai.analysis.adapters.us_probe.us_canonical`) e grava
    ``<eval-preds>/_embeddings/biomedclip/<dataset>.npz`` (ids, sha256 da imagem
    de origem, dHash de 64 bits da imagem inteira, rótulo, grupo, embedding 512).

``fit`` (leve, determinística)
    Ajusta as sondas com :func:`radiologyai.analysis.probes.fit.fit_probe`
    (CV interna agrupada para ``C``; pontos de operação nas predições
    fora-da-dobra do treino), grava os artefatos em ``<weights>/<probe-id>/``,
    escreve e fixa (``pin``) os manifestos ``us-probe-*.yaml`` e grava as
    predições em ``<eval-preds>/<probe-id>/<dataset>/<rótulo>.parquet``.

Desenhos (pré-especificados):

* ``us-probe-breast-malignancy`` — lesão mamária benigna x maligna, região =
  caixa da máscara de REFERÊNCIA ampliada 25 % (substitui a caixa do usuário:
  oráculo). Treino: BUSI deduplicado (sem "normal"); o BUSI NÃO publica id de
  paciente. Duplicatas (dHash de 256 bits, Hamming <= 6) são reduzidas a um
  representante e aglomerados de duplicatas com rótulos conflitantes são
  descartados; grupos da CV = aglomerados mais largos (Hamming <= 24),
  aproximação de "mesmo paciente/aquisição". Teste EXTERNO: BUS-BRA (id de paciente =
  ``Case``) e BrEaST-Lesions-USG (um paciente por imagem); imagens de teste
  quase idênticas a alguma do treino são removidas. Predições internas
  fora-da-dobra do BUSI (CV aninhada) também são gravadas.
* ``us-probe-liver-steatosis`` — esteatose (gordura >= 5 % na biópsia; rótulo
  ``class`` do dataset) no Byra et al.: 55 pacientes, 10 quadros consecutivos
  cada → só avaliação INTERNA por CV aninhada agrupada por paciente; escore
  por paciente = média dos logits dos 10 quadros (mesma agregação do adaptador
  para clipes).
* ``us-probe-liver-lesion-malignancy`` — massa hepática benigna x maligna
  (Zenodo 7272660), região = caixa do polígono da massa ampliada 25 %. Sem id de
  paciente: grupos = aglomerados de quase-duplicatas; avaliação INTERNA por CV
  aninhada (possível vazamento intrapaciente — declarado).

Uso::

    /root/medai-cache/bin/heavy /root/medai-cache/bin/timev .venv/bin/python -I \\
        scripts/probe_fit_us_probes.py embed
    .venv/bin/python -I scripts/probe_fit_us_probes.py fit
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATASETS = Path(os.environ.get("MEDAI_DATASETS", "/root/medai-cache/datasets"))
EVAL_PREDS = Path(os.environ.get("MEDAI_EVAL_PREDS", "/root/medai-cache/eval-preds"))
WEIGHTS_DIR = Path(os.environ.get("MEDAI_WEIGHTS_DIR", "/root/medai-cache/weights/registry"))
ENCODER_SNAPSHOT = Path(
    "/root/medai-cache/hf/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/"
    "snapshots/9f341de24bfb00180f1b847274256e9b65a3a32e"
)
ENCODER_FILE = "open_clip_pytorch_model.bin"
ENCODER_URL = (
    "https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/"
    "9f341de24bfb00180f1b847274256e9b65a3a32e/open_clip_pytorch_model.bin"
)
EMB_DIR = EVAL_PREDS / "_embeddings" / "biomedclip"
MANIFEST_DIR = REPO / "src" / "radiologyai" / "analysis" / "manifests"
MARGIN = 0.25
DUP_HAMMING = 6  # de 256 bits: mesma imagem (recompressão, redimensionamento)
GROUP_HAMMING = 24  # de 256 bits: mesma aquisição/paciente provável (agrupa na CV)
SENS, SPEC = 0.90, 0.90
DATASET_NAMES = ("busi", "busbra", "breast", "byra", "liver")


# --------------------------------------------------------------------------- #
# Leitura dos datasets: (id, cinza uint8 HxW, máscara/None, rótulo 0/1, grupo, sha256)
# --------------------------------------------------------------------------- #
def _gray(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def _largest(mask: np.ndarray) -> np.ndarray:
    from radiologyai.analysis.adapters.us_common import clean_mask

    return clean_mask(mask)


def _items(name: str) -> Iterator[dict[str, Any]]:
    import pandas as pd

    if name == "busi":
        df = pd.read_parquet(DATASETS / "busi/raw/train-00000-of-00001.parquet")
        for r in df.itertuples():
            if r.class_label not in ("benign", "malignant"):
                continue
            raw = r.image["bytes"]
            mask = _gray(r.mask["bytes"]) > 127
            yield {
                "id": str(r.image_id),
                "gray": _gray(raw),
                "mask": _largest(mask),
                "y": int(r.class_label == "malignant"),
                "group": "",
                "sha": hashlib.sha256(raw).hexdigest(),
            }
    elif name == "busbra":
        base = DATASETS / "bus-bra/extracted/BUSBRA"
        df = pd.read_csv(base / "bus_data.csv")
        for r in df.itertuples():
            raw = (base / "Images" / f"{r.ID}.png").read_bytes()
            mask = _gray((base / "Masks" / f"mask_{r.ID[4:]}.png").read_bytes()) > 127
            yield {
                "id": str(r.ID),
                "gray": _gray(raw),
                "mask": _largest(mask),
                "y": int(r.Pathology == "malignant"),
                "group": f"busbra-{r.Case}",
                "sha": hashlib.sha256(raw).hexdigest(),
            }
    elif name == "breast":
        base = DATASETS / "breast-lesions-usg"
        img_dir = base / "extracted/BrEaST-Lesions_USG-images_and_masks"
        df = pd.read_csv(base / "clinical_BrEaST-Lesions-USG_clinical_dat.csv")
        for r in df.itertuples():
            if r.Classification not in ("benign", "malignant") or not isinstance(
                r.Mask_tumor_filename, str
            ):
                continue
            raw = (img_dir / r.Image_filename).read_bytes()
            mask = _gray((img_dir / r.Mask_tumor_filename).read_bytes()) > 0
            if not mask.any():
                continue
            yield {
                "id": str(r.Image_filename),
                "gray": _gray(raw),
                "mask": _largest(mask),
                "y": int(r.Classification == "malignant"),
                "group": f"breast-{r.CaseID}",
                "sha": hashlib.sha256(raw).hexdigest(),
            }
    elif name == "byra":
        import scipy.io as sio

        mat = sio.loadmat(
            DATASETS / "byra-fatty-liver/raw/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
        )["data"][0]
        for r in mat:
            pid = int(r["id"][0, 0])
            for k, arr in enumerate(r["images"]):
                gray = np.ascontiguousarray(arr, dtype=np.uint8)
                yield {
                    "id": f"p{pid:02d}_f{k}",
                    "gray": gray,
                    "mask": None,
                    "y": int(r["class"][0, 0]),
                    "group": f"byra-{pid:02d}",
                    "sha": hashlib.sha256(gray.tobytes()).hexdigest(),
                }
    elif name == "liver":
        from PIL import Image, ImageDraw

        base = DATASETS / "liver-lesion-us-7272660/extracted"
        for cls in ("Benign", "Malignant"):
            files = sorted((base / cls / "image").glob("*.jpg"), key=lambda p: int(p.stem))
            for f in files:
                poly_path = base / cls / "segmentation/mass" / f"{f.stem}.json"
                if not poly_path.is_file():
                    continue
                raw = f.read_bytes()
                gray = _gray(raw)
                poly = json.loads(poly_path.read_text(encoding="utf-8"))
                pts = [(float(p[0]), float(p[1])) for p in poly]
                if len(pts) < 3:
                    continue
                canvas = Image.new("L", (gray.shape[1], gray.shape[0]), 0)
                ImageDraw.Draw(canvas).polygon(pts, fill=1)
                yield {
                    "id": f"{cls.lower()}/{f.name}",
                    "gray": gray,
                    "mask": np.asarray(canvas, dtype=bool),
                    "y": int(cls == "Malignant"),
                    "group": "",
                    "sha": hashlib.sha256(raw).hexdigest(),
                }
    else:
        raise SystemExit(f"dataset desconhecido {name!r}")


def dhash(gray: np.ndarray) -> int:
    """Hash de diferença 64 bits (9x8, bilinear) da imagem inteira (gravado no .npz; legado)."""
    from PIL import Image

    small = np.asarray(
        Image.fromarray(gray).resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16
    )
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def dhash256(gray: np.ndarray) -> np.ndarray:
    """Hash de diferença 256 bits (17x16, bilinear) como vetor booleano.

    O de 64 bits juntava imagens distintas de layout parecido (fígado: 635 → 46
    aglomerados); 256 bits separa duplicatas reais de imagens apenas semelhantes.
    """
    from PIL import Image

    small = np.asarray(
        Image.fromarray(gray).resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16
    )
    out: np.ndarray = (small[:, 1:] > small[:, :-1]).ravel()
    return out


def hashes256(name: str, ids: np.ndarray) -> np.ndarray:
    """dHash 256 de cada imagem do dataset, na ordem de ``ids`` (cache em disco)."""
    cache = EMB_DIR / f"{name}.dhash256.npz"
    if cache.is_file():
        with np.load(cache) as z:
            if z["ids"].tolist() == ids.tolist():
                cached: np.ndarray = z["h"]
                return cached
    by_id = {it["id"]: dhash256(it["gray"]) for it in _items(name)}
    h = np.stack([by_id[str(i)] for i in ids])
    np.savez(cache, ids=ids, h=h)
    return h


def cmd_embed(args: argparse.Namespace) -> int:
    import torch

    from radiologyai.analysis.adapters.us_common import mask_bbox
    from radiologyai.analysis.adapters.us_probe import BiomedCLIPImageEncoder, us_canonical

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    enc = BiomedCLIPImageEncoder(ENCODER_SNAPSHOT / ENCODER_FILE)
    print(f"[embed] codificador carregado em {time.time() - t0:.1f}s", flush=True)
    for name in args.datasets.split(","):
        out = EMB_DIR / f"{name}.npz"
        if out.is_file() and not args.force:
            print(f"[embed] {name}: já existe, pulando")
            continue
        t1 = time.time()
        ids, shas, hashes, ys, groups, embs = [], [], [], [], [], []
        batch: list[np.ndarray] = []

        def flush(batch: list[np.ndarray] = batch, embs: list[np.ndarray] = embs) -> None:
            if batch:
                embs.append(enc.embed(np.stack(batch)))
                batch.clear()

        for item in _items(name):
            box = None if item["mask"] is None else mask_bbox(item["mask"])
            batch.append(us_canonical(item["gray"], box, MARGIN if box is not None else 0.0))
            ids.append(item["id"])
            shas.append(item["sha"])
            hashes.append(dhash(item["gray"]))
            ys.append(item["y"])
            groups.append(item["group"])
            if len(batch) == 32:
                flush()
        flush()
        emb = np.concatenate(embs).astype(np.float32)
        np.savez(
            out,
            ids=np.asarray(ids),
            sha256=np.asarray(shas),
            dhash=np.asarray(hashes, dtype=np.uint64),
            y=np.asarray(ys, dtype=np.int64),
            group=np.asarray(groups),
            emb=emb,
        )
        meta = {
            "encoder": "BiomedCLIP visual (us_probe.BiomedCLIPImageEncoder)",
            "encoder_file_sha256": _sha256(ENCODER_SNAPSHOT / ENCODER_FILE),
            "preprocess": f"us_canonical, margin {MARGIN} (região = caixa da máscara de referência)"
            if name != "byra"
            else "us_canonical, imagem inteira",
            "n": len(ids),
            "seconds": round(time.time() - t1, 1),
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        print(f"[embed] {name}: {len(ids)} imagens em {meta['seconds']}s", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Ajuste
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(EMB_DIR / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distâncias de Hamming entre linhas de dois conjuntos de hashes booleanos."""
    ai, bi = a.astype(np.int32), b.astype(np.int32)
    out: np.ndarray = ai @ (1 - bi).T + (1 - ai) @ bi.T
    return out


def dup_clusters(hashes: np.ndarray, threshold: int = DUP_HAMMING) -> np.ndarray:
    """Aglomerados de quase-duplicatas (união-busca sobre Hamming <= limiar)."""
    n = len(hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = _hamming(hashes, hashes)
    for i, j in zip(*np.nonzero(np.triu(d <= threshold, k=1)), strict=True):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.asarray([find(i) for i in range(n)])


def dedupe(
    data: dict[str, np.ndarray], hashes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Índices mantidos (um por aglomerado de duplicatas, sem conflito de rótulo), grupos
    de CV (aglomerados de quase-duplicatas mais largos) para eles e contagens."""
    clusters = dup_clusters(hashes, DUP_HAMMING)
    wide = dup_clusters(hashes, GROUP_HAMMING)
    keep: list[int] = []
    conflicts = 0
    for c in np.unique(clusters):
        members = np.nonzero(clusters == c)[0]
        if len(set(data["y"][members].tolist())) > 1:
            conflicts += 1
            continue
        keep.append(int(members[0]))
    kept = np.asarray(sorted(keep))
    stats = {
        "n_input": len(clusters),
        "n_duplicate_clusters": len(np.unique(clusters)),
        "n_label_conflict_clusters": conflicts,
        "n_kept": len(keep),
        "n_cv_groups": len(np.unique(wide[kept])),
        "dup_hamming_256": DUP_HAMMING,
        "group_hamming_256": GROUP_HAMMING,
    }
    return kept, wide[kept], stats


def _artifact(
    label: str, x: np.ndarray, y: np.ndarray, shas: np.ndarray, groups: np.ndarray, desc: str
) -> tuple[Any, Any]:
    from radiologyai.analysis.adapters.us_probe import BIOMEDCLIP_ENCODER_ID
    from radiologyai.analysis.probes.artifact import ProbeArtifact, training_data_sha256
    from radiologyai.analysis.probes.fit import fit_probe, oof_operating_points

    fit = fit_probe(x, y, groups)
    ops = oof_operating_points(
        fit.oof_logit,
        y,
        sensitivity_target=SENS,
        specificity_target=SPEC,
        source=f"pontos fora-da-dobra {desc}",
    )
    art = ProbeArtifact(
        label=label,
        encoder_id=BIOMEDCLIP_ENCODER_ID,
        embedding_dim=int(x.shape[1]),
        mean=fit.mean,
        scale=fit.scale,
        coef=fit.coef,
        intercept=fit.intercept,
        training_data_sha256=training_data_sha256(
            list(zip(shas.tolist(), y.tolist(), strict=True))
        ),
        operating_points=tuple(ops),
        meta={
            "C": fit.C,
            "n_train": int(len(y)),
            "n_train_positive": int(y.sum()),
            "training_set": desc,
            "inner_folds": fit.n_folds,
            "seed": fit.seed,
            "recipe": "radiologyai.analysis.probes.fit.fit_probe",
            "preprocess": "radiologyai.analysis.adapters.us_probe.us_canonical",
        },
    )
    return fit, art


def nested_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_outer: int = 5) -> np.ndarray:
    """Logits fora-da-dobra de uma CV externa agrupada; sonda reajustada em cada dobra."""
    from sklearn.model_selection import StratifiedGroupKFold

    from radiologyai.analysis.probes.artifact import ProbeArtifact, probe_logit
    from radiologyai.analysis.probes.fit import fit_probe

    out = np.full(len(y), np.nan)
    splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=20261125)
    for tr, te in splitter.split(x, y, groups):
        n_inner = min(
            5, len(np.unique(groups[tr][y[tr] == 0])), len(np.unique(groups[tr][y[tr] == 1]))
        )
        fit = fit_probe(x[tr], y[tr], groups[tr], n_folds=max(2, n_inner))
        art = ProbeArtifact(
            label="oof",
            encoder_id="oof",
            embedding_dim=int(x.shape[1]),
            mean=fit.mean,
            scale=fit.scale,
            coef=fit.coef,
            intercept=fit.intercept,
            training_data_sha256="0" * 64,
        )
        out[te] = probe_logit(art, x[te])
    return out


def _save_preds(model_id: str, dataset: str, label: str, df: Any, meta: dict[str, Any]) -> None:
    d = EVAL_PREDS / model_id / dataset
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{label}.parquet", index=False)
    (d / f"{label}.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  predições -> {d / f'{label}.parquet'} (n={len(df)})")


def _manifest(
    model_id: str,
    label: str,
    condition: str,
    meaning: str,
    exam: str,
    roi: str,
    art: Any,
    training: list[str],
    summary: str,
    limitations: list[str],
) -> dict[str, Any]:
    ops = [{"label": label, "point": dict(op)} for op in art.operating_points]
    return {
        "id": model_id,
        "version": f"probe-{time.strftime('%Y%m%d')}-biomedclip-9f341de",
        "name": f"Sonda linear ({label}) sobre BiomedCLIP congelado",
        "status": "available",
        "adapter": "radiologyai.analysis.adapters.us_probe:USProbeAdapter",
        "task": "classification",
        "capabilities": ["classify"],
        "summary": summary,
        "visibility_default": "research",
        "route": "interactive" if roi == "box" else "research",
        "source": {
            "encoder": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224@"
            "9f341de24bfb00180f1b847274256e9b65a3a32e",
            "encoder_repository": "https://huggingface.co/microsoft/"
            "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "fit_script": "scripts/probe_fit_us_probes.py",
            "recipe": "radiologyai.analysis.probes.fit.fit_probe",
        },
        "weights": [
            {
                "key": "encoder_weights",
                "url": ENCODER_URL,
                "filename": ENCODER_FILE,
                "note": "checkpoint open_clip completo; só as chaves visual.* são usadas "
                "(torch.load weights_only=True, strict=True)",
                "sha256": "",
            },
            {
                "key": "probe_tensors",
                "url": f"https://artifacts.invalid/radiologyai/{model_id}/probe.safetensors",
                "filename": "probe.safetensors",
                "note": "produzido localmente por scripts/probe_fit_us_probes.py; sem URL pública "
                "(domínio .invalid): pull não baixa, só confere o arquivo local",
                "sha256": "",
            },
            {
                "key": "probe_meta",
                "url": f"https://artifacts.invalid/radiologyai/{model_id}/probe.json",
                "filename": "probe.json",
                "note": "produzido localmente por scripts/probe_fit_us_probes.py; sem URL pública "
                "(domínio .invalid): pull não baixa, só confere o arquivo local",
                "sha256": "",
            },
        ],
        "license": {
            "code": "MIT (BiomedCLIP/open_clip) + Apache-2.0 (este projeto)",
            "weights": "BiomedCLIP: MIT (LICENSE.md do repositório); cartão restringe a pesquisa; "
            "sonda: este projeto",
            "training_data_terms": "; ".join(training),
            "commercial_use": "no",
            "evidence": "LICENSE.md e README.md de "
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 "
            "(revisão 9f341de24bfb00180f1b847274256e9b65a3a32e); DATASET.json dos dados",
        },
        "applicability": {
            "exam_types": [exam],
            "modalities": ["US"],
            "image_kinds": ["projection"] if roi == "box" else ["projection", "cine"],
            "min_pixels": 64 if roi == "box" else 128,
            "adult_only": False,
        },
        "inputs": {"mode": "single", "prompts": "required" if roi == "box" else "none"},
        "training_data": [t.split(" ")[0] for t in training],
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
        "operating_points": ops,
        "limitations": limitations,
        "references": [
            {
                "title": "BiomedCLIP: a multimodal biomedical foundation model pretrained "
                "from fifteen "
                "million scientific image-text pairs",
                "identifier": "arXiv:2303.00915",
                "url": "https://arxiv.org/abs/2303.00915",
            }
        ],
        "config": {"roi": roi, "margin": MARGIN if roi == "box" else 0.0, "max_frames": 10},
    }


def _write(model_id: str, art: Any, manifest: dict[str, Any]) -> None:
    import yaml

    from radiologyai.analysis.probes.artifact import save_probe
    from radiologyai.analysis.registry import pin

    target = WEIGHTS_DIR / model_id
    save_probe(art, target / "probe")
    link = target / ENCODER_FILE
    if not link.exists():
        link.symlink_to((ENCODER_SNAPSHOT / ENCODER_FILE).resolve())
    path = MANIFEST_DIR / f"{model_id}.yaml"
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    pin(path, WEIGHTS_DIR, download=False)
    print(f"  manifesto -> {path}")


def _frame(rows: dict[str, Any]) -> Any:
    import pandas as pd

    return pd.DataFrame(rows)


def cmd_fit(args: argparse.Namespace) -> int:  # noqa: ARG001 - interface dos subcomandos
    from radiologyai.analysis.probes.artifact import probe_logit

    report: dict[str, Any] = {}
    # ---------------- mama: BUSI -> BUS-BRA, BrEaST ----------------
    busi = _load("busi")
    busi_h = hashes256("busi", busi["ids"])
    keep, wide, stats = dedupe(busi, busi_h)
    report["busi_dedupe"] = stats
    x, y = busi["emb"][keep], busi["y"][keep]
    groups = np.asarray([f"busi-g{i}" for i in wide])  # pseudo-paciente: aglomerado largo
    label = "breast_lesion_malignant"
    model_id = "us-probe-breast-malignancy"
    _, art = _artifact(label, x, y, busi["sha256"][keep], groups, f"BUSI deduplicado ({model_id})")
    oof = nested_oof(x, y, groups)
    base_meta = {
        "model_id": model_id,
        "label": label,
        "score": "logit da sonda",
        "roi": "caixa da máscara de referência + 25% (oráculo da caixa do usuário)",
    }
    _save_preds(
        model_id,
        "busi",
        label,
        _frame(
            {
                "patient_id": groups,
                "image_id": busi["ids"][keep],
                "image_sha256": busi["sha256"][keep],
                "y_true": y,
                "y_score": oof,
            }
        ),
        {
            **base_meta,
            "design": "INTERNO: CV aninhada 5 dobras agrupada por aglomerado de "
            "quase-duplicatas; BUSI sem id de paciente (vazamento intrapaciente possível)",
            "dedupe": stats,
        },
    )
    train_hashes = busi_h
    for ds in ("busbra", "breast"):
        t = _load(ds)
        near = (_hamming(hashes256(ds, t["ids"]), train_hashes) <= DUP_HAMMING).any(axis=1)
        sel = ~near
        score = probe_logit(art, t["emb"][sel])
        _save_preds(
            model_id,
            ds,
            label,
            _frame(
                {
                    "patient_id": t["group"][sel],
                    "image_id": t["ids"][sel],
                    "image_sha256": t["sha256"][sel],
                    "y_true": t["y"][sel],
                    "y_score": score,
                }
            ),
            {
                **base_meta,
                "design": "EXTERNO: sonda final (treino BUSI inteiro deduplicado)",
                "n_removed_near_duplicates_of_training": int(near.sum()),
            },
        )
        report[f"{ds}_removed_near_dup"] = int(near.sum())
    _write(
        model_id,
        art,
        _manifest(
            model_id,
            label,
            "us.breast_lesion_malignancy",
            "breast lesion malignancy (suggested)",
            "us_breast",
            "box",
            art,
            ["busi (CC BY 4.0 segundo o espelho MedOtter/BUSI)"],
            "Escore de malignidade de lesão mamária na região de uma caixa do usuário em US de "
            "mama: "
            "sonda linear sobre BiomedCLIP (treino BUSI). Achado sugerido para confirmação médica.",
            [
                "Treino só no BUSI (Egito, um centro); sem id de paciente: quase-duplicatas "
                "agrupadas por "
                "dHash, vazamento intrapaciente ainda possível na CV interna.",
                "BUSI contém marcas de medida gravadas na imagem; a região recortada pode "
                "incluí-las.",
                "Região de treino = caixa da máscara de referência + 25%; na prática a caixa é "
                "do usuário.",
                "Pontos de operação das predições fora-da-dobra do treino; não calibrados para "
                "outra "
                "prevalência. Não substitui BI-RADS nem biópsia.",
            ],
        ),
    )
    # ---------------- fígado: esteatose (Byra) ----------------
    byra = _load("byra")
    label, model_id = "liver_steatosis", "us-probe-liver-steatosis"
    _, art = _artifact(
        label,
        byra["emb"],
        byra["y"],
        byra["sha256"],
        byra["group"],
        f"Byra 55 pacientes ({model_id})",
    )
    oof = nested_oof(byra["emb"], byra["y"], byra["group"])
    pats = np.unique(byra["group"])
    _save_preds(
        model_id,
        "byra",
        label,
        _frame(
            {
                "patient_id": pats,
                "image_id": pats,
                "y_true": [int(byra["y"][byra["group"] == p][0]) for p in pats],
                "y_score": [float(oof[byra["group"] == p].mean()) for p in pats],
            }
        ),
        {
            "model_id": model_id,
            "label": label,
            "score": "média dos logits dos 10 quadros",
            "design": "INTERNO: CV aninhada 5 dobras agrupada por paciente (55 pacientes)",
            "reference": "biópsia, gordura >= 5% (campo class do dataset)",
        },
    )
    _write(
        model_id,
        art,
        _manifest(
            model_id,
            label,
            "us.liver_steatosis",
            "hepatic steatosis (suggested)",
            "us_abdomen",
            "full",
            art,
            ["byra-fatty-liver (CC BY 4.0, Zenodo 1009146)"],
            "Escore de esteatose hepática (gordura >= 5% na biópsia) na imagem inteira de US "
            "do fígado: "
            "sonda linear sobre BiomedCLIP (treino Byra). Só modo pesquisa.",
            [
                "55 pacientes (obesos graves pré-bariátrica), um centro, transdutor setorial: "
                "espectro "
                "muito diferente do US abdominal de rotina.",
                "Avaliação só interna (CV aninhada por paciente); sem teste externo.",
                "Imagem inteira, sem região: anotações gravadas podem influenciar o escore.",
            ],
        ),
    )
    # ---------------- fígado: massa benigna x maligna ----------------
    liver = _load("liver")
    keep, wide, stats = dedupe(liver, hashes256("liver", liver["ids"]))
    report["liver_dedupe"] = stats
    x, y = liver["emb"][keep], liver["y"][keep]
    groups = np.asarray([f"liver-g{i}" for i in wide])
    label, model_id = "liver_lesion_malignant", "us-probe-liver-lesion-malignancy"
    _, art = _artifact(
        label, x, y, liver["sha256"][keep], groups, f"Zenodo 7272660 deduplicado ({model_id})"
    )
    oof = nested_oof(x, y, groups)
    _save_preds(
        model_id,
        "liver-lesion-us-7272660",
        label,
        _frame(
            {
                "patient_id": groups,
                "image_id": liver["ids"][keep],
                "image_sha256": liver["sha256"][keep],
                "y_true": y,
                "y_score": oof,
            }
        ),
        {
            "model_id": model_id,
            "label": label,
            "score": "logit da sonda",
            "roi": "caixa do polígono da massa + 25% (oráculo)",
            "design": "INTERNO: CV aninhada agrupada por aglomerado de quase-duplicatas; dataset "
            "sem id de paciente nem histologia",
            "dedupe": stats,
        },
    )
    _write(
        model_id,
        art,
        _manifest(
            model_id,
            label,
            "us.liver_lesion_malignancy",
            "liver mass malignancy (suggested)",
            "us_abdomen",
            "box",
            art,
            ["liver-lesion-us-7272660 (CC BY 4.0, Zenodo 7272660)"],
            "Escore de malignidade de massa hepática na região de uma caixa do usuário em US "
            "abdominal: "
            "sonda linear sobre BiomedCLIP. Exploratório, só modo pesquisa.",
            [
                "Dataset sem id de paciente, sem histologia e sem metadados de aparelho; "
                "rótulo = pasta.",
                "Avaliação só interna; vazamento intrapaciente possível.",
                "Região de treino = caixa do polígono de referência + 25%.",
            ],
        ),
    )
    out = EVAL_PREDS / "_embeddings" / "biomedclip" / "fit_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--datasets", default=",".join(DATASET_NAMES))
    e.add_argument("--force", action="store_true")
    sub.add_parser("fit")
    args = p.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    return cmd_embed(args) if args.cmd == "embed" else cmd_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
