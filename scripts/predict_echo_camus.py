#!/usr/bin/env python3
"""MedSAM2 (``medsam2-echo-lv``) no CAMUS: traçado do VE + Simpson → VDF, VSF e FE.

Duas etapas:

``fetch``
    Baixa da API Girder pública do CAMUS (sem autenticação) só o necessário:
    ``Info_2CH.cfg``, ``Info_4CH.cfg`` e, por corte, ``_ED``, ``_ED_gt``, ``_ES``,
    ``_ES_gt`` (NIfTI), mais os arquivos da partição oficial e os termos de
    licença. Grava ``DATASET.json`` com tamanho e sha256 de cada arquivo.
    Termos: CC BY-NC-SA 4.0, só pesquisa não comercial; citação obrigatória
    (Leclerc et al., IEEE TMI 2019, doi:10.1109/TMI.2019.2900516).

``run``
    Para cada paciente das partições pedidas (padrão: validação + teste):

    * ``oracle_box`` — caixa = caixa da máscara de REFERÊNCIA do VE + 5 px em
      cada quadro ED/ES. É um ORÁCULO (substituto de uma caixa desenhada por um
      usuário competente), não um modo sem intervenção;
    * ``prompt_free`` — caixa por atlas: mediana das caixas de referência dos
      pacientes da partição de TREINO, normalizadas ao retângulo do setor, por
      corte e instante; depois um refinamento (caixa da 1ª máscara + 5 px).
      Usa o corte e os quadros ED/ES do CAMUS (metadados), que um exame real não
      traz rotulados.

    As medidas usam o MESMO código do produto
    (:func:`radiologyai.analysis.adapters.echo_medsam2.lv_measurements`, Simpson
    biplanar) nas duas convenções de eixo longo:

    * ``shared_max`` — a do código de referência do CAMUS, que gerou a FE do
      ``Info_*.cfg`` (o mesmo Simpson sobre as máscaras de referência reproduz
      essa FE: confere a implementação);
    * ``per_instant`` — a das diretrizes, usada pelo produto.

Saídas em ``<eval-preds>/medsam2-echo-lv/camus/``, uma linha por (paciente, modo):

* ``ef.parquet`` — ``y_true`` = FE do ``Info_*.cfg`` (referência do dataset),
  ``y_pred`` = FE prevista na convenção ``shared_max`` (comparável a ela);
* ``ef_per_instant.parquet`` — ``y_true`` = FE das máscaras de REFERÊNCIA
  (especialista) e ``y_pred`` = FE das máscaras do modelo, ambas na convenção
  ``per_instant`` do produto (isola o efeito da segmentação);

com ``patient_id, image_id, split, mode, image_quality`` (pior dos 2 cortes),
``image_quality_2ch/4ch, sex, age, ef_info, edv/esv`` previstos e de referência,
``dice_*`` e ``abstention`` (NaN em ``y_pred`` = abstenção). E
``summary.json`` com contagens e tempos (sem métricas de desempenho: quem mede é
a fábrica de evidência, ``radiologyai.evidence.measure.measure_agreement``).

Uso (inferência pela trava de trabalho pesado)::

    python scripts/predict_echo_camus.py fetch --out /root/medai-cache/datasets/camus-full
    /root/medai-cache/bin/heavy /root/medai-cache/bin/timev .venv/bin/python -I \\
        scripts/predict_echo_camus.py run --data /root/medai-cache/datasets/camus-full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
NIFTI_FOLDER = "63fde55f73e9f004868fb7ac"  # CAMUS_public/database_nifti
SPLIT_FOLDER = "66e27d12961576b1bad4e4e1"  # CAMUS_public/database_split
EVAL_PREDS = Path(os.environ.get("MEDAI_EVAL_PREDS", "/root/medai-cache/eval-preds"))
WEIGHTS = Path(
    os.environ.get(
        "MEDAI_MEDSAM2_CKPT",
        "/root/medai-cache/weights/registry/medsam2-echo-lv/MedSAM2_US_Heart.pt",
    )
)
VIEWS = {"4CH": "a4c", "2CH": "a2c"}
INSTANTS = ("ED", "ES")
MARGIN = 5.0
QUALITY_ORDER = {"Good": 0, "Medium": 1, "Poor": 2}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _wanted(name: str) -> bool:
    if name.startswith("Info_") or name == "MANDATORY_CITATION.md":
        return True
    return any(
        name.endswith(f"_{v}_{i}{s}.nii.gz") for v in VIEWS for i in INSTANTS for s in ("", "_gt")
    )


def cmd_fetch(args: argparse.Namespace) -> int:
    import httpx

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=120, follow_redirects=True) as c:

        def get(item: dict[str, Any], dest: Path) -> None:
            if not (dest.is_file() and dest.stat().st_size == int(item["size"])):
                r = c.get(f"{API}/item/{item['_id']}/download")
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
            files[str(dest.relative_to(out))] = {
                "item_id": item["_id"],
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
            }

        for item in c.get(f"{API}/item", params={"folderId": SPLIT_FOLDER, "limit": 50}).json():
            get(item, out / item["name"])
        top = c.get(f"{API}/item", params={"folderId": NIFTI_FOLDER, "limit": 50}).json()
        for item in top:
            get(item, out / item["name"])
        folders = c.get(
            f"{API}/folder",
            params={
                "parentType": "folder",
                "parentId": NIFTI_FOLDER,
                "limit": 1000,
                "sort": "name",
            },
        ).json()
        if args.limit:
            folders = folders[: args.limit]
        for k, folder in enumerate(folders):
            d = out / folder["name"]
            d.mkdir(exist_ok=True)
            items = c.get(
                f"{API}/item", params={"folderId": folder["_id"], "limit": 100, "sort": "name"}
            ).json()
            for item in items:
                if _wanted(item["name"]):
                    get(item, d / item["name"])
            if k % 50 == 0:
                print(f"[fetch] {k + 1}/{len(folders)} {folder['name']}", flush=True)
    doc = {
        "name": "CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)",
        "source": f"{API} (coleção CAMUS_public; pasta database_nifti {NIFTI_FOLDER})",
        "licence": "CC BY-NC-SA 4.0 (LICENSE_TERMS.md); somente pesquisa científica não comercial",
        "citation": "Leclerc et al., IEEE TMI 38(9):2198-2210, 2019, doi:10.1109/TMI.2019.2900516",
        "subset": "Info_*.cfg + ED/ES (imagem e máscara de referência), sem half_sequence",
        "n_files": len(files),
        "total_bytes": sum(v["bytes"] for v in files.values()),
        "files": files,
    }
    (out / "DATASET.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"[fetch] {len(files)} arquivos, {doc['total_bytes'] / 1e6:.1f} MB em {out}")
    return 0


def _info(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return dict(line.split(": ", 1) for line in lines if ": " in line)


def _split(data: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("training", "validation", "testing"):
        p = data / f"subgroup_{name}.txt"
        if p.is_file():
            for pid in p.read_text(encoding="utf-8").split():
                out[pid.strip()] = name
    return out


def _load2d(path: Path) -> Any:
    import nibabel as nib
    import numpy as np

    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    zooms = img.header.get_zooms()
    return np.ascontiguousarray(arr.T), (float(zooms[1]), float(zooms[0]))


def _dice(a: Any, b: Any) -> float:
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else 1.0


def cmd_run(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd
    import torch

    from radiologyai.analysis.adapters.echo_medsam2 import (
        MedSAM2Segmenter,
        atlas_box,
        lv_measurements,
    )
    from radiologyai.analysis.adapters.us_common import mask_bbox, pad_box

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    data: Path = args.data
    split = _split(data)
    patients = sorted(p.name for p in data.glob("patient*") if p.is_dir())
    if not split:
        print("ERRO: arquivos subgroup_*.txt ausentes; rode fetch", file=sys.stderr)
        return 2

    # Atlas (sem prompt): caixas de referência dos pacientes de TREINO, normalizadas ao setor.
    boxes: dict[tuple[str, str], list[list[float]]] = {}
    for pid in patients:
        if split.get(pid) != "training":
            continue
        for v in VIEWS:
            for inst in INSTANTS:
                img, _ = _load2d(data / pid / f"{pid}_{v}_{inst}.nii.gz")
                gt, _ = _load2d(data / pid / f"{pid}_{v}_{inst}_gt.nii.gz")
                lv = gt == 1
                if not lv.any():
                    continue
                s = mask_bbox(img > 0)
                b = mask_bbox(lv)
                w, h = s[2] - s[0], s[3] - s[1]
                boxes.setdefault((v, inst), []).append(
                    [(b[0] - s[0]) / w, (b[1] - s[1]) / h, (b[2] - s[0]) / w, (b[3] - s[1]) / h]
                )
    atlas = {k: tuple(float(x) for x in np.median(np.asarray(v), axis=0)) for k, v in boxes.items()}
    n_atlas = {f"{k[0]}_{k[1]}": len(v) for k, v in boxes.items()}
    print(f"[atlas] pacientes de treino por corte/instante: {n_atlas}", flush=True)

    t0 = time.time()
    seg = MedSAM2Segmenter(WEIGHTS, device="cpu")
    t_load = time.time() - t0
    wanted = set(args.splits.split(","))
    todo = [p for p in patients if split.get(p) in wanted]
    if args.limit:
        todo = todo[: args.limit]
    rows: list[dict[str, Any]] = []
    times: list[float] = []
    for n, pid in enumerate(todo):
        info = {v: _info(data / pid / f"Info_{v}.cfg") for v in VIEWS}
        frames: dict[tuple[str, str], tuple[Any, Any, tuple[float, float]]] = {}
        for v in VIEWS:
            for inst in INSTANTS:
                img, sp = _load2d(data / pid / f"{pid}_{v}_{inst}.nii.gz")
                gt, _ = _load2d(data / pid / f"{pid}_{v}_{inst}_gt.nii.gz")
                u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
                frames[(v, inst)] = (u8, gt == 1, sp)
        ref_masks = {(VIEWS[v], i.lower()): (m, sp, 0) for (v, i), (_, m, sp) in frames.items()}
        ref = {m.key: m.value for m in lv_measurements(ref_masks, length_convention="shared_max")}
        ref_pi = {m.key: m.value for m in lv_measurements(ref_masks)}
        q = {v: info[v].get("ImageQuality", "") for v in VIEWS}
        worst = max(q.values(), key=lambda s: QUALITY_ORDER.get(s, 3))
        base = {
            "patient_id": pid,
            "image_id": pid,
            "split": split.get(pid, ""),
            "image_quality": worst,
            "image_quality_2ch": q["2CH"],
            "image_quality_4ch": q["4CH"],
            "sex": info["2CH"].get("Sex"),
            "age": float(info["2CH"]["Age"]) if info["2CH"].get("Age") else float("nan"),
            "y_true": float(info["2CH"]["EF"]),
            "ef_info": float(info["2CH"]["EF"]),
            "ef_ref_masks_simpson": ref.get("lv_ef"),
            "edv_ref_masks": ref.get("lv_edv"),
            "esv_ref_masks": ref.get("lv_esv"),
            "ef_ref_masks_per_instant": ref_pi.get("lv_ef"),
            "edv_ref_masks_per_instant": ref_pi.get("lv_edv"),
            "esv_ref_masks_per_instant": ref_pi.get("lv_esv"),
        }
        for mode in ("oracle_box", "prompt_free"):
            traced = {}
            dices: dict[str, float] = {}
            for (v, inst), (u8, gt, sp) in frames.items():
                t = time.time()
                if mode == "oracle_box":
                    box = pad_box(mask_bbox(gt), u8.shape, MARGIN)
                    mask, _ = seg.segment(u8, box)
                else:
                    mask, _ = seg.segment(u8, atlas_box(u8, atlas[(v, inst)]))
                    if mask.any():
                        mask, _ = seg.segment(u8, pad_box(mask_bbox(mask), u8.shape, MARGIN))
                times.append(time.time() - t)
                traced[(VIEWS[v], inst.lower())] = (mask, sp, 0)
                dices[f"dice_{VIEWS[v]}_{inst.lower()}"] = _dice(mask, gt)
            meas = {m.key: m for m in lv_measurements(traced, length_convention="shared_max")}
            pi = {m.key: m for m in lv_measurements(traced)}
            ef = meas["lv_ef"]
            rows.append(
                {
                    **base,
                    "mode": mode,
                    "y_pred": float("nan") if ef.value is None else float(ef.value),
                    "edv_pred": meas["lv_edv"].value,
                    "esv_pred": meas["lv_esv"].value,
                    "y_pred_per_instant": (
                        float("nan") if pi["lv_ef"].value is None else float(pi["lv_ef"].value)
                    ),
                    "edv_pred_per_instant": pi["lv_edv"].value,
                    "esv_pred_per_instant": pi["lv_esv"].value,
                    "abstention": None if ef.abstention is None else ef.abstention.value,
                    "abstention_note": ef.note,
                    **dices,
                }
            )
        print(
            f"[camus] {n + 1}/{len(todo)} {pid} ref={base['y_true']:.0f} "
            f"refmask={base['ef_ref_masks_simpson']} "
            + " ".join(f"{r['mode']}={r['y_pred']:.1f}" for r in rows[-2:]),
            flush=True,
        )
    out_dir = EVAL_PREDS / "medsam2-echo-lv" / "camus"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "ef.parquet", index=False)
    per_instant = df.assign(y_true=df["ef_ref_masks_per_instant"], y_pred=df["y_pred_per_instant"])
    per_instant.to_parquet(out_dir / "ef_per_instant.parquet", index=False)
    summary = {
        "model_id": "medsam2-echo-lv",
        "weights_sha256": _sha256(WEIGHTS),
        "dataset": "camus",
        "splits": sorted(wanted),
        "n_patients": len(todo),
        "modes": {
            "oracle_box": "caixa da máscara de referência + 5 px (oráculo)",
            "prompt_free": "atlas da partição de treino + 1 refinamento; corte e ED/ES do CAMUS",
        },
        "atlas_n_training_patients": n_atlas,
        "atlas_normalized_boxes": {f"{k[0]}_{k[1]}": list(v) for k, v in atlas.items()},
        "n_abstained": {
            m: int(df[(df["mode"] == m)]["y_pred"].isna().sum())
            for m in ("oracle_box", "prompt_free")
        },
        "timing": {
            "model_load_s": round(t_load, 2),
            "segment_s_mean": round(float(np.mean(times)), 3) if times else None,
            "n_segment_calls": len(times),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--splits", default="validation,testing")
    r.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    return cmd_fetch(args) if args.cmd == "fetch" else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
