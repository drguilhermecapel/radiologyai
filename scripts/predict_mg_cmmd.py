#!/usr/bin/env python3
"""CMMD: download da subamostra determinística e predição do mammoscreen por mama.

Subcomandos
-----------
``download``
    Baixa as séries listadas (formato de ``cmmd_sample300.txt``: ``ID rótulo
    n_imagens bytes SeriesInstanceUID`` por linha) pelo endpoint público
    ``getImage`` do NBIA/TCIA (CMMD, CC BY 4.0, DOI 10.7937/tcia.eqde-4b16) para
    ``<dest>/<ID>/``. Retomável: paciente com o número esperado de arquivos é
    pulado; cada série é extraída num diretório temporário e renomeada no fim.
    Membros do zip são conferidos (nome simples, só ``*.dcm`` e ``LICENSE``).

``predict``
    Roda ``mammo-crop`` + ``mammoscreen`` (adaptadores reais, pesos conferidos
    pelo registro) em cada mama ROTULADA da subamostra e grava
    ``patient_id, breast_id, laterality, views, y_true, y_score`` em parquet.
    ``y_true`` = 1 se a biópsia daquela mama é maligna, 0 se benigna (CMMD é
    inteiramente biopsiado: a medida é discriminação benigno x maligno, não
    rastreamento). A mama contralateral sem rótulo NÃO entra (no subconjunto D2
    ela não é biópsia benigna). Retomável: um JSONL de progresso guarda cada
    mama concluída; o parquet é reescrito a partir dele no fim.

Rode sob a trava de trabalho pesado::

    /root/medai-cache/bin/heavy /root/medai-cache/bin/timev \\
        .venv/bin/python scripts/predict_mg_cmmd.py predict
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

NBIA_GET_IMAGE = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
DEFAULT_LIST = Path("/root/medai-cache/spikes/mammo/meta/cmmd_sample300.txt")
DEFAULT_DEST = Path("/root/medai-cache/datasets/cmmd-sample300")
DEFAULT_CLINICAL = Path("/root/medai-cache/datasets/cmmd-sample/CMMD_clinicaldata_revision.csv")
DEFAULT_OUT = Path("/root/medai-cache/eval-preds/mammoscreen/cmmd/malignancy.parquet")
DEFAULT_WEIGHTS = Path("/root/medai-cache/weights/registry")


def read_list(path: Path) -> list[dict[str, Any]]:
    """Linhas ``ID rótulo n_imagens bytes SeriesUID`` (comentários com ``#`` ignorados)."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) != 5:
            raise ValueError(f"linha inválida em {path}: {line!r}")
        rows.append(
            {
                "patient": parts[0],
                "label": parts[1],
                "n_images": int(parts[2]),
                "bytes": int(parts[3]),
                "series_uid": parts[4],
            }
        )
    return rows


def _complete(dest: Path, row: dict[str, Any]) -> bool:
    folder = dest / row["patient"]
    return folder.is_dir() and len(list(folder.glob("*.dcm"))) == int(row["n_images"])


def _fetch(row: dict[str, Any], dest: Path, retries: int = 4) -> str:
    import httpx

    if _complete(dest, row):
        return "já presente"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=httpx.Timeout(300.0), follow_redirects=True) as client:
                resp = client.get(NBIA_GET_IMAGE, params={"SeriesInstanceUID": row["series_uid"]})
                resp.raise_for_status()
                payload = resp.content
            break
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(5.0 * (attempt + 1))
    else:
        raise RuntimeError(f"{row['patient']}: download falhou: {last}")
    tmp = dest / f".tmp-{row['patient']}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for info in zf.infolist():
            name = info.filename
            if "/" in name or "\\" in name or name.startswith(".") or info.is_dir():
                raise RuntimeError(f"{row['patient']}: membro suspeito no zip: {name!r}")
            if not (name.endswith(".dcm") or name == "LICENSE"):
                raise RuntimeError(f"{row['patient']}: membro inesperado no zip: {name!r}")
            (tmp / name).write_bytes(zf.read(info))
    got = len(list(tmp.glob("*.dcm")))
    if got != row["n_images"]:
        shutil.rmtree(tmp)
        raise RuntimeError(f"{row['patient']}: {got} DICOM, esperados {row['n_images']}")
    final = dest / row["patient"]
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return f"{got} imagens"


def cmd_download(args: argparse.Namespace) -> int:
    rows = read_list(args.list)
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    clinical = dest / "CMMD_clinicaldata_revision.csv"
    if not clinical.exists() and args.clinical.exists():
        shutil.copy2(args.clinical, clinical)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, row, dest): row for row in rows}
        for i, (future, row) in enumerate(futures.items(), start=1):
            try:
                status = future.result()
            except Exception as exc:  # noqa: BLE001 - relatado; o download é retomável
                failures += 1
                status = f"FALHA: {exc}"
            print(f"[{i}/{len(rows)}] {row['patient']}: {status}", flush=True)
    print(f"concluído; falhas: {failures}", flush=True)
    return 1 if failures else 0


def cmd_predict(args: argparse.Namespace) -> int:
    import pandas as pd

    from radiologyai.analysis.base import ModelInputs
    from radiologyai.analysis.registry import Registry
    from radiologyai.catalog.schema import ExamType
    from radiologyai.modalities.mg.loader import load_mammogram

    rows = read_list(args.list)
    clin = pd.read_csv(args.clinical)
    labels: dict[tuple[str, str], int] = {}
    for rec in clin.itertuples():
        cls = str(rec.classification).strip().lower()
        if cls not in ("benign", "malignant"):
            raise ValueError(f"classificação desconhecida {rec.classification!r} para {rec.ID1}")
        key = (str(rec.ID1), str(rec.LeftRight).strip().upper())
        y = int(cls == "malignant")
        if key in labels and labels[key] != y:
            raise ValueError(f"rótulos conflitantes para {key}")
        labels[key] = y

    registry = Registry(weights_dir=args.weights_dir)
    crop = registry.load("mammo-crop")
    screen = registry.load("mammoscreen")
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = out.with_suffix(".progress.jsonl")
    done: dict[str, dict[str, Any]] = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["breast_id"]] = rec
    limit = args.limit or len(rows)
    t_start = time.monotonic()
    n_new = 0
    with progress.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows[:limit], start=1):
            pid = row["patient"]
            folder = args.dest / pid
            if not _complete(args.dest, row):
                print(f"[{i}] {pid}: dados ausentes; pulado", flush=True)
                continue
            lateralities = [lat for (p, lat) in labels if p == pid]
            if all(f"{pid}-{lat}" in done for lat in lateralities):
                continue
            images = tuple(load_mammogram(f) for f in sorted(folder.glob("*.dcm")))
            for lat in sorted(lateralities):
                breast_id = f"{pid}-{lat}"
                if breast_id in done:
                    continue
                sel = tuple(im for im in images if im.meta.laterality == lat)
                if not sel:
                    raise RuntimeError(f"{breast_id}: nenhuma imagem com essa lateralidade")
                roles = _roles(sel)
                inputs = ModelInputs(images=sel, exam_type=ExamType.MG, roles=roles)
                qa = crop.applicability(inputs)
                if not qa.applicable:
                    raise RuntimeError(f"{breast_id}: mammo-crop recusou: {qa.reasons}")
                t0 = time.monotonic()
                crop_out = crop.predict(inputs)
                inputs = ModelInputs(
                    images=sel,
                    exam_type=ExamType.MG,
                    roles=roles,
                    upstream={"mammo-crop": crop_out},
                )
                qa = screen.applicability(inputs)
                if not qa.applicable:
                    raise RuntimeError(f"{breast_id}: mammoscreen recusou: {qa.reasons}")
                pred = screen.predict(inputs)
                seconds = time.monotonic() - t0
                label = "malignancy_left" if lat == "L" else "malignancy_right"
                score = [s.value for s in pred.scores if s.label == label]
                if len(score) != 1:
                    raise RuntimeError(f"{breast_id}: escore {label} ausente")
                rec = {
                    "patient_id": pid,
                    "breast_id": breast_id,
                    "laterality": lat,
                    "views": ",".join(sorted(roles)),
                    "y_true": labels[(pid, lat)],
                    "y_score": float(score[0]),
                    "seconds": round(seconds, 3),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[breast_id] = rec
                n_new += 1
                print(
                    f"[{i}/{limit}] {breast_id} y={rec['y_true']} score={rec['y_score']:.5f} "
                    f"views={rec['views']} {seconds:.1f}s",
                    flush=True,
                )
    frame = pd.DataFrame(sorted(done.values(), key=lambda r: r["breast_id"]))
    frame.to_parquet(out, index=False)
    meta = {
        "model_ids": ["mammo-crop", "mammoscreen"],
        "weights_sha256": {
            "mammo-crop": dict(registry.manifest("mammo-crop").weights[0].model_dump())["sha256"],
            "mammoscreen": dict(registry.manifest("mammoscreen").weights[0].model_dump())["sha256"],
        },
        "dataset": "CMMD (TCIA, CC BY 4.0, doi:10.7937/tcia.eqde-4b16)",
        "subsample_list": str(args.list),
        "unit": "mama rotulada por biópsia (benigno=0, maligno=1)",
        "n_breasts": len(frame),
        "external": "mammoscreen treinado em CBIS-DDSM + RSNA; CMMD não está no treino",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    elapsed = time.monotonic() - t_start
    print(f"{n_new} mamas novas em {elapsed:.0f}s; total {len(frame)} -> {out}", flush=True)
    return 0


def _roles(images: tuple[Any, ...]) -> dict[str, int]:
    from radiologyai.analysis.routing import role_of

    roles: dict[str, int] = {}
    for idx, image in enumerate(images):
        role = role_of(image)
        if role is not None and role not in roles:
            roles[role] = idx
    return roles


def cmd_evidence(args: argparse.Namespace) -> int:
    """Grava ``evidence.parquet`` (formato da fábrica de evidência): uma linha por mama."""
    import pandas as pd

    df = pd.read_parquet(args.out)
    out = pd.DataFrame(
        {
            "image_id": df["breast_id"].astype(str),  # unidade de análise = mama
            "patient_id": df["patient_id"].astype(str),  # bootstrap agrupado por paciente
            "sex": "F",
            "y_true": df["y_true"].astype(int),
            "y_score": df["y_score"].astype(float),
        }
    )
    dst = args.out.with_name("evidence.parquet")
    out.to_parquet(dst, index=False)
    print(f"{len(out)} mamas de {out['patient_id'].nunique()} pacientes -> {dst}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="baixa a subamostra do NBIA (retomável)")
    dl.add_argument("--list", type=Path, default=DEFAULT_LIST)
    dl.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    dl.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    dl.add_argument("--workers", type=int, default=4)
    pr = sub.add_parser("predict", help="roda mammo-crop + mammoscreen por mama (retomável)")
    pr.add_argument("--list", type=Path, default=DEFAULT_LIST)
    pr.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    pr.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    pr.add_argument("--out", type=Path, default=DEFAULT_OUT)
    pr.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    pr.add_argument("--limit", type=int, default=0, help="só os primeiros N pacientes da lista")
    ev = sub.add_parser("evidence", help="converte o parquet de predições para a fábrica")
    ev.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.cmd == "evidence":
        return cmd_evidence(args)
    if args.cmd == "download":
        return cmd_download(args)
    return cmd_predict(args)


if __name__ == "__main__":
    sys.exit(main())
