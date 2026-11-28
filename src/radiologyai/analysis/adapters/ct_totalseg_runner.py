"""Processo filho que roda o TotalSegmentator — NUNCA importado pelo processo principal.

Executado por :class:`~radiologyai.analysis.adapters.ct_totalseg.TotalSegmentatorAdapter`
como ``python -m radiologyai.analysis.adapters.ct_totalseg_runner``. Isolar em
subprocesso: (1) o nnU-Net cria pools ``spawn`` e mantém estado global do torch;
(2) a memória do modelo volta ao sistema ao fim de cada exame; (3) o ambiente
de rede do filho é morto (proxy em porta fechada), de modo que qualquer
tentativa de download ou de estatística de uso FALHA em vez de sair do host.

Antes de importar o TotalSegmentator:

* ``TOTALSEG_HOME_DIR/config.json`` é escrito/forçado com ``send_usage_stats: false``
  (sem ele o TotalSegmentator cria o arquivo com ``true`` e envia telemetria);
* todos os diretórios de pesos esperados devem existir em ``TOTALSEG_WEIGHTS_PATH``
  (senão o TotalSegmentator tentaria baixá-los).

Saída: NIfTI multirrótulo na grade da entrada + JSON lateral com o mapa de
classes, versões e tempo. Código de saída != 0 em qualquer falha.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

USAGE_STATS_OFF = {
    "totalseg_id": "totalseg_OFFLINE00",
    "send_usage_stats": False,
    "prediction_counter": 0,
    "statistics_disclaimer_shown": True,
}


def force_no_usage_stats(home: Path) -> None:
    """Garante ``send_usage_stats: false`` no config.json do TotalSegmentator."""
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "config.json"
    data = dict(USAGE_STATS_OFF)
    if cfg.is_file():
        data = {**json.loads(cfg.read_text(encoding="utf-8")), "send_usage_stats": False}
    cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if json.loads(cfg.read_text(encoding="utf-8")).get("send_usage_stats") is not False:
        raise SystemExit("telemetria do TotalSegmentator não pôde ser desligada")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=("default", "fast", "fastest"), default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expect", action="append", default=[], help="pasta DatasetXXX exigida")
    args = parser.parse_args(argv)

    home = Path(os.environ["TOTALSEG_HOME_DIR"])
    weights = Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
    force_no_usage_stats(home)
    missing = [name for name in args.expect if not (weights / name).is_dir()]
    if missing:
        raise SystemExit(f"pesos ausentes em {weights}: {missing}; download é proibido")

    import importlib.metadata

    import nibabel as nib
    import torch
    from totalsegmentator.map_to_binary import class_map, commercial_models
    from totalsegmentator.python_api import totalsegmentator

    if args.task in commercial_models:
        raise SystemExit(f"tarefa {args.task!r} exige licença comercial; não suportada")
    threads = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    started = time.monotonic()
    seg = totalsegmentator(
        str(args.input),
        str(args.output),
        ml=True,
        task=args.task,
        fast=args.mode == "fast",
        fastest=args.mode == "fastest",
        device=args.device,
        quiet=True,
        nr_thr_saving=1,
    )
    seconds = time.monotonic() - started
    img: Any = nib.load(str(args.input))
    sidecar = {
        "task": args.task,
        "mode": args.mode,
        "class_map": {str(k): v for k, v in class_map[args.task].items()},
        "totalsegmentator_version": importlib.metadata.version("TotalSegmentator"),
        "nnunetv2_version": importlib.metadata.version("nnunetv2"),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": args.device,
        "seconds": round(seconds, 2),
        "output_shape": list(seg.shape),
        "input_shape": list(img.shape),
    }
    args.sidecar.write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
