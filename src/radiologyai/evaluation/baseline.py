"""Orquestrador da linha de base: dataset em disco → artefato de avaliação.

Existe para que o notebook do Colab seja **fino**: uma chamada de função no
kernel, em vez de células ``!python -m ...``. A diferença importa por dois
motivos que já custaram uma execução inteira:

1. Um subprocesso ``!python`` não herda o ``sys.path`` do kernel. Se o pacote
   não estiver instalado (o ``pip install -e .`` falha em silêncio quando a
   versão do Python não bate), o comando quebra com ``No module named`` e o
   notebook **continua** — até morrer numa célula posterior com um erro que
   não tem nada a ver com a causa.
2. Uma função no kernel levanta exceção. "Executar tudo" para na célula certa,
   com o traceback certo.

Toda a lógica aqui é testada localmente sobre um dataset sintético em formato
NIH, sem depender do Colab.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radiologyai.data.manifest import Manifest
from radiologyai.data.nih_cxr14 import (
    LIMITATIONS,
    build_manifest,
    discover_image_dirs,
    find_image,
)
from radiologyai.errors import EvaluationError
from radiologyai.evaluation.predict_dataset import predict_manifest
from radiologyai.evaluation.runner import EvaluationConfig, check_leakage, run_evaluation
from radiologyai.models import get_card
from radiologyai.models.backends import build_backend

DEFAULT_CARD = "xrv-densenet121-pc"
DEFAULT_SEED = 20260101


@dataclass(frozen=True)
class BaselineResult:
    """Saída de :func:`run_baseline`."""

    run_dir: Path
    manifest_path: Path
    n_images: int
    n_patients: int
    leakage_status: str
    macro_auroc: float | None

    @property
    def metrics(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(
            (self.run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        return data


def prepare_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "test",
    limit: int = 0,
) -> Manifest:
    """Constrói e grava o manifest do split pedido."""
    manifest = build_manifest(data_root, split=split)
    if limit:
        manifest.rows = manifest.rows[:limit]
    manifest.to_csv(manifest_path)
    return manifest


def run_baseline(
    data_root: str | Path,
    artifacts_dir: str | Path,
    *,
    card_id: str = DEFAULT_CARD,
    manifest_path: str | Path | None = None,
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
    n_bootstrap: int = 2000,
    batch_size: int = 32,
    limit: int = 0,
    log: Callable[[str], Any] = print,
) -> BaselineResult:
    """Executa a linha de base completa e devolve o diretório do artefato.

    Args:
        data_root: diretório com o CSV oficial, ``test_list.txt`` e as imagens.
        artifacts_dir: raiz de ``artifacts/eval``.
        limit: limita o número de imagens. **Só para teste** — o resultado é
            marcado como parcial e não é a medição completa.
        log: destino das mensagens de progresso.

    Raises:
        EvaluationError: dataset incompleto, sem imagens, ou manifest vazio.
        ModelNotFoundError, WeightsIntegrityError, BackendUnavailableError:
            propagadas do carregamento do modelo — nunca absorvidas.
    """
    data_root = Path(data_root)
    artifacts_dir = Path(artifacts_dir)
    manifest_path = (
        Path(manifest_path) if manifest_path else artifacts_dir / "manifests" / "nih_cxr14_test.csv"
    )

    log(f"dataset: {data_root}")
    manifest = prepare_manifest(data_root, manifest_path, limit=limit)
    log(
        f"manifest: {len(manifest)} imagens · {len(manifest.patient_ids)} pacientes "
        f"· split {manifest.split} · sha256 {manifest.sha256()[:16]}…"
    )
    if limit:
        log(f"AVISO: limitado a {limit} imagens — resultado NÃO é a medição completa.")

    image_dirs = discover_image_dirs(data_root)
    if not image_dirs:
        raise EvaluationError(
            f"nenhum diretório de imagens em {data_root}. Esperado images_0xx/images/ "
            "(distribuição oficial e Kaggle) ou images/."
        )
    log(f"imagens: {len(image_dirs)} diretório(s)")

    # Verifica que a primeira imagem do manifest existe ANTES de carregar o
    # modelo — é o erro mais comum e o mais barato de detectar cedo.
    find_image(manifest.rows[0].image_id, image_dirs)

    card = get_card(card_id)
    leakage = check_leakage(card, manifest)
    if leakage == "in-distribution":
        log(
            "AVISO: este dataset consta entre os dados de treino do modelo. "
            "O resultado será in-distribution, NÃO validação externa."
        )
    else:
        log(f"vazamento: {leakage}")

    log(f"carregando {card.card_id} (verificando sha256 dos pesos)…")
    backend = build_backend(card)
    if device != "cpu" and hasattr(backend, "to"):
        backend.to(device)
    log(f"backend pronto em {device}")

    total = len(manifest)
    last_reported = {"pct": -1}

    def progress(done: int, _total: int) -> None:
        pct = int(100 * done / total)
        if pct // 10 > last_reported["pct"] // 10 or done == total:
            last_reported["pct"] = pct
            log(f"  inferência {done}/{total} ({pct}%)")

    scores = predict_manifest(
        backend=backend,
        manifest=manifest,
        resolve_path=lambda image_id: find_image(image_id, image_dirs),
        batch_size=batch_size,
        progress=progress,
    )

    run_dir = run_evaluation(
        card=card,
        backend=backend,
        manifest=manifest,
        y_score=scores,
        output_dir=artifacts_dir,
        config=EvaluationConfig(seed=seed, n_bootstrap=n_bootstrap),
        dataset_limitations=LIMITATIONS,
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    log(f"artefato: {run_dir}")

    return BaselineResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        n_images=len(manifest),
        n_patients=len(manifest.patient_ids),
        leakage_status=leakage,
        macro_auroc=metrics["macro_auroc"],
    )


def summarize(result: BaselineResult, log: Callable[[str], Any] = print) -> None:
    """Imprime o resumo do artefato, na mesma forma usada pelo notebook."""
    m = result.metrics
    log(f"AUROC macro: {m['macro_auroc']}")
    log(f"rótulos avaliados: {m['n_labels_evaluated']}")
    log(
        f"externo ao treino: {m['dataset']['external_to_training_data']} "
        f"({m['dataset']['leakage_status']})"
    )
    log(f"imagens/pacientes: {m['dataset']['n_images']} / {m['dataset']['n_patients']}")
    log("")
    for name, e in sorted(m["per_label"].items(), key=lambda kv: -kv[1]["auroc"]):
        lo, hi = e["auroc_ci95"]
        log(f"  {name:<26} {e['auroc']:.3f}  IC95 [{lo:.3f}, {hi:.3f}]  n+={e['n_pos']:>5}")
    if m["not_evaluated"]:
        log("")
        log("não avaliados:")
        for x in m["not_evaluated"]:
            log(f"  {x['label']:<28} {x['reason']}")
