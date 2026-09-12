"""Inferência em lote sobre um manifest — o passo que antecede a avaliação."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.errors import EvaluationError

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt

    from radiologyai.data.manifest import Manifest
    from radiologyai.models.backends.base import InferenceBackend


def load_png(path: Path) -> npt.NDArray[np.float32]:
    """Lê uma imagem PNG/JPEG em escala de cinza como fração do fundo de escala.

    Normaliza pelo **máximo do tipo de pixel** (255 para 8 bits, 65535 para
    16 bits), não pelo mínimo/máximo da imagem. É o pipeline canônico do
    torchxrayvision (``imread`` → ``normalize(img, 255)``): o modelo foi
    treinado sem esticar o contraste por imagem, e esticar aqui introduziria
    uma diferença de pré-processamento entre treino e avaliação.

    A primeira linha de base (run ``xrv-densenet121-pc__20260912T202549Z``,
    git ``9d240cc``) foi produzida com min-max por imagem; o artefato registra
    o SHA e permanece reproduzível naquele commit.
    """
    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        raise EvaluationError(
            "Pillow é necessário para ler imagens não-DICOM. "
            "Instale com: pip install 'radiologyai[imaging]'"
        )
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        mode = img.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(img, dtype=np.float32)
            full_scale = 65535.0
        else:
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            full_scale = 255.0

    return np.clip(arr / np.float32(full_scale), 0.0, 1.0).astype(np.float32)


def predict_manifest(
    *,
    backend: InferenceBackend,
    manifest: Manifest,
    resolve_path: Callable[[str], Path],
    batch_size: int = 32,
    progress: Callable[[int, int], Any] | None = None,
    loader: Callable[[Path], npt.NDArray[np.float32]] | None = None,
) -> npt.NDArray[np.float32]:
    """Executa o backend sobre todas as imagens do manifest.

    Args:
        resolve_path: mapeia ``image_id`` para caminho em disco.
        progress: chamado como ``progress(feitas, total)``.

    Returns:
        Matriz ``n_imagens x n_rótulos_do_modelo``, na ordem do manifest.
    """
    import numpy as np

    read = loader or load_png
    total = len(manifest)
    if total == 0:
        raise EvaluationError("manifest vazio")

    scores: list[npt.NDArray[np.float32]] = []
    batch: list[npt.NDArray[np.float32]] = []

    for i, row in enumerate(manifest, start=1):
        batch.append(read(resolve_path(row.image_id)))
        if len(batch) >= batch_size or i == total:
            scores.append(backend.predict_batch(batch))
            batch = []
            if progress:
                progress(i, total)

    return np.concatenate(scores, axis=0).astype(np.float32)
