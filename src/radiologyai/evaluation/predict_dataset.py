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
    """Lê uma imagem PNG/JPEG em escala de cinza, normalizada para [0, 1]."""
    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        raise EvaluationError(
            "Pillow é necessário para ler imagens não-DICOM. "
            "Instale com: pip install 'radiologyai[imaging]'"
        )
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img.convert("F"), dtype=np.float32)

    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


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
