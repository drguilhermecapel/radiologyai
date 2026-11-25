"""Reconstrução de ``ImageData`` a partir do repositório e quadros para o visualizador.

O worker e o endpoint de quadros usam o MESMO caminho de decodificação da
ingestão (:func:`radiologyai.interop.ingest.decode_files`) sobre os arquivos
des-identificados guardados — o que o médico vê é o que o modelo recebeu.

Formato dos quadros (o cliente aplica janela/LUT):

* tons de cinza: inteiros ``uint16`` ou ``int16`` little-endian, linha a linha;
  valor físico = ``armazenado * rescale_slope + rescale_intercept`` (HU em TC);
* cor: ``uint8`` RGB intercalado;
* ``scale``: fator inteiro de redução (média por bloco; rótulos: vizinho mais
  próximo); coordenada no quadro = coordenada na imagem de origem / ``scale``.
"""

from __future__ import annotations

import math
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from radiologyai.errors import IngestionError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.analysis.types import ImageData
    from radiologyai.persistence.blobs import BlobStore
    from radiologyai.persistence.models import Series

__all__ = [
    "FrameData",
    "FrameRangeError",
    "ImageCache",
    "encode_frame",
    "encode_label_frame",
    "load_series_image",
    "plane",
    "scale_for",
]


class FrameRangeError(IngestionError):
    """Índice de quadro fora da série."""


def load_series_image(series: Series, blobs: BlobStore) -> ImageData:
    """Decodifica a série a partir dos arquivos guardados (conferidos por sha256)."""
    from radiologyai.interop.ingest import decode_files

    with tempfile.TemporaryDirectory(prefix="radiologyai-load-") as tmp:
        workdir = Path(tmp)
        files: list[tuple[str, Path]] = []
        for inst in series.instances:
            target = workdir / inst.filename
            # read() confere o sha256: arquivo corrompido em disco levanta.
            target.write_bytes(blobs.read(inst.artifact_sha256))
            files.append((inst.filename, target))
        image = decode_files(
            series.source_format,
            files,
            modality=series.modality,
            overrides=dict(series.meta.get("human_overrides") or {}),
            workdir=workdir,
        )
    if image.meta.content_sha256 != series.content_sha256 and series.source_format != "nifti":
        raise IngestionError(
            f"série {series.id}: conteúdo decodificado não confere com o ingerido "
            f"({image.meta.content_sha256} != {series.content_sha256})"
        )
    return image


class ImageCache:
    """LRU pequeno de séries decodificadas (volumes de TC custam segundos para decodificar)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, ImageData] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> ImageData | None:
        with self._lock:
            image = self._items.get(key)
            if image is not None:
                self._items.move_to_end(key)
            return image

    def put(self, key: str, image: ImageData) -> None:
        if self.capacity <= 0:
            return
        with self._lock:
            self._items[key] = image
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)


def plane(image: ImageData, index: int) -> npt.NDArray[np.float32]:
    """Plano 2D (ou 2D+cor) do quadro ``index`` (corte de volume, quadro de cine)."""
    if image.kind.value == "projection":
        if index != 0:
            raise FrameRangeError("imagem de projeção tem um único quadro (índice 0)")
        return image.pixels
    n = int(image.pixels.shape[0])
    if not 0 <= index < n:
        raise FrameRangeError(f"quadro {index} fora de 0..{n - 1}")
    out: npt.NDArray[np.float32] = image.pixels[index]
    return out


def scale_for(rows: int, cols: int, max_dim: int) -> int:
    """Fator inteiro de redução para que o maior lado caiba em ``max_dim``."""
    return max(1, math.ceil(max(rows, cols) / max(1, max_dim)))


@dataclass(frozen=True, slots=True)
class FrameData:
    data: bytes
    rows: int
    columns: int
    channels: int
    dtype: Literal["uint16", "int16", "uint8"]
    scale: int
    rescale_slope: float
    rescale_intercept: float


def _block_mean(arr: npt.NDArray[Any], f: int) -> npt.NDArray[np.float32]:
    import numpy as np

    if f == 1:
        return arr.astype(np.float32, copy=False)
    h, w = arr.shape[:2]
    ph, pw = (-h) % f, (-w) % f
    pad = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
    padded = np.pad(arr.astype(np.float32, copy=False), pad, mode="edge")
    shape = (padded.shape[0] // f, f, padded.shape[1] // f, f, *padded.shape[2:])
    out: npt.NDArray[np.float32] = padded.reshape(shape).mean(axis=(1, 3), dtype=np.float32)
    return out


def encode_frame(image: ImageData, index: int, max_dim: int) -> FrameData:
    """Quadro reduzido e quantizado sem perda de faixa (ver docstring do módulo)."""
    import numpy as np

    src = plane(image, index)
    f = scale_for(src.shape[0], src.shape[1], max_dim)
    small = _block_mean(src, f)
    rows, cols = int(small.shape[0]), int(small.shape[1])
    if image.color:
        data = np.clip(np.round(small), 0, 255).astype(np.uint8).tobytes()
        return FrameData(data, rows, cols, 3, "uint8", f, 1.0, 0.0)
    lo, hi = image.value_range
    finite = np.nan_to_num(small, nan=lo, posinf=hi, neginf=lo)
    if lo >= 0 and hi <= 65535:
        stored = np.clip(np.round(finite), 0, 65535).astype("<u2")
        return FrameData(stored.tobytes(), rows, cols, 1, "uint16", f, 1.0, 0.0)
    if lo >= -32768 and hi <= 32767:
        stored = np.clip(np.round(finite), -32768, 32767).astype("<i2")
        return FrameData(stored.tobytes(), rows, cols, 1, "int16", f, 1.0, 0.0)
    slope = (hi - lo) / 65535.0
    stored = np.clip(np.round((finite - lo) / slope), 0, 65535).astype("<u2")
    return FrameData(stored.tobytes(), rows, cols, 1, "uint16", f, float(slope), float(lo))


def encode_label_frame(labelmap: npt.NDArray[Any], index: int, max_dim: int) -> FrameData:
    """Quadro de um mapa de rótulos (vizinho mais próximo; mesma grade de ``encode_frame``)."""
    import numpy as np

    if labelmap.ndim == 2:
        if index != 0:
            raise FrameRangeError("mapa 2D tem um único quadro (índice 0)")
        src = labelmap
    elif labelmap.ndim == 3:
        if not 0 <= index < labelmap.shape[0]:
            raise FrameRangeError(f"quadro {index} fora de 0..{labelmap.shape[0] - 1}")
        src = labelmap[index]
    else:
        raise FrameRangeError(f"mapa de rótulos com {labelmap.ndim} dimensões não suportado")
    f = scale_for(src.shape[0], src.shape[1], max_dim)
    small = src[::f, ::f]
    top = int(small.max(initial=0))
    if top < 256:
        return FrameData(
            np.ascontiguousarray(small, dtype=np.uint8).tobytes(),
            int(small.shape[0]),
            int(small.shape[1]),
            1,
            "uint8",
            f,
            1.0,
            0.0,
        )
    return FrameData(
        np.ascontiguousarray(small, dtype="<u2").tobytes(),
        int(small.shape[0]),
        int(small.shape[1]),
        1,
        "uint16",
        f,
        1.0,
        0.0,
    )
