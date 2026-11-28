"""Repositório de artefatos endereçado por conteúdo (mapas de calor, máscaras, laudos).

Arquivos são nomeados pelo sha256 dos próprios bytes: um artefato não pode ser
trocado em silêncio — conteúdo diferente é artefato diferente. Leituras
conferem o hash e recusam referências que escapem do diretório raiz.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.types import ArtifactRef
from radiologyai.errors import ArtifactStoreError, BackendUnavailableError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["MEDIA_EXTENSIONS", "ArtifactStore", "sha256_bytes", "sha256_file"]

MEDIA_EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "application/x-npz": ".npz",
    "application/json": ".json",
    "application/pdf": ".pdf",
    "application/dicom": ".dcm",
    "application/gzip": ".gz",
    "application/octet-stream": ".bin",
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_RELPATH_RE = re.compile(r"^([0-9a-f]{2})/([0-9a-f]{64})(\.[a-z0-9]+)$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """sha256 de um arquivo, lido em blocos (não carrega o arquivo inteiro na memória)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class ArtifactStore:
    """Repositório de escrita única com raiz em ``base_dir``; layout ``<aa>/<sha256><ext>``."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._root = self.base_dir.resolve()

    # -- escrita ---------------------------------------------------------- #
    def put_bytes(
        self, data: bytes, media_type: str, shape: tuple[int, ...] | None = None
    ) -> ArtifactRef:
        """Grava bytes (idempotente) e devolve a referência."""
        if media_type not in MEDIA_EXTENSIONS:
            raise ArtifactStoreError(f"tipo de mídia não suportado: {media_type!r}")
        digest = sha256_bytes(data)
        relpath = f"{digest[:2]}/{digest}{MEDIA_EXTENSIONS[media_type]}"
        target = self._root / relpath
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                Path(tmp).replace(target)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return ArtifactRef(sha256=digest, media_type=media_type, relpath=relpath, shape=shape)

    def put_png_gray(self, array: npt.NDArray[np.uint8]) -> ArtifactRef:
        """Grava imagem 8 bits de um canal (mapa de calor, máscara 2D até 255 rótulos).

        Raises:
            BackendUnavailableError: Pillow ausente (extra ``imaging``).
        """
        import importlib.util

        import numpy as np

        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError("put_png_gray espera array 2D uint8")
        if importlib.util.find_spec("PIL") is None:
            raise BackendUnavailableError(
                "Pillow não instalado; instale o extra: pip install 'radiologyai[imaging]'"
            )
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(array).save(buf, format="PNG", optimize=True)
        return self.put_bytes(buf.getvalue(), "image/png", shape=tuple(array.shape))

    def put_labelmap(self, array: npt.NDArray[np.integer[Any]]) -> ArtifactRef:
        """Grava um mapa de rótulos inteiro de qualquer dimensão como NPZ comprimido."""
        import numpy as np

        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("mapas de rótulos devem ter dtype inteiro")
        if array.size and int(array.min()) < 0:
            raise ValueError("mapas de rótulos não podem ter índices negativos")
        top = int(array.max(initial=0))
        compact = array.astype(np.uint8 if top < 256 else np.uint16 if top < 65536 else np.uint32)
        buf = io.BytesIO()
        np.savez_compressed(buf, labelmap=compact)
        return self.put_bytes(buf.getvalue(), "application/x-npz", shape=tuple(array.shape))

    def put_json(self, data: bytes) -> ArtifactRef:
        return self.put_bytes(data, "application/json")

    # -- leitura ---------------------------------------------------------- #
    def path(self, ref: ArtifactRef | str) -> Path:
        """Resolve uma referência (ou um sha256 nu) para um arquivo, recusando travessia.

        Raises:
            ArtifactStoreError: referência malformada ou fora do repositório.
            FileNotFoundError: artefato inexistente.
        """
        if isinstance(ref, ArtifactRef):
            match = _RELPATH_RE.match(ref.relpath)
            if not match or match.group(2) != ref.sha256 or match.group(1) != ref.sha256[:2]:
                raise ArtifactStoreError(f"relpath de artefato inválido: {ref.relpath!r}")
            candidate = (self._root / ref.relpath).resolve()
            if self._root not in candidate.parents:
                raise ArtifactStoreError("caminho de artefato escapa do repositório")
            if not candidate.is_file():
                raise FileNotFoundError(ref.relpath)
            return candidate
        digest = ref.lower()
        if not _SHA_RE.match(digest):
            raise ArtifactStoreError("não é um sha256 hexadecimal")
        matches = sorted((self._root / digest[:2]).glob(f"{digest}.*"))
        if not matches:
            raise FileNotFoundError(digest)
        return matches[0]

    def read_bytes(self, ref: ArtifactRef | str) -> bytes:
        """Lê e confere o hash do conteúdo.

        Raises:
            ArtifactStoreError: conteúdo corrompido em disco.
        """
        data = self.path(ref).read_bytes()
        expected = ref.sha256 if isinstance(ref, ArtifactRef) else ref.lower()
        if sha256_bytes(data) != expected:
            raise ArtifactStoreError(f"artefato {expected} corrompido em disco")
        return data

    def read_labelmap(self, ref: ArtifactRef | str) -> npt.NDArray[np.integer[Any]]:
        import numpy as np

        with np.load(io.BytesIO(self.read_bytes(ref)), allow_pickle=False) as npz:
            out: npt.NDArray[np.integer[Any]] = np.asarray(npz["labelmap"])
            return out
