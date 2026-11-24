"""Content-addressed artifact store (heatmaps, label maps, rendered frames, reports).

Files are named by the SHA-256 of their bytes, so an artifact can never be
silently replaced: a different content is a different artifact.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from medai.core.types import ArtifactRef

_EXT = {
    "image/png": ".png",
    "application/x-npz": ".npz",
    "application/json": ".json",
    "application/pdf": ".pdf",
    "application/dicom": ".dcm",
    "application/gzip": ".gz",
    "application/octet-stream": ".bin",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class ArtifactStore:
    """Write-once store rooted at ``base_dir``; layout ``<aa>/<sha256><ext>``."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------- #
    def put_bytes(
        self, data: bytes, media_type: str, shape: tuple[int, ...] | None = None
    ) -> ArtifactRef:
        digest = sha256_bytes(data)
        relpath = f"{digest[:2]}/{digest}{_EXT.get(media_type, '.bin')}"
        target = self.base_dir / relpath
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, target)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return ArtifactRef(sha256=digest, media_type=media_type, relpath=relpath, shape=shape)

    def put_png_gray(self, array: npt.NDArray[np.uint8]) -> ArtifactRef:
        """Store a single-channel 8-bit image (heatmap, 2D mask up to 255 labels)."""
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError("put_png_gray expects a 2D uint8 array")
        buf = io.BytesIO()
        Image.fromarray(array, mode="L").save(buf, format="PNG", optimize=True)
        return self.put_bytes(buf.getvalue(), "image/png", shape=array.shape)

    def put_labelmap(self, array: npt.NDArray[np.integer]) -> ArtifactRef:
        """Store an integer label map of any dimensionality as compressed NPZ."""
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("label maps must have an integer dtype")
        compact = array.astype(np.uint8 if int(array.max(initial=0)) < 256 else np.uint16)
        buf = io.BytesIO()
        np.savez_compressed(buf, labelmap=compact)
        return self.put_bytes(buf.getvalue(), "application/x-npz", shape=tuple(array.shape))

    # -- reading ---------------------------------------------------------- #
    def path(self, ref: ArtifactRef | str) -> Path:
        """Resolve a reference (or a bare sha256) to a file, refusing path traversal."""
        if isinstance(ref, ArtifactRef):
            candidate = (self.base_dir / ref.relpath).resolve()
            if self.base_dir.resolve() not in candidate.parents:
                raise ValueError("artifact path escapes the store")
            if not candidate.is_file():
                raise FileNotFoundError(ref.relpath)
            return candidate
        digest = ref.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("not a sha256 hex digest")
        matches = sorted((self.base_dir / digest[:2]).glob(f"{digest}.*"))
        if not matches:
            raise FileNotFoundError(digest)
        return matches[0]

    def read_bytes(self, ref: ArtifactRef | str) -> bytes:
        data = self.path(ref).read_bytes()
        expected = ref.sha256 if isinstance(ref, ArtifactRef) else ref.lower()
        if sha256_bytes(data) != expected:
            raise OSError(f"artifact {expected} is corrupted on disk")
        return data

    def read_labelmap(self, ref: ArtifactRef | str) -> npt.NDArray[np.integer]:
        with np.load(io.BytesIO(self.read_bytes(ref))) as npz:
            return np.asarray(npz["labelmap"])
