"""Pesos extraídos de arquivo são re-hasheados a cada uso; cache de hash não é enganado."""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from typing import Any

import pytest
from analysis_support import WEIGHT_BYTES, manifest_dict

from radiologyai.errors import IntegrityCheckError


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _archive_registry(make_registry: Any) -> Any:
    payload = _zip({"model/weights.pt": b"pesos-verdadeiros"})
    data = manifest_dict("fake-arch", "Brightness")
    data["weights"][0].update(
        {
            "filename": "main.zip",
            "extract": "zip",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    )
    registry = make_registry(data, with_weights=False)
    target = registry.model_dir("fake-arch") / "main.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    return registry


@pytest.mark.requirement("REQ-179")
def test_tampered_extracted_checkpoint_is_replaced_from_verified_archive(
    make_registry: Any,
) -> None:
    registry = _archive_registry(make_registry)
    extracted = registry.verify("fake-arch").path("main") / "model" / "weights.pt"
    assert extracted.read_bytes() == b"pesos-verdadeiros"
    extracted.write_bytes(b"pesos-trocados!!!")  # mesmo tamanho, conteúdo trocado
    again = registry.verify("fake-arch").path("main") / "model" / "weights.pt"
    assert again.read_bytes() == b"pesos-verdadeiros"


@pytest.mark.requirement("REQ-179")
def test_extra_file_planted_in_extracted_dir_forces_reextraction(make_registry: Any) -> None:
    registry = _archive_registry(make_registry)
    root = registry.verify("fake-arch").path("main")
    (root / "model" / "intruso.pt").write_bytes(b"x")
    root = registry.verify("fake-arch").path("main")
    assert not (root / "model" / "intruso.pt").exists()


@pytest.mark.requirement("REQ-179")
def test_same_size_overwrite_with_restored_mtime_is_rehashed(make_registry: Any) -> None:
    data = manifest_dict("fake-bright", "Brightness")
    registry = make_registry(data)
    path = registry.model_dir("fake-bright") / "main.bin"
    registry.verify("fake-bright")
    stat = path.stat()
    path.write_bytes(bytes(b ^ 0xFF for b in WEIGHT_BYTES))  # mesmo tamanho
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # mtime restaurado
    with pytest.raises(IntegrityCheckError, match="integridade"):
        registry.verify("fake-bright")
