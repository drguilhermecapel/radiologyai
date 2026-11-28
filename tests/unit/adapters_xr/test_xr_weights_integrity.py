"""Pesos de RX: conferência do sha256 sobre os bytes desserializados e pickle só declarado."""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

from radiologyai.analysis.adapters.xr_common import load_torch_verified, read_verified
from radiologyai.analysis.base import ResolvedWeights
from radiologyai.errors import IntegrityCheckError

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _weights(path: Path, data: bytes, *, unsafe: bool = False) -> ResolvedWeights:
    path.write_bytes(data)
    return ResolvedWeights(
        files={"model": path},
        sha256={"model": hashlib.sha256(data).hexdigest()},
        unsafe_pickle={"model": unsafe},
    )


@pytest.mark.requirement("REQ-127")
def test_read_verified_detects_swapped_file(tmp_path: Path) -> None:
    weights = _weights(tmp_path / "w.bin", b"conteudo original")
    assert read_verified(weights, "model") == b"conteudo original"
    (tmp_path / "w.bin").write_bytes(b"conteudo trocado depois do registro")
    with pytest.raises(IntegrityCheckError, match="difere do fixado"):
        read_verified(weights, "model")


@pytest.mark.skipif(not HAS_TORCH, reason="torch ausente (extra ml)")
@pytest.mark.requirement("REQ-127")
def test_pickle_requires_manifest_declaration(tmp_path: Path) -> None:
    import torch

    buf = io.BytesIO()
    torch.save({"w": torch.zeros(2)}, buf)
    weights = _weights(tmp_path / "w.pt", buf.getvalue(), unsafe=False)
    with pytest.raises(IntegrityCheckError, match="unsafe_pickle"):
        load_torch_verified(weights, "model", allow_pickle=True)
    state = load_torch_verified(weights, "model", allow_pickle=False)
    assert torch.equal(state["w"], torch.zeros(2))


@pytest.mark.skipif(not HAS_TORCH, reason="torch ausente (extra ml)")
@pytest.mark.requirement("REQ-127")
def test_weights_only_refuses_arbitrary_objects(tmp_path: Path) -> None:
    import pickle

    import torch

    buf = io.BytesIO()
    torch.save({"obj": IntegrityCheckError("x", "y")}, buf, pickle_module=pickle)
    weights = _weights(tmp_path / "w.pt", buf.getvalue(), unsafe=False)
    with pytest.raises(pickle.UnpicklingError):
        load_torch_verified(weights, "model", allow_pickle=False)
