from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiologyai.errors import ModelNotFoundError
from radiologyai.models import backends
from radiologyai.models.backends import OnnxBackend as ExportedOnnxBackend
from radiologyai.models.backends import build_backend
from radiologyai.models.backends.onnx import OnnxBackend
from radiologyai.models.card import ModelCard


class _FakeTensorInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, ...]] = []

    def get_inputs(self) -> list[_FakeTensorInfo]:
        return [_FakeTensorInfo("input")]

    def get_outputs(self) -> list[_FakeTensorInfo]:
        return [_FakeTensorInfo("output")]

    def run(
        self,
        _output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        arr = input_feed["input"]
        self.seen_shapes.append(tuple(int(x) for x in arr.shape))
        return [np.array([[0.2, 0.8]], dtype=np.float32)]


def _make_card_and_weights(
    tmp_path: Path,
    *,
    weights_uri: str | None = None,
) -> tuple[ModelCard, Path]:
    raw = b"fake-onnx-weights-for-unit-test"
    weights_path = tmp_path / "model.onnx"
    weights_path.write_bytes(raw)
    sha256 = hashlib.sha256(raw).hexdigest()
    card = ModelCard(
        card_id="test-onnx-card",
        display_name="Test ONNX Card",
        version="0.1.0",
        modality="CR",
        backend="onnx",
        task="classification",
        labels=("Atelectasis", "Cardiomegaly"),
        input_shape=(1, 1, 8, 8),
        weights_uri=weights_uri or "https://example.invalid/model.onnx",
        weights_sha256=sha256,
        weights_size_bytes=len(raw),
        trained_on=("synthetic",),
        license="MIT",
    )
    return card, weights_path


def test_onnx_backend_is_exported_in_backends_package() -> None:
    assert ExportedOnnxBackend is OnnxBackend
    assert "OnnxBackend" in backends.__all__


def test_build_backend_onnx_with_explicit_weights_path(tmp_path: Path) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    session = _FakeSession()
    backend: Any = build_backend(
        card,
        weights_path=weights_path,
        session_factory=lambda _p, _prov: session,
    )

    assert isinstance(backend, OnnxBackend)
    out = backend.predict(np.zeros((8, 8), dtype=np.float32))
    assert out.shape == (2,)
    assert np.allclose(out, [0.2, 0.8])
    assert session.seen_shapes[-1] == (1, 1, 8, 8)


@pytest.mark.parametrize("use_file_scheme", [True, False])
def test_build_backend_onnx_resolves_local_weights_uri(
    tmp_path: Path,
    use_file_scheme: bool,
) -> None:
    _, weights_path = _make_card_and_weights(tmp_path)
    uri = weights_path.as_uri() if use_file_scheme else str(weights_path)
    card, _ = _make_card_and_weights(tmp_path, weights_uri=uri)
    session = _FakeSession()

    backend: Any = build_backend(
        card,
        session_factory=lambda _p, _prov: session,
    )

    assert isinstance(backend, OnnxBackend)
    out = backend.predict(np.zeros((8, 8), dtype=np.float32))
    assert out.shape == (2,)
    assert np.allclose(out, [0.2, 0.8])


def test_build_backend_onnx_raises_when_local_weights_missing(tmp_path: Path) -> None:
    card, _ = _make_card_and_weights(tmp_path)
    with pytest.raises(ModelNotFoundError):
        build_backend(card)
