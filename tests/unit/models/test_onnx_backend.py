"""Unit tests for the ONNX Runtime inference backend."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

from radiologyai.errors import WeightsIntegrityError
from radiologyai.models.backends.onnx import OnnxBackend
from radiologyai.models.card import ModelCard


@dataclass(frozen=True)
class _FakeInput:
    name: str


class _FakeSession:
    def __init__(
        self,
        inputs: Sequence[_FakeInput] | None = None,
        outputs: Sequence[npt.NDArray[np.float32]] | None = None,
    ) -> None:
        self._inputs = list(inputs) if inputs is not None else [_FakeInput("input")]
        default_out = [np.array([[0.2, 0.8]], dtype=np.float32)]
        self._outputs = list(outputs) if outputs is not None else default_out
        self.seen_shapes: list[tuple[int, ...]] = []

    def get_inputs(self) -> Sequence[_FakeInput]:
        return self._inputs

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: dict[str, npt.NDArray[np.float32]],
    ) -> Sequence[npt.NDArray[np.float32]]:
        del output_names
        tensor = input_feed[self._inputs[0].name]
        self.seen_shapes.append(tuple(tensor.shape))
        return self._outputs


def _make_card_and_weights(
    tmp_path: Path,
    *,
    backend: Literal["torch", "onnx"] = "onnx",
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
        backend=backend,
        task="classification",
        labels=("Atelectasis", "Cardiomegaly"),
        input_shape=(1, 1, 8, 8),
        weights_uri="https://example.invalid/model.onnx",
        weights_sha256=sha256,
        weights_size_bytes=len(raw),
        trained_on=("synthetic",),
        license="MIT",
    )
    return card, weights_path


def test_onnx_backend_valid_flow(tmp_path: Path) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    session = _FakeSession()
    backend = OnnxBackend(
        card,
        weights_path,
        providers=("CPUExecutionProvider",),
        session_factory=lambda _p, _prov: session,
    )

    assert backend.labels == ("Atelectasis", "Cardiomegaly")

    img_2d = np.zeros((8, 8), dtype=np.float32)
    out_2d = backend.predict(img_2d)
    assert out_2d.shape == (2,)
    assert np.allclose(out_2d, [0.2, 0.8])
    assert session.seen_shapes[-1] == (1, 1, 8, 8)

    img_3d = np.zeros((1, 8, 8), dtype=np.float32)
    out_3d = backend.predict(img_3d)
    assert out_3d.shape == (2,)
    assert np.allclose(out_3d, [0.2, 0.8])
    assert session.seen_shapes[-1] == (1, 1, 8, 8)

    img_4d = np.zeros((1, 1, 8, 8), dtype=np.float32)
    out_4d = backend.predict(img_4d)
    assert out_4d.shape == (2,)
    assert session.seen_shapes[-1] == (1, 1, 8, 8)

    batch_out = backend.predict_batch([img_3d, img_4d])
    assert batch_out.shape == (2, 2)

    meta = backend.describe()
    assert meta == {
        "backend": "onnx",
        "card_id": "test-onnx-card",
        "version": "0.1.0",
        "weights_sha256": card.weights_sha256,
        "providers": ["CPUExecutionProvider"],
        "input_name": "input",
        "research_only": True,
    }


def test_onnx_backend_rejects_non_onnx_card(tmp_path: Path) -> None:
    card, weights_path = _make_card_and_weights(tmp_path, backend="torch")
    with pytest.raises(ValueError, match="onnx"):
        OnnxBackend(card, weights_path, session_factory=lambda _p, _prov: _FakeSession())


def test_onnx_backend_rejects_tampered_weights(tmp_path: Path) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    weights_path.write_bytes(b"tampered-weights-bytes-do-not-match")
    with pytest.raises(WeightsIntegrityError):
        OnnxBackend(card, weights_path, session_factory=lambda _p, _prov: _FakeSession())


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_onnx_backend_rejects_non_finite_input(
    tmp_path: Path,
    bad_val: float,
) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    backend = OnnxBackend(
        card,
        weights_path,
        session_factory=lambda _p, _prov: _FakeSession(),
    )
    bad_img = np.zeros((1, 8, 8), dtype=np.float32)
    bad_img[0, 0, 0] = bad_val
    with pytest.raises(ValueError, match="NaN or Inf"):
        backend.predict(bad_img)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_onnx_backend_rejects_non_finite_output(
    tmp_path: Path,
    bad_val: float,
) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    bad_out = np.array([[0.1, bad_val]], dtype=np.float32)
    session = _FakeSession(outputs=[bad_out])
    backend = OnnxBackend(
        card,
        weights_path,
        session_factory=lambda _p, _prov: session,
    )
    img = np.zeros((1, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or Inf"):
        backend.predict(img)


def test_onnx_backend_rejects_wrong_output_count(tmp_path: Path) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    session = _FakeSession(outputs=[np.array([[0.1, 0.2, 0.3]], dtype=np.float32)])
    backend = OnnxBackend(
        card,
        weights_path,
        session_factory=lambda _p, _prov: session,
    )
    with pytest.raises(ValueError, match="Expected 2 outputs"):
        backend.predict(np.zeros((1, 8, 8), dtype=np.float32))


def test_onnx_backend_rejects_session_without_inputs(tmp_path: Path) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    session = _FakeSession(inputs=[])
    with pytest.raises(ValueError, match="no input tensors"):
        OnnxBackend(card, weights_path, session_factory=lambda _p, _prov: session)


@pytest.mark.parametrize(
    "empty_outputs",
    [
        [],
        [np.array([], dtype=np.float32)],
    ],
)
def test_onnx_backend_rejects_empty_output(
    tmp_path: Path,
    empty_outputs: list[npt.NDArray[np.float32]],
) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    session = _FakeSession(outputs=empty_outputs)
    backend = OnnxBackend(
        card,
        weights_path,
        session_factory=lambda _p, _prov: session,
    )
    with pytest.raises(RuntimeError):
        backend.predict(np.zeros((1, 8, 8), dtype=np.float32))


def test_onnx_backend_missing_onnxruntime_raises_import_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card, weights_path = _make_card_and_weights(tmp_path)
    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None) -> Any:
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(ImportError, match="onnxruntime"):
        OnnxBackend(card, weights_path)
