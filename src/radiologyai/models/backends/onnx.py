"""ONNX Runtime inference backend (research use only, not a medical device)."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from radiologyai.models.backends.base import InferenceBackend
from radiologyai.models.card import ModelCard


class OnnxSession(Protocol):
    """Minimal protocol for an ONNX Runtime inference session."""

    def get_inputs(self) -> Sequence[Any]:
        """Return input metadata objects with a ``name`` attribute."""
        ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: dict[str, npt.NDArray[np.float32]],
    ) -> Sequence[Any]:
        """Execute inference and return sequence of output arrays."""
        ...


SessionFactory = Callable[[Path, Sequence[str]], OnnxSession]


class OnnxBackend(InferenceBackend):
    """Fail-closed ONNX Runtime backend for research-only inference."""

    def __init__(
        self,
        card: ModelCard,
        weights_path: str | Path,
        *,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        session_factory: SessionFactory | None = None,
    ) -> None:
        if card.backend != "onnx":
            raise ValueError(f"Expected card.backend == 'onnx', got {card.backend!r}.")

        path = Path(weights_path)
        card.verify_weights(path)

        if session_factory is not None:
            session = session_factory(path, providers)
        else:
            ort = importlib.import_module("onnxruntime")
            session = ort.InferenceSession(str(path), providers=list(providers))

        inputs = session.get_inputs()
        if not inputs:
            raise ValueError("ONNX session reported no input tensors.")

        self._card = card
        self._weights_path = path
        self._providers = tuple(providers)
        self._session = session
        self._input_name: str = str(inputs[0].name)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._card.labels

    def predict(self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        arr = np.asarray(image, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Input image contains NaN or Inf values.")

        batch = np.expand_dims(arr, axis=0) if arr.ndim == 3 else arr
        outputs = self._session.run(None, {self._input_name: batch})
        if not outputs:
            raise RuntimeError("ONNX session returned no outputs.")

        raw = np.asarray(outputs[0], dtype=np.float32)
        if raw.size == 0:
            raise RuntimeError("ONNX session returned an empty output array.")

        probs = raw.reshape(-1)
        if not np.isfinite(probs).all():
            raise ValueError("ONNX model output contains NaN or Inf values.")

        expected = len(self._card.labels)
        if probs.shape != (expected,):
            raise ValueError(f"Expected {expected} outputs, got shape {probs.shape}.")

        return probs.astype(np.float32, copy=False)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "onnx",
            "card_id": self._card.card_id,
            "version": self._card.version,
            "weights_sha256": self._card.weights_sha256,
            "providers": list(self._providers),
            "input_name": self._input_name,
            "research_only": True,
        }
