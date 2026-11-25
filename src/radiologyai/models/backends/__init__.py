"""Backends de inferência. Importados sob demanda — o núcleo não depende de torch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from radiologyai.models.backends.base import InferenceBackend
from radiologyai.models.backends.onnx import OnnxBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from radiologyai.models.backends.onnx import SessionFactory

__all__ = ["InferenceBackend", "OnnxBackend", "build_backend"]


def build_backend(
    card: object,
    *,
    weights_path: str | Path | None = None,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    session_factory: SessionFactory | None = None,
) -> InferenceBackend:
    """Resolve e constrói o backend declarado por um model card."""
    card_id = getattr(card, "card_id", "")
    backend_name = getattr(card, "backend", "")

    if backend_name == "onnx":
        resolved_weights = weights_path
        if resolved_weights is None:
            weights_uri = getattr(card, "weights_uri", "")
            if isinstance(weights_uri, str) and weights_uri.startswith("file://"):
                resolved_weights = Path(weights_uri.removeprefix("file://"))
            elif isinstance(weights_uri, str) and weights_uri and "://" not in weights_uri:
                resolved_weights = Path(weights_uri)
            else:
                from radiologyai.errors import ModelNotFoundError

                raise ModelNotFoundError(
                    f"caminho de pesos local ausente para o card ONNX {card_id!r}"
                )

        return OnnxBackend(
            card,  # type: ignore[arg-type]
            resolved_weights,
            providers=providers,
            session_factory=session_factory,
        )

    from radiologyai.models.backends.xrv import XRV_WEIGHTS
    from radiologyai.models.backends.xrv import build_backend as build_xrv

    if card_id in XRV_WEIGHTS:
        return build_xrv(card)  # type: ignore[arg-type]

    from radiologyai.errors import ModelNotFoundError

    raise ModelNotFoundError(f"nenhum backend registrado para o card {card_id!r}")
