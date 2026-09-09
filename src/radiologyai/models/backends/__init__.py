"""Backends de inferência. Importados sob demanda — o núcleo não depende de torch."""

from __future__ import annotations

from radiologyai.models.backends.base import InferenceBackend

__all__ = ["InferenceBackend", "build_backend"]


def build_backend(card: object) -> InferenceBackend:
    """Resolve e constrói o backend declarado por um model card."""
    from radiologyai.models.backends.xrv import XRV_WEIGHTS
    from radiologyai.models.backends.xrv import build_backend as build_xrv

    card_id = getattr(card, "card_id", "")
    if card_id in XRV_WEIGHTS:
        return build_xrv(card)  # type: ignore[arg-type]

    from radiologyai.errors import ModelNotFoundError

    raise ModelNotFoundError(f"nenhum backend registrado para o card {card_id!r}")
