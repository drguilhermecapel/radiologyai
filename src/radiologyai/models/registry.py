"""Registro de model cards."""

from __future__ import annotations

from pathlib import Path

from radiologyai.errors import ModelNotFoundError
from radiologyai.models.card import ModelCard

CARDS_DIR = Path(__file__).parent / "cards"


def cards_dir() -> Path:
    """Diretório dos model cards versionados."""
    return CARDS_DIR


def list_cards(directory: Path | None = None) -> list[ModelCard]:
    """Carrega todos os model cards de um diretório.

    Um card malformado interrompe a listagem — não é filtrado em silêncio.
    """
    d = directory or CARDS_DIR
    if not d.is_dir():
        return []
    return [ModelCard.from_yaml(p) for p in sorted(d.glob("*.yaml"))]


def get_card(card_id: str, directory: Path | None = None) -> ModelCard:
    """Resolve um model card pelo seu ``card_id``.

    Raises:
        ModelNotFoundError: nenhum card com esse id.
    """
    for card in list_cards(directory):
        if card.card_id == card_id:
            return card
    available = [c.card_id for c in list_cards(directory)]
    raise ModelNotFoundError(
        f"model card {card_id!r} não encontrado; disponíveis: {available or '(nenhum)'}"
    )
