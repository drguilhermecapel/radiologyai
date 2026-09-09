"""Descoberta e resolução de plugins de modalidade."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from radiologyai.errors import OutOfScopeError

if TYPE_CHECKING:
    from radiologyai.modalities.base import ModalityPlugin

ENTRY_POINT_GROUP = "radiologyai.modalities"


@lru_cache(maxsize=1)
def _load_plugins() -> dict[str, ModalityPlugin]:
    """Carrega todos os plugins registrados via entry point.

    Falha alto se um plugin declarado não puder ser carregado. O sistema legado
    envolvia cada import em ``except ImportError: self.x = None`` e subia com
    tudo desativado sem avisar; aqui um plugin quebrado é erro de instalação.
    """
    from importlib.metadata import entry_points

    plugins: dict[str, ModalityPlugin] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        cls = ep.load()
        instance = cls()
        for modality in instance.accepted_modalities:
            plugins[modality.upper()] = instance
    return plugins


def available_modalities() -> dict[str, ModalityPlugin]:
    """Mapa ``Modality DICOM -> plugin``."""
    return dict(_load_plugins())


def resolve(modality: str) -> ModalityPlugin:
    """Resolve o plugin para um valor de DICOM Modality.

    Raises:
        OutOfScopeError: nenhuma modalidade registrada aceita esse valor.
    """
    plugins = _load_plugins()
    key = (modality or "").strip().upper()
    if key not in plugins:
        raise OutOfScopeError(
            f"modalidade {modality!r} não suportada; suportadas: {sorted(plugins)}"
        )
    return plugins[key]
