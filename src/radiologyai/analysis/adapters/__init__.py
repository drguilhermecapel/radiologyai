"""Adaptadores de modelos reais (um módulo por família de modelo).

Somente classes deste pacote podem ser nomeadas em ``adapter:`` de um manifesto.
Cada módulo importa dependências pesadas (torch, monai, ...) DENTRO de
``load``/``predict`` e levanta ``BackendUnavailableError`` nomeando o extra
quando ausentes — nunca ``except ImportError``.
"""

from __future__ import annotations
