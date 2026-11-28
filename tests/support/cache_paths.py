"""Raiz do cache local de pesos e datasets usado pelos testes com dados reais.

Configurável por ``MEDAI_CACHE``. Na CI aponta para um diretório vazio: os
testes que dependem de pesos ou amostras reais são pulados explicitamente, com
motivo, em vez de quebrar na coleta por falta de permissão em ``/root``.
"""

from __future__ import annotations

import os
from pathlib import Path

MEDAI_CACHE = Path(os.environ.get("MEDAI_CACHE", "/root/medai-cache"))
