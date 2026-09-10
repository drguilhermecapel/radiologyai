"""Camada de API REST. Requer o extra [api]."""

from __future__ import annotations

from radiologyai.api.app import API_PREFIX, DISCLAIMER, create_app

__all__ = ["API_PREFIX", "DISCLAIMER", "create_app"]
