"""Configuração — segredos nunca têm default inseguro."""

from __future__ import annotations

import pytest

from radiologyai.config.settings import Settings


def test_no_default_deident_key():
    """medai_config.json do legado trazia 'your-secret-key-here' commitado."""
    assert Settings(_env_file=None).deident_key is None


def test_require_deident_key_fails_loud_when_absent():
    with pytest.raises(RuntimeError, match="MEDAI_DEIDENT_KEY"):
        Settings(_env_file=None).require_deident_key()


def test_error_explains_why_ephemeral_key_is_wrong():
    with pytest.raises(RuntimeError, match="estáveis entre execuções"):
        Settings(_env_file=None).require_deident_key()


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv("MEDAI_DEIDENT_KEY", "chave-de-teste")
    assert Settings(_env_file=None).require_deident_key() == b"chave-de-teste"


def test_paths_are_relative_by_default():
    """Nenhum caminho absoluto de outra máquina (/home/ubuntu/... no legado)."""
    s = Settings(_env_file=None)
    for path in (s.data_dir, s.models_dir, s.artifacts_dir, s.logs_dir):
        assert not path.is_absolute(), f"{path} não deve ser absoluto por default"
