"""Fixtures da plataforma: banco SQLite temporário, dublês de modelo, usuários por papel."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def platform_settings(tmp_path: Path) -> Any:
    from radiologyai.persistence.config import PlatformSettings

    return PlatformSettings(
        database_url=f"sqlite:///{tmp_path / 'platform.db'}",
        store_dir=tmp_path / "store",
        deident_key="chave-de-teste-nao-usar-em-producao",
        worker_enabled=False,
        password_min_length=12,
        artifacts_eval_dir=tmp_path / "sem_artefatos",
    )


@pytest.fixture
def fake_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    import platform_support

    from radiologyai.worker.runtime import AnalysisEngine

    monkeypatch.setitem(sys.modules, platform_support.FAKE_MODULE, platform_support)
    return AnalysisEngine(
        registry=platform_support.build_registry(tmp_path / "models"),
        catalog=platform_support.build_catalog(),
    )


@pytest.fixture
def platform(platform_settings: Any, fake_engine: Any) -> Any:
    from radiologyai.worker.runtime import Platform

    return Platform.create(platform_settings, engine=fake_engine)


@pytest.fixture
def api(platform: Any) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from radiologyai.api import create_app
    from radiologyai.persistence.users import reset_throttle

    reset_throttle()
    with TestClient(create_app(platform=platform, start_worker=False)) as client:
        yield client


@pytest.fixture
def make_user(platform: Any) -> Callable[..., dict[str, str]]:
    """Cria usuário e devolve o cabeçalho de autorização de uma sessão nova."""
    import platform_support

    from radiologyai.persistence.users import create_user, issue_session

    def build(role: str, username: str | None = None, **extra: Any) -> dict[str, str]:
        user = create_user(
            platform.db,
            username=username or f"{role}-user",
            full_name=extra.pop("full_name", f"Pessoa {role}"),
            role=role,
            password=platform_support.PASSWORD,
            **extra,
        )
        token, _ = issue_session(platform.db, user, 1.0)
        return {"Authorization": f"Bearer {token}"}

    return build


@pytest.fixture
def headers(make_user: Callable[..., dict[str, str]]) -> dict[str, dict[str, str]]:
    """Um usuário por papel."""
    return {
        "admin": make_user("admin"),
        "physician": make_user("physician", professional_id="CRM-SP 000000 (teste)"),
        "technician": make_user("technician"),
        "viewer": make_user("viewer"),
    }
