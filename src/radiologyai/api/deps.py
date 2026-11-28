"""Dependências da API: plataforma, autenticação por token opaco e RBAC.

Cada rota declara UMA permissão por ``openapi_extra={"x-permission": ...}`` e
a exige com ``Depends(require(...))``; ``public`` marca as poucas rotas sem
autenticação (saúde, login, métricas medidas). Os testes conferem rota a rota.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from radiologyai.persistence.security import Permission, allowed, hash_token
from radiologyai.worker.runtime import (
    Platform,  # em tempo de execução: o FastAPI resolve a anotação
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from radiologyai.worker.queue import Worker

__all__ = [
    "PUBLIC",
    "Principal",
    "current_principal",
    "get_platform",
    "get_worker",
    "perm",
    "require",
]

PUBLIC = "public"
_SEEN_THROTTLE = timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    username: str
    full_name: str
    role: str
    professional_id: str | None
    session_id: str

    def can(self, permission: Permission) -> bool:
        return allowed(self.role, permission)


def perm(permission: Permission | str) -> dict[str, Any]:
    """``openapi_extra`` que documenta a permissão da rota."""
    return {"x-permission": str(permission)}


def get_platform(request: Request) -> Platform:
    platform: Platform = request.app.state.platform
    return platform


def get_worker(request: Request) -> Worker | None:
    worker: Worker | None = getattr(request.app.state, "worker", None)
    return worker


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def current_principal(
    request: Request, platform: Annotated[Platform, Depends(get_platform)]
) -> Principal:
    """Resolve o token ``Authorization: Bearer``; sessão revogada/expirada => 401."""
    from radiologyai.persistence.models import AuthSession, User, utcnow

    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("autenticação exigida (Authorization: Bearer <token>)")
    digest = hash_token(token.strip())
    from sqlalchemy import select

    with platform.db.session() as session:
        row = session.execute(
            select(AuthSession).where(AuthSession.token_sha256 == digest)
        ).scalar_one_or_none()
        now = utcnow()
        if row is None or row.revoked_at is not None or row.expires_at <= now:
            raise _unauthorized("sessão inválida, expirada ou revogada")
        user = session.get(User, row.user_id)
        if user is None or not user.active:
            raise _unauthorized("usuário inativo")
        if now - row.last_seen_at > _SEEN_THROTTLE:
            row.last_seen_at = now
        return Principal(
            user_id=user.id,
            username=user.username,
            full_name=user.full_name,
            role=user.role,
            professional_id=user.professional_id,
            session_id=row.id,
        )


def require(permission: Permission) -> Callable[..., Principal]:
    """Dependência que exige ``permission``; papel sem permissão => 403."""

    def dependency(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
        if not principal.can(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"papel {principal.role!r} não tem a permissão {permission.value!r}",
            )
        return principal

    dependency.__name__ = f"require_{permission.name.lower()}"
    return dependency


def session_expiry(hours: float) -> datetime:
    return datetime.now(UTC) + timedelta(hours=hours)
