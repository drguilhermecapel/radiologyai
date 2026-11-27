"""Autenticação: login (token opaco), logout (revogação) e ``me``."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from radiologyai.api.deps import PUBLIC, Principal, current_principal, get_platform, perm
from radiologyai.api.schemas import ErrorOut, LoginIn, LoginOut, UserOut
from radiologyai.persistence.security import PERMISSIONS, Role
from radiologyai.worker.runtime import Platform

router = APIRouter(prefix="/auth", tags=["auth"])


def user_out(user: object) -> UserOut:
    from radiologyai.persistence.models import User

    assert isinstance(user, User)
    role = Role(user.role)
    return UserOut(
        id=user.id,
        username=user.username,
        full_name=user.full_name,
        role=role.value,
        professional_id=user.professional_id,
        active=user.active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        permissions=tuple(sorted(p.value for p, roles in PERMISSIONS.items() if role in roles)),
    )


@router.post(
    "/login",
    response_model=LoginOut,
    responses={401: {"model": ErrorOut}, 429: {"model": ErrorOut}},
    openapi_extra=perm(PUBLIC),
)
def login(
    body: LoginIn, platform: Annotated[Platform, Depends(get_platform)], request: Request
) -> LoginOut:
    """Troca usuário e senha por um token opaco de sessão (``Authorization: Bearer``).

    Falhas repetidas são limitadas por (usuário, cliente): quem erra de um IP
    não tranca o usuário nos demais.
    """
    from radiologyai.persistence.users import authenticate, issue_session

    client = request.client.host if request.client is not None else ""
    user = authenticate(platform.db, body.username, body.password, client=client)
    token, expires = issue_session(platform.db, user, platform.settings.session_ttl_hours)
    return LoginOut(token=token, expires_at=expires, user=user_out(user))


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={401: {"model": ErrorOut}},
    openapi_extra=perm("authenticated"),
)
def logout(
    principal: Annotated[Principal, Depends(current_principal)],
    platform: Annotated[Platform, Depends(get_platform)],
) -> None:
    """Revoga a sessão corrente (o token deixa de valer imediatamente)."""
    from radiologyai.persistence.users import revoke_session

    revoke_session(
        platform.db,
        principal.session_id,
        actor_id=principal.user_id,
        actor_username=principal.username,
    )


@router.get(
    "/me",
    response_model=UserOut,
    responses={401: {"model": ErrorOut}},
    openapi_extra=perm("authenticated"),
)
def me(
    principal: Annotated[Principal, Depends(current_principal)],
    platform: Annotated[Platform, Depends(get_platform)],
    request: Request,
) -> UserOut:
    """Usuário da sessão e suas permissões (o frontend esconde o que não pode fazer)."""
    from radiologyai.persistence.models import User

    del request
    with platform.db.session() as session:
        return user_out(session.get(User, principal.user_id))
