"""Gestão de usuários (somente administrador). Sem exclusão: desativação revoga sessões."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.auth import user_out
from radiologyai.api.schemas import ErrorOut, UserCreateIn, UserOut, UserUpdateIn
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(prefix="/users", tags=["users"])
_P = Permission.USERS_MANAGE


@router.get("", response_model=list[UserOut], openapi_extra=perm(_P))
def list_users(
    principal: Annotated[Principal, Depends(require(_P))],
    platform: Annotated[Platform, Depends(get_platform)],
) -> list[UserOut]:
    """Todos os usuários."""
    from sqlalchemy import select

    from radiologyai.persistence.models import User

    del principal
    with platform.db.session() as session:
        return [
            user_out(u) for u in session.execute(select(User).order_by(User.username)).scalars()
        ]


@router.post(
    "",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorOut}, 422: {"model": ErrorOut}},
    openapi_extra=perm(_P),
)
def create(
    principal: Annotated[Principal, Depends(require(_P))],
    platform: Annotated[Platform, Depends(get_platform)],
    body: UserCreateIn,
) -> UserOut:
    """Cria usuário (senha com política mínima; hash argon2id)."""
    from radiologyai.persistence.users import create_user

    user = create_user(
        platform.db,
        username=body.username,
        full_name=body.full_name,
        role=body.role,
        password=body.password,
        professional_id=body.professional_id,
        min_length=platform.settings.password_min_length,
        actor_id=principal.user_id,
        actor_username=principal.username,
    )
    return user_out(user)


@router.patch(
    "/{user_id}",
    response_model=UserOut,
    responses={404: {"model": ErrorOut}, 422: {"model": ErrorOut}},
    openapi_extra=perm(_P),
)
def update(
    principal: Annotated[Principal, Depends(require(_P))],
    platform: Annotated[Platform, Depends(get_platform)],
    user_id: str,
    body: UserUpdateIn,
) -> UserOut:
    """Altera nome, papel, registro profissional, senha ou ativação.

    Mudança de papel, senha ou desativação revoga as sessões do usuário.
    """
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import User, utcnow
    from radiologyai.persistence.security import check_password_policy, hash_password
    from radiologyai.persistence.users import revoke_user_sessions

    if user_id == principal.user_id and (body.active is False or body.role not in (None, "admin")):
        raise HTTPException(status_code=409, detail="não remova o próprio acesso de administrador")
    changed: list[str] = []
    with platform.db.session() as session:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="usuário inexistente")
        if body.full_name is not None:
            user.full_name = body.full_name
            changed.append("full_name")
        if body.professional_id is not None:
            user.professional_id = body.professional_id or None
            changed.append("professional_id")
        if body.role is not None and body.role != user.role:
            user.role = body.role
            changed.append("role")
        if body.active is not None and body.active != user.active:
            user.active = body.active
            changed.append("active")
        if body.password is not None:
            check_password_policy(body.password, platform.settings.password_min_length)
            user.password_hash = hash_password(body.password)
            changed.append("password")
        user.updated_at = utcnow()
    if {"role", "active", "password"} & set(changed):
        revoke_user_sessions(platform.db, user_id)
    append_event(
        platform.db,
        "user.updated",
        actor_id=principal.user_id,
        actor_username=principal.username,
        subject_type="user",
        subject_id=user_id,
        payload={"fields": changed, "role": user.role, "active": user.active},
    )
    return user_out(user)
