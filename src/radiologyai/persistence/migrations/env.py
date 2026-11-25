"""Ambiente Alembic da plataforma.

A conexão vem de ``config.attributes["connection"]`` (``Database.migrate``) ou,
na linha de comando, da URL ``sqlalchemy.url``.
"""

from __future__ import annotations

from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool

from radiologyai.persistence.models import Base

config = context.config
target_metadata = Base.metadata


def _run(connection: Any) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with engine.connect() as conn:
        _run(conn)


if context.is_offline_mode():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
else:
    run_migrations_online()
