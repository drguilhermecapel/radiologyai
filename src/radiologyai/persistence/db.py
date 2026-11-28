"""Motor, sessões e migrações (Alembic) do banco da plataforma.

O esquema só muda por migração Alembic (``persistence/migrations``); a
aplicação nunca chama ``create_all`` em produção. ``Database.migrate()`` aplica
as migrações pendentes até ``head`` e é idempotente.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.errors import BackendUnavailableError, ConfigurationError

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session, sessionmaker

__all__ = ["MIGRATIONS_DIR", "Database", "require_platform"]

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def require_platform() -> None:
    """Exige o extra ``platform``; sem ele, erro que nomeia o extra."""
    missing = [
        name
        for name in ("sqlalchemy", "alembic", "argon2")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise BackendUnavailableError(
            f"plataforma indisponível: faltam {missing}. "
            "Instale com: pip install 'radiologyai[platform]'"
        )


class Database:
    """Motor SQLAlchemy + fábrica de sessões para uma URL."""

    def __init__(self, url: str) -> None:
        require_platform()
        from sqlalchemy import create_engine, event
        from sqlalchemy.orm import sessionmaker

        self.url = url
        kwargs: dict[str, Any] = {"future": True}
        if url.startswith("sqlite"):
            path = url.split("///", 1)[1] if "///" in url else ""
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine: Engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA busy_timeout=30000")
                if path and path != ":memory:":
                    cur.execute("PRAGMA journal_mode=WAL")
                cur.close()

        self.sessions: sessionmaker[Session] = sessionmaker(
            self.engine, expire_on_commit=False, future=True
        )

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Sessão com commit ao sair sem erro e rollback em exceção."""
        session = self.sessions()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def alembic_config(self) -> Any:
        from alembic.config import Config

        cfg = Config()
        cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
        cfg.set_main_option("sqlalchemy.url", self.url.replace("%", "%%"))
        return cfg

    def migrate(self) -> str:
        """Aplica migrações até ``head``; devolve a revisão corrente.

        Raises:
            ConfigurationError: banco em revisão desconhecida por este código.
        """
        from alembic import command
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory

        cfg = self.alembic_config()
        with self.engine.begin() as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
        script = ScriptDirectory.from_config(cfg)
        with self.engine.connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
        if current != script.get_current_head():
            raise ConfigurationError(
                f"banco na revisão {current!r}; este código espera {script.get_current_head()!r}"
            )
        return str(current)

    def revision(self) -> str | None:
        """Revisão Alembic corrente do banco (``None`` se não migrado)."""
        from alembic.runtime.migration import MigrationContext

        with self.engine.connect() as conn:
            rev = MigrationContext.configure(conn).get_current_revision()
        return None if rev is None else str(rev)

    def head(self) -> str:
        from alembic.script import ScriptDirectory

        return str(ScriptDirectory.from_config(self.alembic_config()).get_current_head())

    def dispose(self) -> None:
        self.engine.dispose()
