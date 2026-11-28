"""Persistência da plataforma: banco relacional, sessões, auditoria e artefatos.

Requer o extra ``platform`` (SQLAlchemy 2, Alembic, argon2-cffi). SQLite é o
padrão (``MEDAI_DATABASE_URL`` ausente); Postgres entra pela URL.

Módulos:

* :mod:`radiologyai.persistence.config` — ``PlatformSettings`` (variáveis ``MEDAI_``).
* :mod:`radiologyai.persistence.db` — motor, sessões e migrações Alembic.
* :mod:`radiologyai.persistence.models` — tabelas ORM.
* :mod:`radiologyai.persistence.security` — senhas argon2, tokens opacos, papéis.
* :mod:`radiologyai.persistence.audit` — eventos de auditoria encadeados por hash.
* :mod:`radiologyai.persistence.blobs` — artefatos endereçados por conteúdo + índice.

Importar este pacote não importa SQLAlchemy; os submódulos o fazem.
"""

from __future__ import annotations

__all__: list[str] = []
