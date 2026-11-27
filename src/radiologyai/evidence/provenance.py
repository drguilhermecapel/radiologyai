"""Proveniência de uma medição: git, pesos e comando — lidos, nunca digitados."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

from radiologyai.errors import EvidenceError

REPO_ROOT = Path(__file__).resolve().parents[3]
CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """sha256 do conteúdo de um arquivo (lido em blocos de 1 MiB)."""
    p = Path(path)
    if not p.is_file():
        raise EvidenceError(f"arquivo para hash não encontrado: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


# Arquivos que determinam o resultado de uma medição: o código que mede e julga,
# o protocolo, o plano (declarações de vazamento) e o ambiente travado. Mudanças de
# outras frentes (adaptadores, API, interface) não alteram um registro medido a
# partir de predições já gravadas, e por isso não o tornam irreprodutível.
MEASUREMENT_SCOPE: tuple[str, ...] = (
    "src/radiologyai/evidence",
    "src/radiologyai/evaluation",
    "src/radiologyai/calibration",
    "src/radiologyai/catalog/schema.py",
    "src/radiologyai/catalog/protocol.yaml",
    "src/radiologyai/errors.py",
    "scripts/evidence_run.py",
    "evidence/measured/plan.yaml",
    "pyproject.toml",
    "uv.lock",
)


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=repo
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError(f"não foi possível ler o estado git em {repo}: {exc}") from exc


def dirty_paths(
    repo: Path = REPO_ROOT, scope: Sequence[str] | None = MEASUREMENT_SCOPE
) -> list[str]:
    """Arquivos não commitados que afetam a medição.

    Com ``scope`` (padrão: :data:`MEASUREMENT_SCOPE`), conta modificados **e não
    rastreados** dentro do escopo — um módulo novo ainda não commitado também
    torna o registro irreprodutível a partir do sha. Com ``scope=None``, conta só
    os rastreados modificados da árvore inteira.

    Raises:
        EvidenceError: git indisponível ou diretório fora de um repositório.
    """
    if scope is None:
        status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    else:
        status = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *scope)
    return sorted(line[3:] for line in status.splitlines() if line.strip())


def git_state(
    repo: Path = REPO_ROOT, scope: Sequence[str] | None = MEASUREMENT_SCOPE
) -> tuple[str, bool]:
    """``(sha do HEAD, código da medição sujo?)``.

    "Sujo" segue :func:`dirty_paths`: um registro medido a partir de código não
    commitado não é reproduzível a partir do sha.

    Raises:
        EvidenceError: git indisponível ou diretório fora de um repositório. Uma
            medição sem sha não é registrada — não existe sha "desconhecido".
    """
    sha = _git(repo, "rev-parse", "HEAD").strip()
    if len(sha) != 40:
        raise EvidenceError(f"sha git inesperado: {sha!r}")
    return sha, bool(dirty_paths(repo, scope))
