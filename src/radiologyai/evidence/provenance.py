"""Proveniência de uma medição: git, pesos e comando — lidos, nunca digitados."""

from __future__ import annotations

import hashlib
import subprocess
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


def git_state(repo: Path = REPO_ROOT) -> tuple[str, bool]:
    """``(sha do HEAD, árvore suja?)``.

    "Suja" considera apenas arquivos rastreados modificados: um registro medido
    a partir de código não commitado não é reproduzível a partir do sha.

    Raises:
        EvidenceError: git indisponível ou diretório fora de um repositório. Uma
            medição sem sha não é registrada — não existe sha "desconhecido".
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError(f"não foi possível ler o estado git em {repo}: {exc}") from exc
    if len(sha) != 40:
        raise EvidenceError(f"sha git inesperado: {sha!r}")
    return sha, bool(status)
