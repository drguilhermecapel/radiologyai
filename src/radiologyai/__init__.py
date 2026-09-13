"""RadiologyAI — plataforma de pesquisa em interpretação de imagens radiológicas.

AVISO: software de pesquisa. Não é dispositivo médico. Nenhum modelo validado
clinicamente. Ver HONEST_STATUS.md na raiz do repositório.

Este módulo é deliberadamente vazio de efeitos colaterais e não importa nenhuma
dependência pesada. `import radiologyai` deve custar poucos milissegundos e NÃO
deve trazer torch, SimpleITK nem numpy para a memória (ROADMAP §8, portão da Fase 1).
"""

from __future__ import annotations

__version__ = "2.0.0.dev0"

__all__ = ["__version__", "selftest"]


def selftest() -> dict[str, object]:
    """Verifica que o ambiente satisfaz o núcleo. Falha alto, nunca em silêncio.

    Diferente do sistema legado — que envolvia cada import em
    ``except ImportError: self.x = None`` e subia com todos os subsistemas
    desativados sem avisar — esta função levanta exceção ao primeiro problema.
    """
    import importlib.util
    import sys

    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError(f"Python 3.11 a 3.13 é obrigatório; encontrado {sys.version.split()[0]}")

    required = ["numpy", "pydicom", "pydantic", "yaml"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f"Dependências obrigatórias ausentes: {', '.join(missing)}")

    optional = {}
    for name in ["SimpleITK", "sklearn", "scipy", "torch", "fastapi"]:
        optional[name] = importlib.util.find_spec(name) is not None

    return {"version": __version__, "python": sys.version.split()[0], "optional": optional}
