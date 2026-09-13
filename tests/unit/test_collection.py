"""Portão estrutural: nenhum teste pode existir sem ser executado.

`norecursedirs` casa pelo nome base do diretório, em qualquer profundidade. Uma
entrada pensada para a raiz do repositório (`data/`, com os fantasmas
sintéticos) silenciou `tests/unit/data/` inteiro — 66 testes presentes no
repositório, verdes quando rodados à mão, e jamais executados pelo CI.

Um teste que não roda é pior que um teste ausente: ele cria a aparência de
cobertura. Este módulo torna essa classe de defeito detectável na própria
suíte, em vez de depender de alguém reparar numa contagem.
"""

from __future__ import annotations

import fnmatch
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests"


def _config() -> dict[str, object]:
    with (REPO / "pyproject.toml").open("rb") as fh:
        dados = tomllib.load(fh)
    cfg: dict[str, object] = dados["tool"]["pytest"]["ini_options"]
    return cfg


def _padroes_de_exclusao() -> list[str]:
    valor = _config().get("norecursedirs", [])
    assert isinstance(valor, list)
    return [str(p) for p in valor]


@pytest.mark.requirement("REQ-003")
def test_nenhum_diretorio_de_teste_casa_com_norecursedirs() -> None:
    padroes = _padroes_de_exclusao()
    silenciados = [
        d.relative_to(REPO)
        for d in TESTS.rglob("*")
        if d.is_dir()
        and d.name != "__pycache__"
        and any(fnmatch.fnmatch(d.name, padrao) for padrao in padroes)
    ]
    assert not silenciados, (
        f"estes diretórios de teste nunca são coletados: {silenciados}. "
        f"norecursedirs={padroes} casa pelo nome base, em qualquer profundidade."
    )


@pytest.mark.requirement("REQ-003")
def test_todo_arquivo_de_teste_foi_importado(request: pytest.FixtureRequest) -> None:
    """Cada ``test_*.py`` sob ``tests/`` tem de estar em ``sys.modules``.

    Se a coleção pulou um diretório, seus módulos nunca foram importados — e
    este teste, que roda dentro da mesma sessão, enxerga a lacuna.

    Só faz sentido na suíte completa: numa execução parcial os outros módulos
    legitimamente não foram importados. O CI roda `pytest` sem alvo, então o
    portão está vivo lá — que é onde ele precisa estar.
    """
    alvos = {Path(a.split("::")[0]).resolve() for a in request.config.args}
    if not alvos & {TESTS.resolve(), REPO.resolve()}:
        pytest.skip("execução parcial; este portão vale na suíte completa")

    no_disco = {p for p in TESTS.rglob("test_*.py") if "__pycache__" not in p.parts}
    importados = {
        Path(mod.__file__).resolve()
        for mod in list(sys.modules.values())
        if getattr(mod, "__file__", None) and str(getattr(mod, "__file__", "")).endswith(".py")
    }
    faltando = sorted(p.relative_to(REPO) for p in no_disco if p.resolve() not in importados)
    assert not faltando, f"arquivos de teste presentes mas nunca importados: {faltando}"
