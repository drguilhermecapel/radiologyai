"""Interface de linha de comando."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from radiologyai import __version__, selftest

app = typer.Typer(
    name="radiologyai",
    help=(
        "RadiologyAI — plataforma de pesquisa em imagem radiológica.\n\n"
        "AVISO: software de pesquisa, não é dispositivo médico. "
        "Nenhum modelo validado clinicamente."
    ),
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Mostra a versão do pacote."""
    typer.echo(__version__)


@app.command("selftest")
def selftest_cmd() -> None:
    """Verifica o ambiente. Falha alto se o núcleo estiver incompleto."""
    try:
        result = selftest()
    except RuntimeError as exc:
        typer.secho(f"FALHOU: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command()
def modalities() -> None:
    """Lista as modalidades registradas e seu estado de implementação."""
    from radiologyai.modalities import available_modalities

    seen: dict[str, dict[str, Any]] = {}
    for plugin in available_modalities().values():
        seen[plugin.code] = plugin.describe()

    for code in sorted(seen):
        info = seen[code]
        status = "implementada" if info["implemented"] else "declarada, NÃO implementada"
        typer.echo(
            f"{code:<4} {str(info['dimensionality']):<5} "
            f"{','.join(info['accepted_modalities']):<10} {status}"
        )


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="Arquivo DICOM a inspecionar")],
) -> None:
    """Lê um DICOM, extrai metadados e aplica o gate de escopo da modalidade.

    Não executa modelo algum. Serve para verificar se um exame está dentro do
    uso pretendido antes de qualquer processamento.
    """
    from radiologyai.errors import RadiologyAIError
    from radiologyai.io.metadata import extract_metadata
    from radiologyai.io.reader import read_dicom
    from radiologyai.modalities import resolve

    try:
        ds = read_dicom(path, stop_before_pixels=True)
        metadata = extract_metadata(ds)
        typer.echo(metadata.model_dump_json(indent=2))

        plugin = resolve(metadata.modality)
        report = plugin.validate(metadata)
        if report.accepted:
            typer.secho(f"\nDENTRO DO ESCOPO ({plugin.code})", fg=typer.colors.GREEN)
        else:
            typer.secho(f"\nFORA DO ESCOPO ({plugin.code}):", fg=typer.colors.YELLOW)
            for reason in report.reasons:
                typer.echo(f"  - {reason}")
            raise typer.Exit(code=2)
    except RadiologyAIError as exc:
        typer.secho(f"ERRO: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def cards() -> None:
    """Lista os model cards registrados e se possuem desempenho medido."""
    from radiologyai.models import list_cards

    found = list_cards()
    if not found:
        typer.echo(
            "Nenhum model card registrado.\n"
            "Nenhum modelo foi treinado ou validado. Ver ROADMAP.md §5, Fase 1."
        )
        return
    for card in found:
        measured = "medido" if card.has_measured_performance else "SEM MEDIÇÃO"
        typer.echo(f"{card.card_id:<32} {card.modality:<4} {card.backend:<6} {measured}")


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
