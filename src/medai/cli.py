"""Command line entry point (`medai`). Sub-commands are registered by their modules."""

from __future__ import annotations

import typer

from medai import __version__

app = typer.Typer(
    name="medai",
    help="MedAI Radiologia - research-only radiology AI platform.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
