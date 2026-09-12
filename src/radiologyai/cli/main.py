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


@app.command()
def manifest(
    data_root: Annotated[Path, typer.Argument(help="Diretório do NIH ChestX-ray14")],
    out: Annotated[Path, typer.Option(help="CSV de saída")] = Path(
        "datasets/manifests/nih_cxr14_test.csv"
    ),
    split: Annotated[str, typer.Option(help="test, train_val ou all")] = "test",
) -> None:
    """Constrói o manifest do NIH ChestX-ray14 a partir dos arquivos oficiais.

    O manifest é commitado; os pixels não. É o registro que permite a um
    terceiro reproduzir exatamente a mesma avaliação.
    """
    from radiologyai.data.nih_cxr14 import build_manifest
    from radiologyai.errors import RadiologyAIError

    try:
        m = build_manifest(data_root, split=split)
    except RadiologyAIError as exc:
        typer.secho(f"ERRO: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    m.to_csv(out)
    typer.echo(f"{len(m)} imagens · {len(m.patient_ids)} pacientes · split {m.split}")
    typer.echo(f"manifest_sha256: {m.sha256()}")
    for name in m.label_names:
        n = m.positives(name)
        typer.echo(f"  {name:<20} {n:>6} positivos ({n / len(m):.2%})")
    typer.secho(f"\nGravado: {out}", fg=typer.colors.GREEN)


@app.command()
def evaluate(
    card_id: Annotated[str, typer.Option("--card", help="card_id do model card")],
    data_root: Annotated[Path, typer.Option("--data-root", help="Raiz do dataset")],
    manifest_path: Annotated[
        Path | None, typer.Option("--manifest", help="CSV do manifest (gerado se omitido)")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Diretório de artefatos")] = Path(
        "artifacts/eval"
    ),
    seed: Annotated[int, typer.Option(help="Seed do bootstrap")] = 20260101,
    bootstrap: Annotated[int, typer.Option(help="Reamostragens do bootstrap")] = 2000,
    batch_size: Annotated[int, typer.Option(help="Tamanho do lote")] = 32,
    limit: Annotated[int, typer.Option(help="Limita o nº de imagens (só para teste)")] = 0,
    device: Annotated[str, typer.Option(help="cpu ou cuda")] = "cpu",
) -> None:
    """Mede o desempenho de um modelo num dataset e grava o artefato.

    Este é o único caminho pelo qual um número de desempenho pode entrar no
    repositório. Grava artifacts/eval/<run_id>/metrics.json com git_sha,
    weights_sha256, manifest_sha256, seed e versões de biblioteca.
    """
    from radiologyai.errors import RadiologyAIError
    from radiologyai.evaluation.baseline import run_baseline, summarize

    try:
        result = run_baseline(
            data_root,
            out,
            card_id=card_id,
            manifest_path=manifest_path,
            device=device,
            seed=seed,
            n_bootstrap=bootstrap,
            batch_size=batch_size,
            limit=limit,
            log=typer.echo,
        )
    except RadiologyAIError as exc:
        typer.secho(f"ERRO: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("")
    summarize(result, log=typer.echo)
    typer.secho(f"\nArtefato: {result.run_dir}", fg=typer.colors.GREEN)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
