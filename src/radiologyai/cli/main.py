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
    manifest_path: Annotated[Path, typer.Option("--manifest", help="CSV do manifest")],
    data_root: Annotated[Path, typer.Option("--data-root", help="Raiz das imagens")],
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
    from radiologyai.data.manifest import Manifest
    from radiologyai.data.nih_cxr14 import LIMITATIONS, discover_image_dirs, find_image
    from radiologyai.errors import RadiologyAIError
    from radiologyai.evaluation import EvaluationConfig, check_leakage, run_evaluation
    from radiologyai.evaluation.predict_dataset import predict_manifest
    from radiologyai.models import get_card
    from radiologyai.models.backends import build_backend

    try:
        card = get_card(card_id)
        m = Manifest.from_csv(manifest_path, name="NIH ChestX-ray14", split="oficial test")
        if limit:
            m.rows = m.rows[:limit]
            typer.secho(
                f"AVISO: limitado a {limit} imagens — resultado NÃO é a medição completa.",
                fg=typer.colors.YELLOW,
            )

        leakage = check_leakage(card, m)
        if leakage == "in-distribution":
            typer.secho(
                "AVISO: este dataset consta entre os dados de treino do modelo. "
                "O resultado será in-distribution, NÃO validação externa.",
                fg=typer.colors.RED,
            )
        else:
            typer.secho(f"Vazamento: {leakage}", fg=typer.colors.GREEN)

        backend = build_backend(card)
        if device != "cpu" and hasattr(backend, "to"):
            backend.to(device)

        image_dirs = discover_image_dirs(data_root)
        if not image_dirs:
            typer.secho(f"ERRO: nenhum diretório de imagens em {data_root}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.echo(f"{len(image_dirs)} diretório(s) de imagem · {len(m)} imagens")

        with typer.progressbar(length=len(m), label="Inferência") as bar:  # type: ignore[var-annotated]
            state = {"done": 0}

            def tick(done: int, _total: int) -> None:
                bar.update(done - state["done"])
                state["done"] = done

            scores = predict_manifest(
                backend=backend,
                manifest=m,
                resolve_path=lambda image_id: find_image(image_id, image_dirs),
                batch_size=batch_size,
                progress=tick,
            )

        run_dir = run_evaluation(
            card=card,
            backend=backend,
            manifest=m,
            y_score=scores,
            output_dir=out,
            config=EvaluationConfig(seed=seed, n_bootstrap=bootstrap),
            dataset_limitations=LIMITATIONS,
        )
    except RadiologyAIError as exc:
        typer.secho(f"ERRO: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    import json

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    typer.secho(f"\nAUROC macro: {metrics['macro_auroc']}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"{metrics['n_labels_evaluated']} rótulos avaliados\n")
    for name, entry in sorted(metrics["per_label"].items(), key=lambda kv: -kv[1]["auroc"]):
        lo, hi = entry["auroc_ci95"]
        typer.echo(
            f"  {name:<24} {entry['auroc']:.3f}  IC95 [{lo:.3f}, {hi:.3f}]  n+={entry['n_pos']}"
        )
    typer.secho(f"\nArtefato: {run_dir}", fg=typer.colors.GREEN)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
