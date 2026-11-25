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
    dataset: Annotated[
        str, typer.Option("--dataset", help="nih-cxr14 (NLP) ou chexpert-valid (radiologistas)")
    ] = "nih-cxr14",
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
            dataset=dataset,
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


# --------------------------------------------------------------------------- #
# Plataforma: servidor, usuários, análise local
# --------------------------------------------------------------------------- #
users_app = typer.Typer(
    help="Usuários da plataforma. Não há credencial padrão: o primeiro admin nasce aqui.",
    no_args_is_help=True,
)
app.add_typer(users_app, name="users")


def _platform_db() -> Any:
    from radiologyai.persistence.config import get_platform_settings
    from radiologyai.persistence.db import Database

    settings = get_platform_settings()
    db = Database(settings.database_url)
    db.migrate()
    return db, settings


def _read_password(password_stdin: bool) -> str:
    if password_stdin:
        return sys.stdin.readline().rstrip("\n")
    value: str = typer.prompt("Senha", hide_input=True, confirmation_prompt=True)
    return value


@app.command()
def migrate() -> None:
    """Aplica as migrações Alembic pendentes ao banco de MEDAI_DATABASE_URL."""
    from radiologyai.errors import RadiologyAIError

    try:
        db, _ = _platform_db()
    except RadiologyAIError as exc:
        typer.secho(f"ERRO: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"banco na revisão {db.revision()}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface de escuta")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Porta")] = 8000,
    worker: Annotated[bool, typer.Option(help="Sobe o worker de análises em processo")] = True,
) -> None:
    """Sobe a API (uvicorn) com o worker em processo. Configuração por MEDAI_*."""
    import uvicorn

    from radiologyai.api import create_app

    typer.echo(
        "AVISO: software de pesquisa, não é dispositivo médico. "
        "Toda saída é achado sugerido para confirmação médica."
    )
    uvicorn.run(create_app(start_worker=worker), host=host, port=port, log_level="info")


@users_app.command("create")
def users_create(
    username: Annotated[str, typer.Option(help="Nome de usuário")],
    full_name: Annotated[str, typer.Option("--full-name", help="Nome completo")],
    role: Annotated[str, typer.Option(help="admin, physician, technician ou viewer")],
    professional_id: Annotated[
        str | None, typer.Option("--professional-id", help="Registro profissional (ex.: CRM)")
    ] = None,
    password_stdin: Annotated[
        bool, typer.Option("--password-stdin", help="Lê a senha da entrada padrão")
    ] = False,
) -> None:
    """Cria usuário (senha argon2id; nunca aceita senha por argumento de linha de comando)."""
    from radiologyai.errors import RadiologyAIError
    from radiologyai.persistence.users import create_user

    db, settings = _platform_db()
    password = _read_password(password_stdin)
    try:
        user = create_user(
            db,
            username=username,
            full_name=full_name,
            role=role,
            password=password,
            professional_id=professional_id,
            min_length=settings.password_min_length,
            actor_username="cli",
        )
    except (RadiologyAIError, ValueError) as exc:
        typer.secho(f"ERRO: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(f"usuário {user.username} ({user.role}) criado: {user.id}", fg=typer.colors.GREEN)


@users_app.command("list")
def users_list() -> None:
    """Lista usuários (sem hashes)."""
    from sqlalchemy import select

    from radiologyai.persistence.models import User

    db, _ = _platform_db()
    with db.session() as session:
        for u in session.execute(select(User).order_by(User.username)).scalars():
            state = "ativo" if u.active else "inativo"
            typer.echo(f"{u.username:<24} {u.role:<11} {state:<8} {u.full_name}")


@users_app.command("set-password")
def users_set_password(
    username: Annotated[str, typer.Argument(help="Usuário")],
    password_stdin: Annotated[bool, typer.Option("--password-stdin")] = False,
) -> None:
    """Troca a senha e revoga as sessões abertas do usuário."""
    from sqlalchemy import select

    from radiologyai.errors import RadiologyAIError
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import User, utcnow
    from radiologyai.persistence.security import check_password_policy, hash_password
    from radiologyai.persistence.users import revoke_user_sessions

    db, settings = _platform_db()
    password = _read_password(password_stdin)
    try:
        check_password_policy(password, settings.password_min_length)
    except RadiologyAIError as exc:
        typer.secho(f"ERRO: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    with db.session() as session:
        user = session.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if user is None:
            typer.secho("ERRO: usuário inexistente", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        user.password_hash = hash_password(password)
        user.updated_at = utcnow()
        user_id = user.id
    revoked = revoke_user_sessions(db, user_id)
    append_event(
        db,
        "user.password_reset",
        actor_username="cli",
        subject_type="user",
        subject_id=user_id,
        payload={"revoked_sessions": revoked},
    )
    typer.echo(f"senha alterada; {revoked} sessão(ões) revogada(s)")


@app.command("analyze")
def analyze_cmd(
    paths: Annotated[list[Path], typer.Argument(help="Arquivos DICOM/NIfTI/PNG ou diretório")],
    exam_type: Annotated[
        str | None, typer.Option("--exam-type", help="cxr, ct_chest, mg ... (padrão: inferido)")
    ] = None,
    model: Annotated[
        list[str] | None, typer.Option("--model", help="Modelo explícito (repetível)")
    ] = None,
    include_research: Annotated[
        bool, typer.Option("--include-research", help="Roda modelos exploratórios (nível D)")
    ] = False,
    modality: Annotated[str | None, typer.Option(help="Modalidade (NIfTI/PNG)")] = None,
    view_position: Annotated[
        str | None, typer.Option("--view-position", help="Preenche ViewPosition ausente")
    ] = None,
    age: Annotated[float | None, typer.Option(help="Preenche a idade ausente (anos)")] = None,
    body_part: Annotated[str | None, typer.Option("--body-part")] = None,
    store_dir: Annotated[Path, typer.Option("--store-dir", help="Repositório de artefatos")] = Path(
        "artifacts/store"
    ),
    out: Annotated[Path | None, typer.Option(help="Grava o AnalysisResult em JSON")] = None,
) -> None:
    """Analisa um exame LOCALMENTE com o pipeline (sem banco, sem servidor).

    Imprime o que rodou, o que se absteve e por quê, e os achados por nível.
    Nível D só aparece com --include-research.
    """
    import hashlib

    from radiologyai.analysis import imaging
    from radiologyai.analysis.artifacts import ArtifactStore
    from radiologyai.analysis.pipeline import analyze
    from radiologyai.analysis.registry import Registry
    from radiologyai.analysis.routing import infer_exam_type
    from radiologyai.analysis.types import ExamInputs, Modality
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.catalog.schema import ExamType
    from radiologyai.errors import RadiologyAIError
    from radiologyai.interop.ingest import apply_overrides

    overrides: dict[str, Any] = {}
    if view_position:
        overrides["view_position"] = view_position
    if age is not None:
        overrides["patient_age_years"] = age
    if body_part:
        overrides["body_part"] = body_part
    try:
        images = []
        for p in paths:
            kwargs: dict[str, Any] = {}
            if modality and p.name.lower().endswith((".nii", ".nii.gz", ".png", ".jpg", ".jpeg")):
                kwargs["modality"] = Modality(modality)
            images.append(apply_overrides(imaging.load_path(p, **kwargs), overrides))
        chosen = ExamType(exam_type) if exam_type else infer_exam_type(images)
        typer.echo(f"tipo de exame: {chosen.value} ({'declarado' if exam_type else 'inferido'})")
        result = analyze(
            ExamInputs(images=tuple(images), exam_type=chosen),
            registry=Registry(),
            catalog=Catalog.load(),
            store=ArtifactStore(store_dir),
            model_ids=model or None,
            include_research=include_research,
        )
    except RadiologyAIError as exc:
        typer.secho(f"ERRO: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    payload = result.model_dump_json()
    typer.echo(
        f"status: {result.status} · result_sha256 {hashlib.sha256(payload.encode()).hexdigest()}"
    )
    for run in result.runs:
        reasons = ",".join(r.value for r in run.qa.reasons) if run.qa and run.qa.reasons else ""
        detail = run.error or reasons
        secs = f"{run.seconds:.1f}s" if run.seconds is not None else "-"
        typer.echo(f"  {run.model_id:<36} {run.status:<11} {secs:>7} {detail}")
    shown = result.findings if include_research else tuple(result.main_findings())
    hidden = len(result.findings) - len(shown)
    typer.echo(f"achados exibidos: {len(shown)} · nível D omitidos: {hidden}")
    for f in shown:
        score = "—" if f.raw_score is None else f"{f.raw_score:.4g}"
        typer.echo(
            f"  [{f.evidence_level.value}] {f.condition_id:<36} {f.decision:<13} "
            f"{f.score_kind}={score}"
        )
    typer.echo(
        "Achados sugeridos para confirmação médica. Software de pesquisa; não é diagnóstico."
    )
    if out is not None:
        out.write_text(payload, encoding="utf-8")
        typer.secho(f"resultado gravado em {out}", fg=typer.colors.GREEN)


@app.command("openapi")
def openapi_cmd(
    out: Annotated[Path, typer.Option(help="Arquivo YAML de saída")] = Path(
        "docs/OPENAPI_SPEC.yaml"
    ),
) -> None:
    """Gera o contrato OpenAPI a partir da aplicação (nunca escrito à mão)."""
    from radiologyai.api.openapi import build_spec_app, export_openapi

    out.write_text(export_openapi(build_spec_app()), encoding="utf-8")
    typer.secho(f"OpenAPI gravado em {out}", fg=typer.colors.GREEN)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
