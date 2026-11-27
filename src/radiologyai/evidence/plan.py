"""Plano de medição: a lista declarada de (modelo, dataset, rótulo) e o seu desfecho.

O plano (``evidence/measured/plan.yaml``) é escrito **antes** de medir e fica sob
controle de versão. Ele declara, por medição, o arquivo de predições, os dados de
treino do modelo (que decidem o vazamento), o padrão de referência, os pesos (cujo
hash é **lido** do arquivo, nunca digitado) e o manifesto contra o qual esse hash é
conferido. Conjuntos incompletos entram como ``pending`` com o motivo, e conjuntos
que não são avaliação de condição entram como ``excluded``: nada some do relato.

A execução produz, além dos registros, o razão (``ledger.yaml``) com o desfecho de
cada item do plano — aprovado, reprovado, não elegível, falha, pendente, excluído —
e se o registro aguarda a condição no catálogo.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from radiologyai.catalog.schema import Protocol, ReferenceStandard
from radiologyai.errors import EvidenceError, RadiologyAIError
from radiologyai.evidence.leakage import lineage
from radiologyai.evidence.measure import (
    MeasuredEvidence,
    MeasurementContext,
    measure_agreement,
    measure_binary,
    measure_detection,
)
from radiologyai.evidence.provenance import REPO_ROOT, sha256_file
from radiologyai.evidence.store import EVIDENCE_ROOT, measured_path, write_measured

PLAN_FORMAT = "radiologyai.evidence/plan-v1"
LEDGER_FORMAT = "radiologyai.evidence/ledger-v1"
PLAN_PATH = EVIDENCE_ROOT / "measured" / "plan.yaml"
LEDGER_PATH = EVIDENCE_ROOT / "measured" / "ledger.yaml"
MANIFESTS_DIR = REPO_ROOT / "src" / "radiologyai" / "analysis" / "manifests"

JobStatus = Literal["measure", "pending", "excluded"]
Outcome = Literal["aprovado", "reprovado", "nao_elegivel", "falha", "pendente", "excluido"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class PlanJob(_Frozen):
    """Um item do plano."""

    id: str
    status: JobStatus
    reason: str | None = None  # obrigatório para pending/excluded
    endpoint: Literal["discrimination", "agreement", "detection"] = "discrimination"
    model_id: str | None = None
    dataset_id: str | None = None  # chave em radiologyai.evidence.leakage.DATASETS
    label: str | None = None
    condition_id: str | None = None
    predictions: str | None = None  # relativo a preds_root, ou absoluto
    label_column: str = ""  # vazio = parquet por rótulo; senão, rótulo do multirrótulo
    score_scale: Literal["probability", "logit"] = "probability"
    reference_column: str | None = None  # agreement
    measured_column: str | None = None  # agreement
    # detection: regra fixa pré-especificada; a coluna de escore traz a decisão 0/1 já
    # tomada pela regra (nenhum ajuste nos dados de avaliação).
    detection_rule: str | None = None
    where: dict[str, str] = {}
    training_data: tuple[str, ...] = ()
    no_training_data_declared: bool = False
    # Datasets em que o modelo, o codificador, hiperparâmetros ou o pré-processamento
    # foram ESCOLHIDOS comparando desempenho. Avaliar num deles é vazamento ("known").
    selection_data: tuple[str, ...] = ()
    internal_holdout: bool = False
    # Entrada do modelo derivada do padrão de referência (ex.: caixa = caixa da máscara
    # de referência + margem); descreve como. O registro não sustenta nível A.
    reference_derived_input: str | None = None
    reference_standard: ReferenceStandard | None = None
    population: str | None = None
    weights: dict[str, str] = {}  # nome -> caminho do arquivo (hash lido)
    manifest: str | None = None  # id do manifesto contra o qual os hashes são conferidos
    caveats: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> PlanJob:
        if self.status != "measure":
            if not (self.reason and self.reason.strip()):
                raise ValueError(f"{self.id}: item {self.status} exige 'reason'")
            return self
        missing = [
            name
            for name in ("model_id", "dataset_id", "label", "condition_id", "predictions")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"{self.id}: item 'measure' sem {missing}")
        if not self.weights:
            raise ValueError(f"{self.id}: item 'measure' sem 'weights' (arquivos a hashear)")
        if not self.training_data and not self.no_training_data_declared:
            raise ValueError(
                f"{self.id}: declare training_data ou no_training_data_declared: true "
                "(o registro sairá com leakage 'possible')"
            )
        if self.endpoint == "agreement" and not (self.reference_column and self.measured_column):
            raise ValueError(f"{self.id}: agreement exige reference_column e measured_column")
        if self.endpoint != "agreement" and (self.reference_column or self.measured_column):
            raise ValueError(f"{self.id}: reference/measured_column só valem para agreement")
        if (self.endpoint == "detection") != bool(self.detection_rule):
            raise ValueError(f"{self.id}: detection_rule é obrigatória em detection e só nela")
        return self


class Plan(_Frozen):
    format: Literal["radiologyai.evidence/plan-v1"]
    protocol_id: str
    preds_root: str
    jobs: tuple[PlanJob, ...]

    @model_validator(mode="after")
    def _unique(self) -> Plan:
        seen: set[str] = set()
        targets: set[tuple[str | None, ...]] = set()
        for job in self.jobs:
            if job.id in seen:
                raise ValueError(f"id de item repetido no plano: {job.id!r}")
            seen.add(job.id)
            if job.status == "measure":
                target = (job.dataset_id, job.model_id, job.label, job.endpoint)
                if target in targets:
                    raise ValueError(f"dois itens gravariam o mesmo registro: {target}")
                targets.add(target)
        return self


class LedgerEntry(_Frozen):
    """Desfecho de um item do plano."""

    job: str
    outcome: Outcome
    model_id: str | None = None
    dataset_id: str | None = None
    label: str | None = None
    condition_id: str | None = None
    endpoint: str | None = None
    awaiting_condition: bool = False  # condição ausente de conditions.yaml
    record: str | None = None  # caminho relativo a evidence/
    kind: str | None = None
    leakage: str | None = None
    n_total: int | None = None
    n_positive: int | None = None
    reasons: tuple[str, ...] = ()


class Ledger(_Frozen):
    format: Literal["radiologyai.evidence/ledger-v1"] = "radiologyai.evidence/ledger-v1"
    protocol_id: str
    plan: str
    entries: tuple[LedgerEntry, ...]


# --------------------------------------------------------------------------- #
def load_plan(path: Path = PLAN_PATH) -> Plan:
    """Lê e valida o plano.

    Raises:
        EvidenceError: arquivo ausente, YAML inválido, item inconsistente ou chave
            de dataset fora da tabela de linhagem.
    """
    import yaml

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceError(f"não foi possível ler o plano {path}: {exc}") from exc
    try:
        plan = Plan.model_validate(raw)
    except ValidationError as exc:
        raise EvidenceError(f"plano inválido em {path}: {exc}") from exc
    for job in plan.jobs:
        if job.status == "measure":
            assert job.dataset_id is not None  # garantido pelo validador
            lineage(job.dataset_id)
            for key in (*job.training_data, *job.selection_data):
                lineage(key)
    return plan


def manifest_hashes(manifest_id: str, manifests_dir: Path = MANIFESTS_DIR) -> dict[str, str]:
    """``{filename: sha256}`` dos pesos declarados num manifesto de modelo."""
    import yaml

    path = manifests_dir / f"{manifest_id}.yaml"
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceError(f"manifesto {manifest_id!r} ilegível em {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("id") != manifest_id:
        raise EvidenceError(f"{path}: manifesto sem id {manifest_id!r}")
    out: dict[str, str] = {}
    for item in raw.get("weights") or ():
        if isinstance(item, dict) and item.get("filename") and item.get("sha256"):
            out[str(item["filename"])] = str(item["sha256"])
    return out


def hash_weights(
    job: PlanJob,
    *,
    hasher: Callable[[Path], str] = sha256_file,
    manifests_dir: Path = MANIFESTS_DIR,
) -> dict[str, str]:
    """Hash de cada arquivo de pesos do item; conferido contra o manifesto, se houver.

    A chave gravada é o nome do arquivo (como no manifesto). Com ``manifest``, todo
    arquivo cujo nome conste do manifesto DEVE ter o mesmo hash, e ao menos um deve
    constar — senão a medição não corresponde ao artefato distribuído.

    Raises:
        EvidenceError: arquivo ausente, hash divergente do manifesto, nenhum arquivo
            do item no manifesto, ou dois arquivos com o mesmo nome.
    """
    out: dict[str, str] = {}
    for name, raw_path in job.weights.items():
        path = Path(raw_path)
        key = path.name
        if key in out:
            raise EvidenceError(f"{job.id}: dois pesos com o nome de arquivo {key!r} ({name})")
        out[key] = hasher(path)
    if job.manifest:
        declared = manifest_hashes(job.manifest, manifests_dir)
        matched = [k for k in out if k in declared]
        if not matched:
            raise EvidenceError(
                f"{job.id}: nenhum arquivo de pesos do item consta do manifesto "
                f"{job.manifest!r} ({sorted(declared)})"
            )
        for key in matched:
            if out[key] != declared[key]:
                raise EvidenceError(
                    f"{job.id}: sha256 de {key} ({out[key]}) difere do manifesto "
                    f"{job.manifest!r} ({declared[key]}); as predições não correspondem "
                    "ao artefato distribuído"
                )
    return out


def resolve_predictions(plan: Plan, job: PlanJob) -> Path:
    assert job.predictions is not None
    p = Path(job.predictions)
    return p if p.is_absolute() else Path(plan.preds_root) / p


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def execute_job(
    plan: Plan,
    job: PlanJob,
    protocol: Protocol,
    *,
    known_conditions: set[str],
    root: Path,
    git_sha: str,
    git_dirty: bool,
    command: str,
    hasher: Callable[[Path], str] = sha256_file,
    manifests_dir: Path = MANIFESTS_DIR,
    n_bootstrap: int | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> LedgerEntry:
    """Executa um item ``measure`` e grava o registro; devolve a linha do razão.

    Um registro cuja condição não existe em ``conditions.yaml`` é gravado em
    ``measured/_awaiting_condition/`` (fora da leitura de ``levels.json``) e marcado
    ``awaiting_condition`` — a medição não se perde nem promove condição inexistente.
    Erros de medição viram ``falha`` com o motivo; nada é gravado nesse caso.
    """
    from radiologyai.evidence.predictions import read_continuous, read_label

    base: dict[str, Any] = {
        "job": job.id,
        "model_id": job.model_id,
        "dataset_id": job.dataset_id,
        "label": job.label,
        "condition_id": job.condition_id,
        "endpoint": job.endpoint,
    }
    if job.status != "measure":
        outcome: Outcome = "pendente" if job.status == "pending" else "excluido"
        return LedgerEntry(**base, outcome=outcome, reasons=(job.reason or "",))
    model_id, dataset_id, label, condition_id = _identity(job)
    awaiting = condition_id not in known_conditions
    target = measured_path(root, dataset_id, model_id, label, awaiting=awaiting)
    try:
        weights = hash_weights(job, hasher=hasher, manifests_dir=manifests_dir)
        path = resolve_predictions(plan, job)
        common_inputs: dict[str, Any] = {
            "plan_job": job.id,
            "predictions_file": _relative(path, Path(plan.preds_root)),
        }

        def context(dataset_sha256: str) -> MeasurementContext:
            return MeasurementContext(
                model_id=model_id,
                label=label,
                condition_id=condition_id,
                dataset_id=dataset_id,
                training_data=tuple(job.training_data),
                selection_data=tuple(job.selection_data),
                internal_holdout=job.internal_holdout,
                reference_derived_input=job.reference_derived_input,
                reference_standard=job.reference_standard,
                population=job.population,
                weights_sha256=weights,
                git_sha=git_sha,
                git_dirty=git_dirty,
                command=command,
                dataset_sha256=dataset_sha256,
                created_at=now(),
                caveats=tuple(job.caveats),
            )

        evidence: MeasuredEvidence
        if job.endpoint == "discrimination":
            preds = read_label(path, job.label_column)
            score_column = f"y_score__{job.label_column}" if job.label_column else "y_score"
            evidence = measure_binary(
                context(preds.dataset_sha256),
                protocol,
                patient_ids=preds.patient_ids,
                y_true=preds.y_true,
                y_score=preds.y_score,
                score_scale=job.score_scale,
                subgroup_columns=preds.subgroups,
                n_bootstrap=n_bootstrap,
                inputs={
                    **common_inputs,
                    "predictions_sha256": preds.predictions_sha256,
                    "n_missing_label_excluded": preds.n_missing_label,
                    "score_column": score_column,
                },
                evidence_ref=f"{_relative(target, root.parent)}#report",
            )
        elif job.endpoint == "detection":
            preds = read_label(path, job.label_column)
            decisions = {float(v) for v in preds.y_score}
            if not decisions <= {0.0, 1.0}:
                raise EvidenceError(
                    f"{job.id}: detection exige decisão 0/1 na coluna de escore; "
                    f"valores {sorted(decisions)[:5]}"
                )
            judged_det = (
                protocol
                if n_bootstrap is None
                else protocol.model_copy(update={"bootstrap_iterations": n_bootstrap})
            )
            evidence = measure_detection(
                context(preds.dataset_sha256),
                judged_det,
                rule=job.detection_rule or "",
                patient_ids=preds.patient_ids,
                y_true=preds.y_true,
                y_pred=preds.y_score.astype("int64"),
                inputs={
                    **common_inputs,
                    "predictions_sha256": preds.predictions_sha256,
                    "n_missing_label_excluded": preds.n_missing_label,
                    "detection_rule": job.detection_rule,
                },
            )
        else:
            if not (job.reference_column and job.measured_column):  # validador garante
                raise EvidenceError(f"{job.id}: agreement sem colunas")
            cont = read_continuous(
                path,
                reference_column=job.reference_column,
                measured_column=job.measured_column,
                where=job.where,
            )
            judged = (
                protocol
                if n_bootstrap is None
                else protocol.model_copy(update={"bootstrap_iterations": n_bootstrap})
            )
            evidence = measure_agreement(
                context(cont.dataset_sha256),
                judged,
                patient_ids=cont.patient_ids,
                measured=cont.measured,
                reference=cont.reference,
                abstained_patient_ids=cont.abstained_patient_ids,
                inputs={
                    **common_inputs,
                    "predictions_sha256": cont.predictions_sha256,
                    "reference_column": job.reference_column,
                    "measured_column": job.measured_column,
                    "where": dict(job.where),
                    "n_abstained": cont.n_abstained,
                    "n_rows_outside_filter": cont.n_filtered_out,
                },
            )
    except RadiologyAIError as exc:
        return LedgerEntry(
            **base, outcome="falha", awaiting_condition=awaiting, reasons=(str(exc),)
        )
    written = write_measured(evidence, root, awaiting=awaiting)
    rec = evidence.record
    verdict: Outcome
    if rec.passes_protocol is None:
        verdict = "nao_elegivel"
    else:
        verdict = "aprovado" if rec.passes_protocol else "reprovado"
    return LedgerEntry(
        **base,
        outcome=verdict,
        awaiting_condition=awaiting,
        record=_relative(written, root),
        kind=rec.kind,
        leakage=rec.leakage,
        n_total=rec.n_total,
        n_positive=rec.n_positive,
        reasons=rec.failure_reasons,
    )


def _identity(job: PlanJob) -> tuple[str, str, str, str]:
    """``(model_id, dataset_id, label, condition_id)`` de um item ``measure``."""
    m, d, lab, c = job.model_id, job.dataset_id, job.label, job.condition_id
    if not (m and d and lab and c):
        raise EvidenceError(f"{job.id}: item 'measure' sem identidade completa")
    return m, d, lab, c


def render_ledger(ledger: Ledger) -> str:
    """YAML determinístico do razão (sem horário: função do plano e dos registros)."""
    import yaml

    data = ledger.model_dump(mode="json")
    header = (
        "# Gerado por scripts/evidence_run.py --plan; não editar à mão.\n"
        "# Desfecho de cada item de evidence/measured/plan.yaml. Itens 'pendente' e\n"
        "# 'excluido' não têm registro: nenhuma métrica é extrapolada de conjunto incompleto.\n"
    )
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def write_ledger(ledger: Ledger, path: Path = LEDGER_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ledger(ledger), encoding="utf-8")
    return path


__all__ = [
    "LEDGER_PATH",
    "PLAN_PATH",
    "Ledger",
    "LedgerEntry",
    "Plan",
    "PlanJob",
    "execute_job",
    "hash_weights",
    "load_plan",
    "manifest_hashes",
    "render_ledger",
    "write_ledger",
]
