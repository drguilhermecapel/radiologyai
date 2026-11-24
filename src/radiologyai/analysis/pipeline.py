"""O único lugar em que saída de modelo vira resultado.

CONTRATO (assinatura de ``analyze`` e semântica de decisão).

``analyze`` roda os modelos roteados em ordem de dependência e aplica, nesta
ordem: portões de aplicabilidade (QA) -> predição -> calibração -> pontos de
operação (política primeiro, manifesto como recurso) -> decisão -> nível de
evidência do catálogo e visibilidade -> explicações. Adaptadores nunca fazem
nenhum desses passos.

Modelo indisponível, planejado, não aplicável ou que falha é REGISTRADO como
tal em ``AnalysisResult.runs``. Nada é colocado no lugar da sua saída.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from radiologyai.analysis.base import (
    AdapterOutput,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
)
from radiologyai.analysis.policy import ManifestOnlyPolicy, SafetyPolicy
from radiologyai.analysis.routing import RoutingTable, load_routing, role_of
from radiologyai.analysis.types import (
    PRIVATE_SCHEME,
    AbstentionReason,
    AnalysisResult,
    CodedConcept,
    ExamInputs,
    Explanation,
    Finding,
    Measurement,
    ModelProvenance,
    ModelRun,
    QAReport,
    ScoreKind,
    Segmentation,
)
from radiologyai.calibration.decision import OperatingPoint, decide, op_normalize
from radiologyai.catalog.schema import EvidenceLevel, ExamType, visibility_for
from radiologyai.errors import ExamTypeUnknownError, ManifestError, ModelUnavailableError

if TYPE_CHECKING:
    from radiologyai.analysis.artifacts import ArtifactStore
    from radiologyai.analysis.manifest import LabelSpec, ModelManifest
    from radiologyai.analysis.registry import Registry
    from radiologyai.catalog.catalog import Catalog

__all__ = ["analyze", "candidate_inputs", "private_code"]

log = logging.getLogger("radiologyai.analysis.pipeline")


def private_code(key: str) -> CodedConcept:
    """Código no esquema privado 99MEDAI para chaves sem código oficial."""
    return CodedConcept(scheme=PRIVATE_SCHEME, value=key, meaning=key.replace("_", " "))


# --------------------------------------------------------------------------- #
# Seleção de entradas
# --------------------------------------------------------------------------- #
def candidate_inputs(
    manifest: ModelManifest, exam: ExamInputs, upstream: dict[str, AdapterOutput]
) -> list[tuple[ModelInputs, list[int]]]:
    """Entradas possíveis para um modelo, cada uma com o mapa índice local -> índice no exame.

    Modelos ``single`` recebem uma imagem por vez (a primeira aplicável é usada);
    modelos ``multi`` recebem todas as imagens com os papéis resolvidos por
    :func:`~radiologyai.analysis.routing.role_of`.
    """
    deps = {dep: upstream[dep] for dep in manifest.depends_on if dep in upstream}
    if manifest.inputs.mode == "multi":
        roles: dict[str, int] = {}
        for idx, image in enumerate(exam.images):
            role = role_of(image)
            if role is not None and role not in roles:
                roles[role] = idx
        inputs = ModelInputs(
            images=exam.images,
            exam_type=exam.exam_type,
            roles=roles,
            upstream=deps,
            prompts=exam.prompts,
        )
        return [(inputs, list(range(len(exam.images))))]
    return [
        (
            ModelInputs(
                images=(image,), exam_type=exam.exam_type, upstream=deps, prompts=exam.prompts
            ),
            [idx],
        )
        for idx, image in enumerate(exam.images)
    ]


# --------------------------------------------------------------------------- #
# Conversão da saída do adaptador
# --------------------------------------------------------------------------- #
class _Collector:
    def __init__(self, catalog: Catalog, policy: SafetyPolicy, store: ArtifactStore) -> None:
        self.catalog = catalog
        self.policy = policy
        self.store = store
        self.provenance: list[ModelProvenance] = []
        self.findings: list[Finding] = []
        self.segmentations: list[Segmentation] = []
        self.measurements: list[Measurement] = []
        self.explanations: list[Explanation] = []
        self.warnings: list[str] = []

    def _level(self, manifest: ModelManifest, spec: LabelSpec) -> EvidenceLevel:
        if not self.catalog.has(spec.condition_id):
            self.warnings.append(
                f"{manifest.id}:{spec.key} aponta para condição fora do catálogo "
                f"{spec.condition_id!r}; tratada como exploratória (nível D)"
            )
            return EvidenceLevel.D
        return self.catalog.level(spec.condition_id, manifest.id, spec.key)

    def _points(self, manifest: ModelManifest, label: str) -> list[OperatingPoint]:
        return self.policy.operating_points(manifest.id, label) or manifest.points_for(label)

    def add(
        self,
        manifest: ModelManifest,
        output: AdapterOutput,
        index_map: list[int],
        provenance: ModelProvenance,
    ) -> list[Finding]:
        """Converte uma saída de adaptador; devolve os achados produzidos.

        Raises:
            ManifestError: o adaptador emitiu algo que o manifesto não declara.
        """
        prov_idx = len(self.provenance)
        declared = {spec.key: spec for spec in manifest.labels}
        produced: list[Finding] = []
        segmentations: list[Segmentation] = []
        measurements: list[Measurement] = []

        def exam_idx(local: int) -> int:
            if not 0 <= local < len(index_map):
                raise ManifestError(f"{manifest.id}: image_idx {local} fora da entrada")
            return index_map[local]

        seg_ids: dict[str, str] = {}
        for labelmap in output.labelmaps:
            seg_id = f"{manifest.id}:{labelmap.name}"
            seg_ids[labelmap.name] = seg_id
            segmentations.append(
                Segmentation(
                    id=seg_id,
                    name=labelmap.name,
                    labels={i: private_code(k) for i, k in labelmap.labels.items()},
                    label_keys=dict(labelmap.labels),
                    mask=self.store.put_labelmap(labelmap.array),
                    image_idx=exam_idx(labelmap.image_idx),
                    provenance_idx=prov_idx,
                )
            )

        by_measure: dict[str, RawMeasurement] = {}
        for raw in output.measurements:
            by_measure[raw.key] = raw
            measurements.append(
                Measurement(
                    name=raw.key,
                    code=private_code(raw.key),
                    value=raw.value,
                    unit=raw.unit,
                    method=raw.method,
                    derived_from=tuple(seg_ids.get(name, name) for name in raw.derived_from),
                    geometry=tuple(raw.geometry),
                    preconditions=dict(raw.preconditions),
                    abstention_reason=raw.abstention if raw.value is None else None,
                    note=raw.note,
                    image_idx=exam_idx(raw.image_idx),
                    provenance_idx=prov_idx,
                )
            )

        for score in output.scores:
            spec = declared.get(score.label)
            if spec is None or spec.kind != "binary":
                raise ManifestError(
                    f"{manifest.id}: escore para rótulo binário não declarado {score.label!r}"
                )
            if score.kind != spec.score_kind:
                raise ManifestError(
                    f"{manifest.id}: {score.label!r} emitido como {score.kind!r}, "
                    f"manifesto declara {spec.score_kind!r}"
                )
            produced.append(
                self._scored_finding(
                    manifest, spec, float(score.value), score.kind, prov_idx, exam_idx(0)
                )
            )

        for label, classes in output.class_scores.items():
            spec = declared.get(label)
            if spec is None or spec.kind != "categorical":
                raise ManifestError(
                    f"{manifest.id}: classes para rótulo categórico não declarado {label!r}"
                )
            if not classes or set(classes) != set(spec.classes or ()):
                raise ManifestError(f"{manifest.id}: classes de {label!r} diferem do manifesto")
            best = max(classes, key=lambda name: classes[name])
            level = self._level(manifest, spec)
            produced.append(
                Finding(
                    condition_id=spec.condition_id,
                    label=spec.key,
                    code=spec.code,
                    kind="categorical",
                    raw_score=float(classes[best]),
                    score_kind=spec.score_kind,
                    class_scores={k: float(v) for k, v in classes.items()},
                    category=best,
                    decision="category",
                    evidence_level=level,
                    visibility=visibility_for(level),
                    image_idx=exam_idx(0),
                    provenance_idx=prov_idx,
                )
            )

        for spec in manifest.labels:
            if spec.kind != "derived" or spec.measurement not in by_measure:
                continue
            raw = by_measure[spec.measurement]
            level = self._level(manifest, spec)
            if raw.value is None:
                produced.append(
                    Finding(
                        condition_id=spec.condition_id,
                        label=spec.key,
                        code=spec.code,
                        kind="derived",
                        raw_score=None,
                        score_kind="measurement",
                        decision="abstained",
                        abstention_reason=raw.abstention,
                        evidence_level=level,
                        visibility=visibility_for(level),
                        measurement_ref=raw.key,
                        image_idx=exam_idx(raw.image_idx),
                        provenance_idx=prov_idx,
                    )
                )
                continue
            finding = self._scored_finding(
                manifest,
                spec,
                float(raw.value),
                "measurement",
                prov_idx,
                exam_idx(raw.image_idx),
            )
            produced.append(
                Finding.model_validate(
                    {**finding.model_dump(), "kind": "derived", "measurement_ref": raw.key}
                )
            )

        # Só publica no resultado depois que toda a saída foi validada.
        self.provenance.append(provenance)
        self.segmentations.extend(segmentations)
        self.measurements.extend(measurements)
        self.warnings.extend(f"{manifest.id}: {w}" for w in output.warnings)
        self.findings.extend(produced)
        return produced

    def _scored_finding(
        self,
        manifest: ModelManifest,
        spec: LabelSpec,
        value: float,
        kind: ScoreKind,
        prov_idx: int,
        image_idx: int,
    ) -> Finding:
        scores: dict[str, float] = {kind: value}
        probability: float | None = None
        calibration_id: str | None = None
        if kind != "measurement":
            calibrated = self.policy.calibrate(manifest.id, spec.key, value, kind)
            if calibrated is not None:
                probability, calibration_id = calibrated
                scores["calibrated"] = probability
        decision, used = decide(scores, self._points(manifest, spec.key))
        level = self._level(manifest, spec)
        return Finding(
            condition_id=spec.condition_id,
            label=spec.key,
            code=spec.code,
            kind="binary",
            raw_score=value,
            score_kind=kind,
            op_normalized=op_normalize(scores, used),
            probability=probability,
            calibration_id=calibration_id,
            operating_points=tuple(used),
            decision=decision,
            evidence_level=level,
            visibility=visibility_for(level),
            image_idx=image_idx,
            provenance_idx=prov_idx,
        )

    def add_explanations(
        self,
        manifest: ModelManifest,
        adapter: ModelAdapter,
        inputs: ModelInputs,
        index_map: list[int],
        produced: list[Finding],
        top_k: int,
    ) -> None:
        """Pede explicações ao modelo real para os achados positivos/indeterminados."""
        if "explain" not in manifest.capabilities or top_k <= 0:
            return
        wanted = [f for f in produced if f.decision in ("positive", "indeterminate")]
        wanted.sort(
            key=lambda f: f.op_normalized if f.op_normalized is not None else -1.0, reverse=True
        )
        wanted = wanted[:top_k]
        if not wanted:
            return
        try:
            maps = adapter.explain(inputs, [f.label for f in wanted])
        except Exception as exc:  # noqa: BLE001 - registrado; achado segue sem mapa
            self.warnings.append(f"{manifest.id}: explicação falhou ({type(exc).__name__}: {exc})")
            return
        import numpy as np

        by_label = {f.label: f for f in wanted}
        new_ids: dict[str, list[str]] = {}
        for item in maps:
            finding = by_label.get(item.label)
            if finding is None:
                self.warnings.append(f"{manifest.id}: explicação para rótulo não pedido descartada")
                continue
            if item.heatmap.ndim != 2 or not np.isfinite(item.heatmap).all():
                self.warnings.append(
                    f"{manifest.id}: explicação de {item.label!r} inválida (não 2D ou não finita)"
                )
                continue
            if not 0 <= item.image_idx < len(index_map):
                self.warnings.append(f"{manifest.id}: explicação com image_idx inválido")
                continue
            heat = np.clip(item.heatmap.astype(np.float32), 0.0, 1.0)
            ref = self.store.put_png_gray((heat * 255.0).round().astype(np.uint8))
            explanation = Explanation(
                id=f"{manifest.id}:{item.label}:{item.method}:{index_map[item.image_idx]}",
                method=item.method,
                target_label=item.label,
                map=ref,
                placement=item.placement,
                image_idx=index_map[item.image_idx],
                provenance_idx=finding.provenance_idx,
            )
            self.explanations.append(explanation)
            new_ids.setdefault(item.label, []).append(explanation.id)
        if not new_ids:
            return
        for i, finding in enumerate(self.findings):
            if finding.label in new_ids and finding in wanted:
                self.findings[i] = finding.model_copy(
                    update={"explanation_ids": (*finding.explanation_ids, *new_ids[finding.label])}
                )


# --------------------------------------------------------------------------- #
# Ponto de entrada
# --------------------------------------------------------------------------- #
def analyze(
    exam: ExamInputs,
    *,
    registry: Registry,
    catalog: Catalog,
    store: ArtifactStore,
    policy: SafetyPolicy | None = None,
    routing: RoutingTable | None = None,
    model_ids: Sequence[str] | None = None,
    include_research: bool = False,
    explain_top_k: int = 5,
) -> AnalysisResult:
    """Analisa um exame com os modelos roteados (ou pedidos explicitamente).

    Args:
        exam: imagens decodificadas e tipo de exame (decidido por metadados ou humano).
        registry: manifestos e pesos verificados.
        catalog: catálogo de condições; fonte do nível de evidência.
        store: repositório de artefatos (máscaras, mapas de explicação).
        policy: calibração e pontos ajustados; padrão: só pontos do manifesto.
        routing: tabela de roteamento; padrão: ``load_routing(registry)``.
        model_ids: modelos explícitos (ignora o roteamento); dependências entram sozinhas.
        include_research: roda também os modelos exploratórios (modo pesquisa).
        explain_top_k: máximo de achados explicados por modelo.

    Raises:
        ExamTypeUnknownError: tipo de exame desconhecido; um humano deve escolher.
    """
    if exam.exam_type is ExamType.UNKNOWN:
        raise ExamTypeUnknownError(
            "tipo de exame desconhecido; um humano deve escolhê-lo antes da análise"
        )
    policy = policy or ManifestOnlyPolicy()
    if model_ids is None:
        table = routing or load_routing(registry)
        requested = table.models_for(exam.exam_type, include_research=include_research)
        if exam.prompts:
            requested = [*requested, *table.interactive_for(exam.exam_type)]
    else:
        requested = list(model_ids)

    collector = _Collector(catalog, policy, store)
    runs: list[ModelRun] = []
    outputs: dict[str, AdapterOutput] = {}

    for model_id in registry.execution_order(requested):
        manifest = registry.manifest(model_id)
        if manifest.status != "available":
            runs.append(
                ModelRun(
                    model_id=model_id,
                    status="planned" if manifest.status == "planned" else "unavailable",
                    error=manifest.planned_reason or f"status {manifest.status!r}",
                )
            )
            continue
        missing = [dep for dep in manifest.depends_on if dep not in outputs]
        if missing:
            runs.append(
                ModelRun(
                    model_id=model_id,
                    status="abstained",
                    qa=QAReport.reject(AbstentionReason.UPSTREAM),
                    error=f"modelo(s) a montante sem saída: {', '.join(missing)}",
                )
            )
            continue
        try:
            adapter = registry.load(model_id)
        except ModelUnavailableError as exc:
            runs.append(ModelRun(model_id=model_id, status="unavailable", error=str(exc)))
            continue

        started = time.monotonic()
        chosen: tuple[ModelInputs, list[int]] | None = None
        last_qa: QAReport | None = None
        try:
            for inputs, index_map in candidate_inputs(manifest, exam, outputs):
                last_qa = adapter.applicability(inputs)
                if last_qa.applicable:
                    chosen = (inputs, index_map)
                    break
            if chosen is None:
                runs.append(
                    ModelRun(
                        model_id=model_id,
                        status="abstained",
                        qa=last_qa or QAReport.reject(AbstentionReason.MISSING_INPUT),
                        seconds=time.monotonic() - started,
                    )
                )
                continue
            inputs, index_map = chosen
            output = adapter.predict(inputs)
            produced = collector.add(manifest, output, index_map, registry.provenance(model_id))
            collector.add_explanations(
                manifest, adapter, inputs, index_map, produced, explain_top_k
            )
        except Exception as exc:  # noqa: BLE001 - falha registrada, nunca substituída
            log.exception("modelo %s falhou", model_id)
            runs.append(
                ModelRun(
                    model_id=model_id,
                    status="failed",
                    qa=last_qa,
                    error=f"{type(exc).__name__}: {exc}",
                    seconds=time.monotonic() - started,
                )
            )
            continue
        outputs[model_id] = output
        runs.append(
            ModelRun(
                model_id=model_id,
                status="completed",
                qa=last_qa,
                seconds=time.monotonic() - started,
                provenance_idx=len(collector.provenance) - 1,
            )
        )

    completed = sum(r.status == "completed" for r in runs)
    broken = sum(r.status in ("failed", "unavailable") for r in runs)
    status: Literal["completed", "partial", "abstained", "failed"]
    if completed and not broken:
        status = "completed"
    elif completed:
        status = "partial"
    elif broken:
        status = "failed"
    else:
        status = "abstained"
    if not runs:
        collector.warnings.append(
            f"nenhum modelo roteado para o tipo de exame {exam.exam_type.value!r}"
        )

    return AnalysisResult(
        analysis_id=uuid.uuid4(),
        created_at=datetime.now(UTC),
        exam_type=exam.exam_type,
        input_sha256=tuple(image.sha256 for image in exam.images),
        status=status,
        research_mode=include_research,
        provenance=tuple(collector.provenance),
        runs=tuple(runs),
        findings=tuple(collector.findings),
        segmentations=tuple(collector.segmentations),
        measurements=tuple(collector.measurements),
        explanations=tuple(collector.explanations),
        warnings=tuple(collector.warnings),
    )
