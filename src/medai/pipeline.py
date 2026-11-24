"""The single place where model output becomes a result.

FROZEN CONTRACT (``analyze`` signature and decision semantics).

``analyze`` runs the routed models in dependency order and applies, in this
order: applicability gates -> prediction -> calibration -> operating points
-> decision -> evidence level and visibility -> explanations. Adapters never
do any of these steps themselves.

A model that is unavailable, not applicable or fails is *recorded* as such in
``AnalysisResult.runs``. Nothing is ever substituted for its output.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from medai.catalog.catalog import Catalog
from medai.core.artifacts import ArtifactStore
from medai.core.errors import ExamTypeUnknown, ManifestError, ModelUnavailable
from medai.core.policy import ManifestOnlyPolicy, SafetyPolicy
from medai.core.types import (
    AbstentionReason,
    AnalysisResult,
    CodedConcept,
    Decision,
    EvidenceLevel,
    ExamInputs,
    ExamType,
    Explanation,
    Finding,
    Measurement,
    ModelProvenance,
    ModelRun,
    OperatingPoint,
    QAReport,
    ScoreKind,
    Segmentation,
)
from medai.models.base import AdapterOutput, ModelAdapter, ModelInputs, RawMeasurement
from medai.models.manifest import LabelSpec, ModelManifest
from medai.models.registry import Registry
from medai.routing import RoutingTable, load_routing, role_of

log = logging.getLogger("medai.pipeline")

PRIVATE_SCHEME = "99MEDAI"
_BOUNDED_SCALES = frozenset({"sigmoid", "softmax", "calibrated"})


# --------------------------------------------------------------------------- #
# Decision logic (pure functions; exhaustively unit-tested)
# --------------------------------------------------------------------------- #
def _passes(score: float, point: OperatingPoint) -> bool:
    return score >= point.threshold if point.direction == "ge" else score <= point.threshold


def decide(
    scores: dict[str, float], points: Sequence[OperatingPoint]
) -> tuple[Decision, list[OperatingPoint]]:
    """Turn scores into a decision using only sourced operating points.

    ``scores`` maps a scale name ("sigmoid", "calibrated", "measurement", ...)
    to the value on that scale. Points whose scale is not available are ignored.

    * rule-out and rule-in points both present -> three-way decision; the zone
      between them is ``"indeterminate"`` (the low-confidence band).
    * a single usable point -> two-way decision.
    * no usable point -> ``"score_only"``: the score is shown without a call.
    """
    usable = [p for p in points if p.on in scores]
    if not usable:
        return "score_only", []
    rule_out = next((p for p in usable if p.role == "rule_out"), None)
    rule_in = next((p for p in usable if p.role == "rule_in"), None)
    single = next((p for p in usable if p.role in ("balanced", "guideline")), None)

    if rule_out is not None and rule_in is not None:
        if _passes(scores[rule_in.on], rule_in):
            return "positive", [rule_out, rule_in]
        if not _passes(scores[rule_out.on], rule_out):
            return "negative", [rule_out, rule_in]
        return "indeterminate", [rule_out, rule_in]
    point = single or rule_in or rule_out
    assert point is not None
    return ("positive" if _passes(scores[point.on], point) else "negative"), [point]


def op_normalize(scores: dict[str, float], points: Sequence[OperatingPoint]) -> float | None:
    """Rescale a bounded score so that 0.5 sits at the primary operating point."""
    if not points:
        return None
    primary = next((p for p in points if p.role in ("balanced", "guideline")), None)
    primary = primary or next((p for p in points if p.role == "rule_in"), points[0])
    if primary.on not in _BOUNDED_SCALES or primary.on not in scores:
        return None
    score, threshold = scores[primary.on], primary.threshold
    if primary.direction == "le":
        score, threshold = 1.0 - score, 1.0 - threshold
    if not 0.0 < threshold < 1.0:
        return None
    score = min(1.0, max(0.0, score))
    if score < threshold:
        return score / (2.0 * threshold)
    return 1.0 - (1.0 - score) / (2.0 * (1.0 - threshold))


def private_code(key: str) -> CodedConcept:
    return CodedConcept(scheme=PRIVATE_SCHEME, value=key, meaning=key.replace("_", " "))


# --------------------------------------------------------------------------- #
# Input selection
# --------------------------------------------------------------------------- #
def candidate_inputs(
    manifest: ModelManifest, exam: ExamInputs, upstream: dict[str, AdapterOutput]
) -> list[tuple[ModelInputs, list[int]]]:
    """Possible inputs for a model, each with the map local image index -> exam index."""
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
# Conversion of adapter output
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
                f"{manifest.id}:{spec.key} maps to unknown condition {spec.condition_id!r}; "
                "treated as exploratory"
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
        """Convert one adapter output; returns the findings it produced."""
        prov_idx = len(self.provenance)
        self.provenance.append(provenance)
        declared = {spec.key: spec for spec in manifest.labels}
        produced: list[Finding] = []

        def exam_idx(local: int) -> int:
            return index_map[local] if 0 <= local < len(index_map) else index_map[0]

        seg_ids: dict[str, str] = {}
        for labelmap in output.labelmaps:
            seg_id = f"{manifest.id}:{labelmap.name}"
            seg_ids[labelmap.name] = seg_id
            self.segmentations.append(
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
            if raw.value is None and raw.abstention is None:
                raise ManifestError(
                    f"{manifest.id}: measurement {raw.key!r} has neither a value nor an abstention reason"
                )
            by_measure[raw.key] = raw
            self.measurements.append(
                Measurement(
                    name=raw.key,
                    code=private_code(raw.key),
                    value=raw.value,
                    unit=raw.unit,
                    method=raw.method,
                    derived_from=[seg_ids.get(name, name) for name in raw.derived_from],
                    geometry=raw.geometry,
                    preconditions=raw.preconditions,
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
                    f"{manifest.id}: adapter emitted score for undeclared binary label {score.label!r}"
                )
            if not np.isfinite(score.value):
                raise ManifestError(f"{manifest.id}: non-finite score for {score.label!r}")
            produced.append(
                self._scored_finding(manifest, spec, float(score.value), score.kind, prov_idx)
            )

        for label, classes in output.class_scores.items():
            spec = declared.get(label)
            if spec is None or spec.kind != "categorical":
                raise ManifestError(
                    f"{manifest.id}: adapter emitted classes for undeclared categorical label {label!r}"
                )
            if not classes or set(classes) != set(spec.classes or []):
                raise ManifestError(
                    f"{manifest.id}: class set for {label!r} differs from the manifest"
                )
            best = max(classes, key=lambda name: classes[name])
            level = self._level(manifest, spec)
            produced.append(
                Finding(
                    condition_id=spec.condition_id,
                    label=spec.key,
                    code=spec.code,
                    kind="categorical",
                    raw_score=float(classes[best]),
                    score_kind="softmax",
                    class_scores={k: float(v) for k, v in classes.items()},
                    category=best,
                    decision="category",
                    evidence_level=level,
                    visibility="research" if level is EvidenceLevel.D else "main",
                    provenance_idx=prov_idx,
                )
            )

        for spec in manifest.labels:
            if spec.kind != "derived" or spec.measurement not in by_measure:
                continue
            raw = by_measure[spec.measurement]
            level = self._level(manifest, spec)
            visibility = "research" if level is EvidenceLevel.D else "main"
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
                        visibility=visibility,
                        measurement_ref=raw.key,
                        image_idx=exam_idx(raw.image_idx),
                        provenance_idx=prov_idx,
                    )
                )
                continue
            finding = self._scored_finding(
                manifest, spec, float(raw.value), "measurement", prov_idx
            )
            finding.kind = "derived"
            finding.measurement_ref = raw.key
            finding.image_idx = exam_idx(raw.image_idx)
            produced.append(Finding.model_validate(finding.model_dump()))

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
            operating_points=used,
            decision=decision,
            evidence_level=level,
            visibility="research" if level is EvidenceLevel.D else "main",
            measurement_ref=spec.measurement if kind == "measurement" else None,
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
        except Exception as exc:
            self.warnings.append(f"{manifest.id}: explanation failed ({type(exc).__name__}: {exc})")
            return
        by_label = {f.label: f for f in wanted}
        prov_idx = wanted[0].provenance_idx
        for item in maps:
            finding = by_label.get(item.label)
            if finding is None or item.heatmap.ndim != 2:
                continue
            heat = np.clip(np.nan_to_num(item.heatmap, nan=0.0), 0.0, 1.0)
            ref = self.store.put_png_gray((heat * 255.0).round().astype(np.uint8))
            local = item.image_idx if 0 <= item.image_idx < len(index_map) else 0
            explanation = Explanation(
                id=f"{manifest.id}:{item.label}:{item.method}",
                method=item.method,
                target_label=item.label,
                map=ref,
                placement=item.placement,
                image_idx=index_map[local],
                provenance_idx=prov_idx,
            )
            self.explanations.append(explanation)
            finding.explanation_ids.append(explanation.id)


# --------------------------------------------------------------------------- #
# Entry point
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
    """Analyse one exam with the routed (or explicitly requested) models."""
    if exam.exam_type is ExamType.UNKNOWN:
        raise ExamTypeUnknown("exam type is unknown; a human must choose it before analysis")
    policy = policy or ManifestOnlyPolicy()
    if model_ids is None:
        table = routing or load_routing()
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
        missing = [dep for dep in manifest.depends_on if dep not in outputs]
        if missing:
            runs.append(
                ModelRun(
                    model_id=model_id,
                    status="abstained",
                    qa=QAReport.reject(AbstentionReason.UPSTREAM),
                    error=f"upstream model(s) produced no output: {', '.join(missing)}",
                )
            )
            continue
        try:
            adapter = registry.load(model_id)
        except ModelUnavailable as exc:
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
        except Exception as exc:
            log.exception("model %s failed", model_id)
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
        collector.warnings.append(f"no model is routed for exam type {exam.exam_type.value!r}")

    return AnalysisResult(
        analysis_id=uuid.uuid4(),
        created_at=datetime.now(UTC),
        exam_type=exam.exam_type,
        input_sha256=[image.sha256 for image in exam.images],
        status=status,
        provenance=collector.provenance,
        runs=runs,
        findings=collector.findings,
        segmentations=collector.segmentations,
        measurements=collector.measurements,
        explanations=collector.explanations,
        warnings=collector.warnings,
    )
