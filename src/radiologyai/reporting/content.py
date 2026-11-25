"""Conteúdo canônico do laudo preliminar e os hashes que a assinatura amarra.

Regras:

* Só achados de visibilidade ``main`` (níveis A-C) entram no laudo. Achados
  exploratórios (nível D) NUNCA entram, mesmo que a análise tenha rodado em
  modo pesquisa; o laudo informa apenas quantos foram omitidos.
* Cada achado traz o nível, o significado do nível, o ponto de operação com
  origem (quando houve decisão) e a revisão do médico (aceito / rejeitado /
  emendado) — ou ``não revisado``.
* Nenhuma frase afirma normalidade. Sem achados A-C, o laudo diz exatamente
  isso e que ausência de achado sugerido não significa exame normal.
* ``report_sha256`` é o sha256 do JSON canônico do conteúdo; a assinatura
  cobre ``report_sha256`` + ``result_sha256`` + signatário + horário.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from radiologyai.analysis.types import DISCLAIMER_PT

if TYPE_CHECKING:
    from radiologyai.analysis.types import AnalysisResult, Finding
    from radiologyai.catalog.catalog import Catalog

__all__ = [
    "HEADER_PT",
    "LEVEL_MEANING_PT",
    "NO_FINDINGS_PT",
    "ReportContent",
    "ReportFinding",
    "ReportRun",
    "ReviewInfo",
    "build_report_content",
    "canonical_bytes",
    "sha256_hex",
    "signature_digest",
]

HEADER_PT = "Uso exclusivo em pesquisa — não é laudo diagnóstico"
NO_FINDINGS_PT = (
    "Nenhum achado com nível de evidência A, B ou C nesta análise. A ausência de achado "
    "sugerido NÃO significa exame normal; o sistema não afirma normalidade."
)
LEVEL_MEANING_PT: dict[str, str] = {
    "A": "Nível A — validado externamente por medição deste projeto, sob protocolo pré-registrado.",
    "B": "Nível B — validação publicada revisada por pares ou limiar de diretriz, com "
    "citação conferida na fonte.",
    "C": "Nível C — validado internamente por medição deste projeto; sem validação externa.",
}
DECISION_PT: dict[str, str] = {
    "positive": "achado sugerido (acima do ponto de inclusão)",
    "negative": "abaixo do ponto de exclusão",
    "indeterminate": "indeterminado (banda de baixa confiança)",
    "category": "categoria",
    "score_only": "somente escore (sem ponto de operação com origem)",
    "abstained": "abstenção",
}
ACTION_PT: dict[str, str] = {
    "accept": "aceito pelo médico",
    "reject": "rejeitado pelo médico",
    "amend": "emendado pelo médico",
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReviewInfo(_Frozen):
    action: Literal["accept", "reject", "amend"]
    amended_decision: str | None = None
    amended_text: str | None = None
    comment: str | None = None
    reviewer_username: str
    reviewer_name: str
    reviewed_at: str


class ReportFinding(_Frozen):
    idx: int
    condition_id: str
    condition_name: str
    code_scheme: str
    code_value: str
    code_meaning: str
    label: str
    model_id: str
    model_version: str
    evidence_level: Literal["A", "B", "C"]
    decision: str
    decision_pt: str
    raw_score: float | None
    score_kind: str
    op_normalized: float | None
    probability: float | None
    category: str | None
    operating_point: str | None  # "nome (papel) limiar sobre escala — origem"
    abstention_reason: str | None
    measurement: str | None  # "valor unidade (método)"
    image_idx: int
    review: ReviewInfo | None


class ReportRun(_Frozen):
    model_id: str
    status: str
    detail: str | None


class ReportProvenance(_Frozen):
    model_id: str
    model_version: str
    weights_sha256: dict[str, str]
    manifest_sha256: str
    code_version: str
    license_weights: str
    commercial_use: str


class ReportContent(_Frozen):
    schema_version: Literal["1"] = "1"
    kind: Literal["preliminary_research_report"] = "preliminary_research_report"
    header: str = HEADER_PT
    disclaimer: str = DISCLAIMER_PT
    analysis_id: str
    study_id: str
    study_uid: str | None
    patient_pseudo_id: str | None
    exam_type: str
    analysis_created_at: str
    result_sha256: str
    input_sha256: tuple[str, ...]
    findings: tuple[ReportFinding, ...]
    level_meanings: dict[str, str]
    excluded_research_findings: int
    unreviewed_findings: int
    runs: tuple[ReportRun, ...]
    provenance: tuple[ReportProvenance, ...]
    warnings: tuple[str, ...]
    no_findings_statement: str | None


def canonical_bytes(data: Any) -> bytes:
    """JSON canônico (chaves ordenadas, separadores compactos, UTF-8)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def signature_digest(
    *, report_id: str, report_sha256: str, result_sha256: str, signer_id: str, signed_at: datetime
) -> str:
    """Hash que amarra a assinatura ao conteúdo exibido e ao resultado analisado."""
    return sha256_hex(
        canonical_bytes(
            {
                "report_id": report_id,
                "report_sha256": report_sha256,
                "result_sha256": result_sha256,
                "signer_id": signer_id,
                "signed_at": signed_at.isoformat(),
            }
        )
    )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4g}"


def _point(finding: Finding) -> str | None:
    if not finding.operating_points:
        return None
    parts = []
    for p in finding.operating_points:
        sign = ">=" if p.direction == "ge" else "<="
        parts.append(f"{p.name} ({p.role}): {p.on} {sign} {_fmt(p.threshold)} — origem: {p.source}")
    return "; ".join(parts)


def build_report_content(
    result: AnalysisResult,
    *,
    result_sha256: str,
    analysis_id: str,
    study_id: str,
    study_uid: str | None,
    patient_pseudo_id: str | None,
    reviews: dict[int, ReviewInfo],
    catalog: Catalog | None,
    study_warnings: tuple[str, ...] = (),
) -> ReportContent:
    """Monta o conteúdo do laudo. ``reviews`` mapeia índice do achado -> última revisão."""
    measurements = {m.name: m for m in result.measurements}
    findings: list[ReportFinding] = []
    excluded = 0
    for idx, finding in enumerate(result.findings):
        if finding.visibility != "main" or finding.evidence_level.value == "D":
            excluded += 1
            continue
        prov = result.provenance[finding.provenance_idx]
        name = finding.code.meaning
        if catalog is not None and catalog.has(finding.condition_id):
            name = catalog.condition(finding.condition_id).name_pt
        measurement = None
        if finding.measurement_ref and finding.measurement_ref in measurements:
            m = measurements[finding.measurement_ref]
            value = "—" if m.value is None else f"{m.value:.4g}"
            measurement = f"{value} {m.unit} ({m.method})"
        level: Literal["A", "B", "C"]
        if finding.evidence_level.value == "A":
            level = "A"
        elif finding.evidence_level.value == "B":
            level = "B"
        else:
            level = "C"
        findings.append(
            ReportFinding(
                idx=idx,
                condition_id=finding.condition_id,
                condition_name=name,
                code_scheme=finding.code.scheme,
                code_value=finding.code.value,
                code_meaning=finding.code.meaning,
                label=finding.label,
                model_id=prov.model_id,
                model_version=prov.model_version,
                evidence_level=level,
                decision=finding.decision,
                decision_pt=DECISION_PT.get(finding.decision, finding.decision),
                raw_score=finding.raw_score,
                score_kind=finding.score_kind,
                op_normalized=finding.op_normalized,
                probability=finding.probability,
                category=finding.category,
                operating_point=_point(finding),
                abstention_reason=(
                    finding.abstention_reason.value if finding.abstention_reason else None
                ),
                measurement=measurement,
                image_idx=finding.image_idx,
                review=reviews.get(idx),
            )
        )
    runs = tuple(
        ReportRun(
            model_id=r.model_id,
            status=r.status,
            detail=r.error
            or (
                ", ".join(reason.value for reason in r.qa.reasons)
                if r.qa and r.qa.reasons
                else None
            ),
        )
        for r in result.runs
    )
    provenance = tuple(
        ReportProvenance(
            model_id=p.model_id,
            model_version=p.model_version,
            weights_sha256=dict(p.weights_sha256),
            manifest_sha256=p.manifest_sha256,
            code_version=p.code_version,
            license_weights=p.license_weights,
            commercial_use=p.commercial_use,
        )
        for p in result.provenance
    )
    return ReportContent(
        analysis_id=analysis_id,
        study_id=study_id,
        study_uid=study_uid,
        patient_pseudo_id=patient_pseudo_id,
        exam_type=result.exam_type.value,
        analysis_created_at=result.created_at.isoformat(),
        result_sha256=result_sha256,
        input_sha256=result.input_sha256,
        findings=tuple(findings),
        level_meanings=dict(LEVEL_MEANING_PT),
        excluded_research_findings=excluded,
        unreviewed_findings=sum(1 for f in findings if f.review is None),
        runs=runs,
        provenance=provenance,
        warnings=(*(f"ingestão: {w}" for w in study_warnings), *result.warnings),
        no_findings_statement=None if findings else NO_FINDINGS_PT,
    )
