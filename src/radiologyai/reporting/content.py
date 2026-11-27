"""Conteúdo canônico do laudo preliminar e os hashes que a assinatura amarra.

Regras:

* Só achados de visibilidade ``main`` (níveis A-C) entram no laudo. Achados
  exploratórios (nível D) NUNCA entram, mesmo que a análise tenha rodado em
  modo pesquisa; o laudo informa apenas quantos foram omitidos.
* Cada achado traz o nível, o significado do nível, o ponto de operação com
  origem (quando houve decisão) e a revisão do médico (aceito / rejeitado /
  emendado) — ou ``não revisado``.
* O nível é RECONFERIDO no momento do laudo contra o catálogo corrente: vale
  o mais restritivo entre o gravado na análise e o atual; achado rebaixado a D
  sai do laudo e é contado.
* A decisão do médico (aceite, rejeição, emenda) é o valor primário do achado;
  a sugestão do modelo aparece separada. Nenhuma frase ou código afirma que a
  condição foi excluída: escore abaixo do limiar "não exclui a condição".
* Segmentações só entram (SEG, exportação) quando ligadas a um achado do laudo
  que o médico não rejeitou (:mod:`radiologyai.reporting.exposure`).
* Laudo só existe para análise em que ao menos um modelo concluiu; o
  conteúdo carrega o status do resultado e quantos modelos concluíram.
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
from radiologyai.catalog.schema import LEVEL_MEANING_PT as _CATALOG_LEVEL_MEANING_PT

if TYPE_CHECKING:
    from collections.abc import Callable

    from radiologyai.analysis.types import AnalysisResult, Finding
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.catalog.schema import EvidenceLevel

    #: (condition_id, model_id, label) -> nível efetivo corrente (catálogo + manifesto).
    CurrentLevel = Callable[[str, str, str], EvidenceLevel]

__all__ = [
    "EXPORT_DECISION",
    "PHYSICIAN_EXPORT",
    "export_codes",
    "HEADER_PT",
    "LEVEL_MEANING_PT",
    "NO_FINDINGS_PT",
    "ReportContent",
    "ReportFinding",
    "ReportRun",
    "ReportSegmentation",
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
# Fonte única do significado dos níveis: o catálogo (sem "validado", que exageraria o
# que um nível C sustenta). O laudo só mostra A–C; D nunca entra.
LEVEL_MEANING_PT: dict[str, str] = {k: _CATALOG_LEVEL_MEANING_PT[k] for k in ("A", "B", "C")}
PARTIAL_RUNS_PT = (
    "{broken} de {total} modelo(s) roteado(s) não concluíram (indisponível ou falha); "
    "o que eles avaliariam NÃO foi analisado."
)
DECISION_PT: dict[str, str] = {
    "positive": "achado sugerido (acima do ponto de inclusão)",
    "negative": "escore abaixo do limiar de exclusão (não exclui a condição)",
    "indeterminate": "indeterminado (banda de baixa confiança)",
    "category": "categoria",
    "score_only": "somente escore (sem ponto de operação com origem)",
    "abstained": "abstenção",
}
#: Código e texto exportados (SR/FHIR) por decisão do MODELO. Escore abaixo do
#: limiar nunca é codificado como "negativo": isso leria como exclusão da doença.
EXPORT_DECISION: dict[str, tuple[str, str]] = {
    "positive": ("decision.positive", "suggestion: above rule-in threshold"),
    "negative": ("decision.below_threshold", "below threshold; does not exclude condition"),
    "indeterminate": ("decision.indeterminate", "suggestion: indeterminate"),
    "category": ("decision.category", "suggestion: category"),
    "score_only": ("decision.score_only", "score only; no operating point"),
    "abstained": ("decision.abstained", "abstention"),
}
#: Código e texto exportados para a decisão FINAL do médico.
PHYSICIAN_EXPORT: dict[str, tuple[str, str]] = {
    "positive": ("physician-decision.positive", "physician: positive"),
    "negative": ("physician-decision.negative", "physician: negative"),
    "indeterminate": ("physician-decision.indeterminate", "physician: indeterminate"),
    "rejected": ("physician-decision.rejected", "physician rejected model suggestion"),
    "amended_text": ("physician-decision.amended-text", "physician amended (free text)"),
}
PHYSICIAN_DECISION_PT: dict[str, str] = {
    "positive": "positivo segundo o médico",
    "negative": "negativo segundo o médico",
    "indeterminate": "indeterminado segundo o médico",
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
    # Nível gravado na análise quando difere do efetivo (reconferido no laudo).
    level_at_analysis: Literal["A", "B", "C"] | None = None
    # Decisão final do médico: "rejected", "amended_text", uma decisão emendada,
    # ou a do modelo quando aceita; None = não revisado.
    physician_decision: str | None = None
    physician_decision_pt: str = "não revisado"
    measurement_only: bool = False  # achado derivado sem decisão: nomeia a MEDIDA


class ReportSegmentation(_Frozen):
    """Segmentação que pode acompanhar o laudo (SEG, exportação)."""

    idx: int  # índice no resultado completo
    name: str
    model_id: str
    finding_idx: tuple[int, ...]


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
    result_status: str | None = None
    completed_models: int | None = None
    runs_statement: str | None = None
    demoted_findings: int = 0
    segmentations: tuple[ReportSegmentation, ...] = ()


def canonical_bytes(data: Any) -> bytes:
    """JSON canônico (chaves ordenadas, separadores compactos, UTF-8)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def signature_digest(
    *,
    report_id: str,
    report_sha256: str,
    result_sha256: str,
    signer_id: str,
    signed_at: datetime,
    key: bytes,
) -> str:
    """HMAC-SHA256 que amarra a assinatura ao conteúdo exibido e ao resultado analisado.

    A chave é do servidor e não fica no banco: quem só pode escrever no banco
    não consegue forjar uma assinatura válida.
    """
    import hmac

    message = canonical_bytes(
        {
            "report_id": report_id,
            "report_sha256": report_sha256,
            "result_sha256": result_sha256,
            "signer_id": signer_id,
            "signed_at": signed_at.isoformat(),
        }
    )
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4g}"


def _point(finding: Finding) -> str | None:
    if not finding.operating_points:
        if finding.decision_note:
            return f"sem decisão (só escore): {finding.decision_note}"
        return None
    parts = []
    for p in finding.operating_points:
        sign = ">=" if p.direction == "ge" else "<="
        text = f"{p.name} ({p.role}): {p.on} {sign} {_fmt(p.threshold)} — origem: {p.source}"
        if p.evidence_ref:
            text += f" — registro: {p.evidence_ref}"
        parts.append(text)
    return "; ".join(parts)


_STRENGTH = {"A": 3, "B": 2, "C": 1, "D": 0}


def export_codes(finding: ReportFinding) -> tuple[tuple[str, str], tuple[str, str] | None]:
    """(código da sugestão do modelo, código da decisão do médico ou ``None``) para SR/FHIR.

    A aceitação herda a decisão do modelo, mas codificada como do médico; escore
    abaixo do limiar aceito continua "não exclui" (o médico não emendou).
    """
    model = EXPORT_DECISION.get(
        finding.decision, (f"decision.{finding.decision}", finding.decision)
    )
    final = finding.physician_decision
    if final is None:
        return model, None
    if finding.review is not None and finding.review.action == "accept":
        value, meaning = model
        return model, (f"physician-accepted.{value.split('.', 1)[1]}", f"accepted: {meaning}"[:64])
    return model, PHYSICIAN_EXPORT.get(final, (f"physician-decision.{final}", final))


def physician_decision(
    decision: str, decision_pt: str, review: ReviewInfo | None
) -> tuple[str | None, str]:
    """Decisão final do médico e seu texto (o valor primário nos documentos)."""
    if review is None:
        return None, "não revisado"
    if review.action == "reject":
        return "rejected", "rejeitado pelo médico (sugestão do modelo descartada)"
    if review.action == "amend":
        if review.amended_decision:
            text = PHYSICIAN_DECISION_PT.get(review.amended_decision, review.amended_decision)
            return review.amended_decision, f"emendado pelo médico: {text}"
        return "amended_text", f"emendado pelo médico (texto): {review.amended_text or ''}"
    return decision, f"aceito pelo médico: {decision_pt}"


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
    current_level: CurrentLevel | None = None,
) -> ReportContent:
    """Monta o conteúdo do laudo. ``reviews`` mapeia índice do achado -> última revisão.

    ``current_level`` dá o nível efetivo AGORA (catálogo corrente + restrição do
    manifesto); sem ele, vale o gravado na análise.
    """
    from radiologyai.reporting.exposure import exposed_artifacts

    measurements = {(m.provenance_idx, m.name): m for m in result.measurements}
    findings: list[ReportFinding] = []
    excluded = 0
    demoted = 0
    extra_warnings: list[str] = []
    for idx, finding in enumerate(result.findings):
        if finding.visibility != "main" or finding.evidence_level.value == "D":
            excluded += 1
            continue
        prov = result.provenance[finding.provenance_idx]
        stored = finding.evidence_level.value
        effective = stored
        if current_level is not None:
            now = current_level(finding.condition_id, prov.model_id, finding.label).value
            if _STRENGTH[now] < _STRENGTH[stored]:
                effective = now
        if effective == "D":
            excluded += 1
            demoted += 1
            extra_warnings.append(
                f"achado {idx} ({finding.condition_id}, {prov.model_id}:{finding.label}) era nível "
                f"{stored} na análise e hoje é exploratório (nível D): omitido do laudo"
            )
            continue
        if effective != stored:
            extra_warnings.append(
                f"achado {idx} ({finding.condition_id}) era nível {stored} na análise; "
                f"nível corrente {effective} (vale o mais restritivo)"
            )
        name = finding.code.meaning
        if catalog is not None and catalog.has(finding.condition_id):
            name = catalog.condition(finding.condition_id).name_pt
        measurement_only = finding.kind == "derived" and finding.decision in (
            "score_only",
            "abstained",
        )
        if measurement_only:
            # A medida não decide a condição: o laudo nomeia a MEDIDA, nunca a
            # condição (ex.: "FE reduzida") que o modelo não avaliou.
            name = f"Medida: {finding.code.meaning}"
        measurement = None
        key = (finding.provenance_idx, finding.measurement_ref)
        if finding.measurement_ref and key in measurements:
            m = measurements[key]
            value = "—" if m.value is None else f"{m.value:.4g}"
            measurement = f"{value} {m.unit} ({m.method})"
        level: Literal["A", "B", "C"] = (
            "A" if effective == "A" else "B" if effective == "B" else "C"
        )
        decision_pt = DECISION_PT.get(finding.decision, finding.decision)
        review = reviews.get(idx)
        final, final_pt = physician_decision(finding.decision, decision_pt, review)
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
                decision_pt=decision_pt,
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
                review=review,
                level_at_analysis=None if effective == stored else level_of(stored),
                physician_decision=final,
                physician_decision_pt=final_pt,
                measurement_only=measurement_only,
            )
        )
    kept = [f.idx for f in findings if f.physician_decision != "rejected"]
    exposure = exposed_artifacts(result, kept)
    segmentations = tuple(
        ReportSegmentation(
            idx=i,
            name=seg.name,
            model_id=result.provenance[seg.provenance_idx].model_id,
            finding_idx=tuple(
                j for j in kept if result.findings[j].provenance_idx == seg.provenance_idx
            ),
        )
        for i, seg in enumerate(result.segmentations)
        if i in exposure.segmentations
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
    completed = sum(r.status == "completed" for r in result.runs)
    broken = sum(r.status in ("failed", "unavailable") for r in result.runs)
    runs_statement = (
        PARTIAL_RUNS_PT.format(broken=broken, total=len(result.runs)) if broken else None
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
        warnings=(
            *(f"ingestão: {w}" for w in study_warnings),
            *result.warnings,
            *extra_warnings,
        ),
        no_findings_statement=None if findings else NO_FINDINGS_PT,
        result_status=result.status,
        completed_models=completed,
        runs_statement=runs_statement,
        demoted_findings=demoted,
        segmentations=segmentations,
    )


def level_of(value: str) -> Literal["A", "B", "C"] | None:
    """``"A"``/``"B"``/``"C"`` tipados; ``None`` para qualquer outro valor."""
    if value == "A":
        return "A"
    if value == "B":
        return "B"
    if value == "C":
        return "C"
    return None
