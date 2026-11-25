"""PDF do laudo preliminar de pesquisa, em português (reportlab).

O PDF é uma renderização do :class:`~radiologyai.reporting.content.ReportContent`
mais o bloco de assinatura; não acrescenta nenhum número que não esteja no
conteúdo hasheado. Gerado com ``invariant=1``: o mesmo conteúdo dá os mesmos bytes.
"""

from __future__ import annotations

import importlib.util
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.errors import BackendUnavailableError
from radiologyai.reporting.content import ACTION_PT, HEADER_PT

if TYPE_CHECKING:
    from radiologyai.reporting.content import ReportContent

__all__ = ["SignOff", "render_pdf"]


@dataclass(frozen=True, slots=True)
class SignOff:
    """Bloco de assinatura; ``None`` nos campos quando ainda não assinado."""

    report_id: str
    report_sha256: str
    signer_name: str | None = None
    signer_username: str | None = None
    professional_id: str | None = None
    signed_at: str | None = None
    signature_sha256: str | None = None


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.4g}"


def render_pdf(content: ReportContent, sign: SignOff) -> bytes:
    """Renderiza o PDF.

    Raises:
        BackendUnavailableError: reportlab ausente (extra ``platform``).
    """
    if importlib.util.find_spec("reportlab") is None:
        raise BackendUnavailableError(
            "reportlab não instalado; instale o extra: pip install 'radiologyai[platform]'"
        )
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle("corpo", parent=styles["BodyText"], fontSize=9, leading=11.5)
    small = ParagraphStyle("pequeno", parent=body, fontSize=7.5, leading=9.5)
    h1 = ParagraphStyle("t1", parent=styles["Heading1"], fontSize=14, leading=17)
    h2 = ParagraphStyle("t2", parent=styles["Heading2"], fontSize=11.5, leading=14)
    h3 = ParagraphStyle("t3", parent=styles["Heading3"], fontSize=10, leading=12)

    def on_page(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#8a1c1c"))
        canvas.setFont("Helvetica-Bold", 9.5)
        canvas.drawCentredString(A4[0] / 2, A4[1] - 12 * mm, HEADER_PT)
        canvas.setFillColor(colors.grey)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(15 * mm, 10 * mm, f"report_sha256 {content_sha(sign)}")
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"página {doc.page}")
        canvas.restoreState()

    def content_sha(s: SignOff) -> str:
        return s.report_sha256

    story: list[Any] = [
        Paragraph("Relatório preliminar de pesquisa", h1),
        Paragraph(
            "Achados sugeridos para confirmação médica. Este documento não é laudo "
            "diagnóstico e não deve orientar conduta clínica.",
            body,
        ),
        Spacer(1, 4 * mm),
    ]
    ident = [
        ["Análise", content.analysis_id],
        ["Estudo (id interno)", content.study_id],
        ["Estudo (UID pseudonimizado)", content.study_uid or "—"],
        ["Paciente (pseudônimo)", content.patient_pseudo_id or "—"],
        ["Tipo de exame", content.exam_type],
        ["Análise executada em (UTC)", content.analysis_created_at],
        ["result_sha256", content.result_sha256],
        ["report_sha256", sign.report_sha256],
    ]
    table = Table(
        [[Paragraph(_esc(a), small), Paragraph(_esc(b), small)] for a, b in ident],
        colWidths=[48 * mm, 127 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story += [table, Spacer(1, 5 * mm), Paragraph("Achados por nível de evidência", h2)]

    if content.no_findings_statement:
        story.append(Paragraph(_esc(content.no_findings_statement), body))
    for level in ("A", "B", "C"):
        items = [f for f in content.findings if f.evidence_level == level]
        if not items:
            continue
        story.append(Paragraph(_esc(content.level_meanings[level]), h3))
        for f in items:
            review = (
                "não revisado"
                if f.review is None
                else f"{ACTION_PT[f.review.action]} ({f.review.reviewer_name}, "
                f"{f.review.reviewed_at})"
            )
            lines = [
                f"<b>{_esc(f.condition_name)}</b> — {_esc(f.decision_pt)}",
                f"Modelo: {_esc(f.model_id)} {_esc(f.model_version)}; rótulo {_esc(f.label)}; "
                f"código {_esc(f.code_scheme)}:{_esc(f.code_value)}",
                f"Escore bruto ({_esc(f.score_kind)}): {_num(f.raw_score)}; reescala no ponto de "
                f"operação: {_num(f.op_normalized)}; probabilidade calibrada: "
                f"{_num(f.probability)}",
            ]
            if f.category:
                lines.append(f"Categoria: {_esc(f.category)}")
            if f.measurement:
                lines.append(f"Medida: {_esc(f.measurement)}")
            if f.operating_point:
                lines.append(f"Ponto de operação: {_esc(f.operating_point)}")
            if f.abstention_reason:
                lines.append(f"Abstenção: {_esc(f.abstention_reason)}")
            lines.append(f"Revisão: {_esc(review)}")
            if f.review and f.review.amended_text:
                lines.append(f"Texto emendado: {_esc(f.review.amended_text)}")
            if f.review and f.review.comment:
                lines.append(f"Comentário: {_esc(f.review.comment)}")
            story.append(
                KeepTogether([Paragraph(line, body) for line in lines] + [Spacer(1, 2 * mm)])
            )
    if content.excluded_research_findings:
        story.append(
            Paragraph(
                f"{content.excluded_research_findings} achado(s) exploratório(s) (nível D) "
                "foram omitidos deste relatório por regra: nível D só aparece no modo pesquisa.",
                small,
            )
        )

    story += [Spacer(1, 4 * mm), Paragraph("Modelos executados", h2)]
    run_rows = [["Modelo", "Situação", "Detalhe"]] + [
        [Paragraph(_esc(r.model_id), small), r.status, Paragraph(_esc(r.detail or "—"), small)]
        for r in content.runs
    ]
    if len(run_rows) == 1:
        story.append(Paragraph("Nenhum modelo foi executado nesta análise.", body))
    else:
        runs = Table(run_rows, colWidths=[55 * mm, 22 * mm, 98 * mm], repeatRows=1)
        runs.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(runs)

    story += [Spacer(1, 4 * mm), Paragraph("Proveniência", h2)]
    for p in content.provenance:
        weights = ", ".join(f"{k}={v}" for k, v in sorted(p.weights_sha256.items())) or "—"
        story.append(
            Paragraph(
                f"<b>{_esc(p.model_id)}</b> {_esc(p.model_version)} · código {_esc(p.code_version)}"
                f" · uso comercial: {_esc(p.commercial_use)} · licença dos pesos: "
                f"{_esc(p.license_weights)}<br/>pesos sha256: {_esc(weights)}<br/>"
                f"manifesto sha256: {_esc(p.manifest_sha256)}",
                small,
            )
        )
    if content.warnings:
        story += [Spacer(1, 3 * mm), Paragraph("Avisos", h3)]
        story += [Paragraph(_esc(w), small) for w in content.warnings]

    story += [Spacer(1, 6 * mm), Paragraph("Revisão e assinatura do médico", h2)]
    if sign.signed_at is None:
        story.append(
            Paragraph(
                "<b>NÃO ASSINADO.</b> Este relatório ainda não foi revisado e assinado por "
                "médico responsável.",
                body,
            )
        )
    else:
        story.append(
            Paragraph(
                f"Assinado por <b>{_esc(sign.signer_name)}</b> ({_esc(sign.signer_username)}"
                f"{', ' + _esc(sign.professional_id) if sign.professional_id else ''}) em "
                f"{_esc(sign.signed_at)} UTC.<br/>A assinatura cobre report_sha256 "
                f"{_esc(sign.report_sha256)} e result_sha256 {_esc(content.result_sha256)}."
                f"<br/>signature_sha256: {_esc(sign.signature_sha256)}",
                body,
            )
        )
    story += [Spacer(1, 4 * mm), Paragraph(_esc(content.disclaimer), small)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=20 * mm,
        bottomMargin=16 * mm,
        title="Relatório preliminar de pesquisa",
        author="RadiologyAI (software de pesquisa)",
        subject=HEADER_PT,
        invariant=1,
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()
