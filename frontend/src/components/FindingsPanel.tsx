/**
 * Painel de achados: agrupado por nível de evidência, com significado em
 * linguagem simples, escore x ponto de operação (nunca "probabilidade" sem
 * calibração), abstenções com motivo, proveniência e revisão por achado.
 */
import { useState } from "react";
import type { ReactNode } from "react";

import { api, unwrap, type Schemas } from "../api/client";
import { useI18n } from "../i18n/i18n";
import type { MessageKey } from "../i18n/messages";
import {
  formatScore,
  groupFindings,
  isReviewable,
  latestReviews,
  operatingPointRelations,
  scoreDisplay,
  type Finding,
  type IndexedFinding,
  type Review,
} from "../lib/evidence";
import { DecisionBadge, Hash, LevelBadge, Pill } from "./common";

type AnalysisOut = Schemas["AnalysisOut"];
type Provenance = Schemas["ModelProvenance"];

function ScoreBlock({ f }: { f: Finding }): ReactNode {
  const { t } = useI18n();
  const s = scoreDisplay(f);
  switch (s.kind) {
    case "none":
      return null;
    case "probability":
      return (
        <p className="score">
          {t("findings.probability")}: <strong>{formatScore(s.value)}</strong>{" "}
          <span className="muted small">{t("findings.calibration", { id: s.calibrationId })}</span>
        </p>
      );
    case "category":
      return (
        <div className="score">
          <p>
            <strong>{t("findings.category", { category: s.category })}</strong>
          </p>
          <details>
            <summary>{t("findings.classScores")}</summary>
            <ul className="kv">
              {Object.entries(s.classScores).map(([k, v]) => (
                <li key={k}>
                  {k}: {formatScore(v)}
                </li>
              ))}
            </ul>
          </details>
        </div>
      );
    case "score":
      return (
        <p className="score">
          {t("findings.score", { kind: s.scoreKind })}: <strong data-testid="raw-score">{formatScore(s.raw)}</strong>
          {s.opNormalized !== null ? (
            <span className="muted small">
              {" "}
              · {t("findings.scoreNorm")}: {formatScore(s.opNormalized)}
            </span>
          ) : null}
        </p>
      );
  }
}

function OperatingPoints({ f }: { f: Finding }): ReactNode {
  const { t } = useI18n();
  const rel = operatingPointRelations(f);
  if (rel.length === 0) {
    return f.decision === "abstained" ? null : <p className="muted small">{t("findings.op.none")}</p>;
  }
  return (
    <ul className="op-list" aria-label={t("findings.op")}>
      {rel.map(({ op, value, meets }) => (
        <li key={op.name}>
          <span>
            {t("findings.op.desc", {
              name: op.name,
              role: op.role,
              dir: op.direction === "le" ? "≤" : "≥",
              threshold: formatScore(op.threshold),
              on: op.on,
            })}
          </span>
          <br />
          <span className="muted small">
            {value !== null
              ? t("findings.op.value", {
                  value: formatScore(value),
                  meets: meets ? t("findings.op.meets") : t("findings.op.notMeets"),
                })
              : t("findings.op.notComparable")}
          </span>
          <br />
          <span className="muted small">{t("findings.op.source", { source: op.source })}</span>
        </li>
      ))}
    </ul>
  );
}

function ProvenanceBlock({ p }: { p: Provenance | undefined }): ReactNode {
  const { t } = useI18n();
  if (!p) return null;
  return (
    <details className="provenance">
      <summary>
        {t("findings.provenance")}: {t("findings.model", { id: p.model_id, version: p.model_version })}
        {p.commercial_use === "no" ? (
          <>
            {" "}
            <Pill tone="warn">{t("findings.noncommercial")}</Pill>
          </>
        ) : null}
      </summary>
      <ul className="kv">
        <li>{t("findings.license", { license: p.license_weights })}</li>
        <li>{t("findings.commercial", { value: t(`models.commercial.${p.commercial_use}` as MessageKey) })}</li>
        <li>
          {t("findings.manifest")}: <Hash value={p.manifest_sha256} />
        </li>
        {Object.entries(p.weights_sha256).map(([k, v]) => (
          <li key={k}>
            {t("findings.weights")} {k}: <Hash value={v} />
          </li>
        ))}
        <li className="muted small">code {p.code_version}</li>
      </ul>
    </details>
  );
}

interface ReviewControlsProps {
  analysisId: string;
  resultSha256: string;
  item: IndexedFinding;
  current: Review | undefined;
  onReviewed: () => void;
}

function ReviewControls({ analysisId, resultSha256, item, current, onReviewed }: ReviewControlsProps): ReactNode {
  const { t, dt } = useI18n();
  const [mode, setMode] = useState<"accept" | "reject" | "amend" | null>(null);
  const [comment, setComment] = useState("");
  const [amendDecision, setAmendDecision] = useState<"" | "positive" | "negative" | "indeterminate">("");
  const [amendText, setAmendText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (): Promise<void> => {
    if (!mode) return;
    setBusy(true);
    setError(null);
    try {
      const body: Schemas["ReviewIn"] = {
        finding_idx: item.fullIdx,
        action: mode,
        result_sha256: resultSha256,
        ...(comment.trim() ? { comment: comment.trim() } : {}),
        ...(mode === "amend" && amendDecision ? { amended_decision: amendDecision } : {}),
        ...(mode === "amend" && amendText.trim() ? { amended_text: amendText.trim() } : {}),
      };
      unwrap(
        await api.POST("/api/v1/analyses/{analysis_id}/reviews", {
          params: { path: { analysis_id: analysisId } },
          body,
        }),
      );
      setMode(null);
      setComment("");
      setAmendDecision("");
      setAmendText("");
      onReviewed();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const amendInvalid = mode === "amend" && !amendDecision && !amendText.trim();
  return (
    <div className="review-controls" data-testid={`review-${item.fullIdx}`}>
      {current ? (
        <p className={`review-current review-${current.action}`}>
          {t("rv.current", {
            action: t(`rv.action.${current.action}` as MessageKey),
            user: current.reviewer_username,
            when: dt(current.created_at),
          })}
          {current.amended_decision ? ` · ${current.amended_decision}` : ""}
          {current.amended_text ? ` · “${current.amended_text}”` : ""}
          {current.comment ? ` · ${current.comment}` : ""}
        </p>
      ) : null}
      <div className="btn-row">
        {(["accept", "reject", "amend"] as const).map((a) => (
          <button
            key={a}
            type="button"
            className={mode === a ? "btn btn-small active" : "btn btn-small"}
            aria-pressed={mode === a}
            onClick={() => setMode(mode === a ? null : a)}
            data-testid={`rv-${a}-${item.fullIdx}`}
          >
            {t(`rv.${a}` as MessageKey)}
          </button>
        ))}
      </div>
      {mode ? (
        <div className="review-form">
          {mode === "amend" ? (
            <>
              <label className="field">
                <span>{t("rv.amendDecision")}</span>
                <select value={amendDecision} onChange={(e) => setAmendDecision(e.target.value as typeof amendDecision)}>
                  <option value="">—</option>
                  <option value="positive">{t("decision.positive")}</option>
                  <option value="negative">{t("decision.negative")}</option>
                  <option value="indeterminate">{t("decision.indeterminate")}</option>
                </select>
              </label>
              <label className="field">
                <span>{t("rv.amendText")}</span>
                <textarea value={amendText} maxLength={4000} onChange={(e) => setAmendText(e.target.value)} />
              </label>
            </>
          ) : null}
          <label className="field">
            <span>{t("rv.comment")}</span>
            <textarea value={comment} maxLength={4000} onChange={(e) => setComment(e.target.value)} />
          </label>
          {error ? (
            <p className="error-box" role="alert">
              {error}
            </p>
          ) : null}
          <div className="btn-row">
            <button
              type="button"
              className="btn btn-primary btn-small"
              disabled={busy || amendInvalid}
              onClick={() => void submit()}
              data-testid={`rv-submit-${item.fullIdx}`}
            >
              {t("rv.submit")}
            </button>
            <button type="button" className="btn btn-small" onClick={() => setMode(null)}>
              {t("rv.cancel")}
            </button>
          </div>
          <p className="muted small">{t("rv.hashNote")}</p>
        </div>
      ) : null}
    </div>
  );
}

export interface FindingsPanelProps {
  analysis: AnalysisOut;
  researchView: boolean;
  canReview: boolean;
  activeExplanationId: string | null;
  onShowExplanation: (id: string) => void;
  onReviewed: () => void;
}

export function FindingsPanel(props: FindingsPanelProps): ReactNode {
  const { analysis, researchView, canReview } = props;
  const { t } = useI18n();
  const result = analysis.result;
  if (!result) return null;
  const groups = groupFindings(result.findings, analysis.finding_indices, researchView);
  const sha = analysis.result_sha256 ?? "";
  const reviews = latestReviews(analysis.reviews.filter((r) => r.result_sha256 === sha));
  const mainCount = groups.filter((g) => g.level !== "D").reduce((n, g) => n + g.items.length, 0);
  const pending = groups
    .flatMap((g) => g.items)
    .filter((it) => isReviewable(it.finding) && !reviews.has(it.fullIdx)).length;
  const measurementsByName = new Map(result.measurements.map((m) => [m.name, m]));

  return (
    <section className="findings card" aria-labelledby="findings-title" data-testid="findings-panel">
      <h2 id="findings-title">{t("findings.title")}</h2>
      <p className="disclaimer">{t("findings.disclaimer")}</p>
      {!researchView && analysis.hidden_research_findings > 0 ? (
        <p className="muted small" data-testid="hidden-research">
          {t("research.hidden", { count: analysis.hidden_research_findings })}
        </p>
      ) : null}
      {mainCount === 0 ? (
        <p className="no-findings" data-testid="no-findings">
          {t("findings.none")}
        </p>
      ) : canReview ? (
        <p className={pending > 0 ? "pending-note" : "ok-note"} aria-live="polite" data-testid="pending-count">
          {pending > 0 ? t("rv.pending", { count: pending }) : t("rv.allDone")}
        </p>
      ) : null}

      {groups.map((g) => (
        <section key={g.level} className={`finding-group group-${g.badge.tone}`} aria-label={t(g.badge.labelKey as MessageKey)}>
          <h3>
            <LevelBadge level={g.level} />{" "}
            {t("findings.group", { label: t(g.badge.labelKey as MessageKey), count: g.items.length })}
          </h3>
          <p className="level-meaning">{t(g.badge.meaningKey as MessageKey)}</p>
          <ul className="finding-list">
            {g.items.map((it) => {
              const f = it.finding;
              const m = f.measurement_ref ? measurementsByName.get(f.measurement_ref) : undefined;
              return (
                <li key={it.fullIdx} className="finding" data-testid={`finding-${it.fullIdx}`}>
                  <div className="finding-head">
                    <strong>{f.code.meaning}</strong> <DecisionBadge decision={f.decision} />
                    {g.level === "D" ? <Pill tone="research">{t("findings.researchItem")}</Pill> : null}
                  </div>
                  <p className="muted small">
                    {t("findings.code", { scheme: f.code.scheme, value: f.code.value, meaning: f.code.meaning })} ·{" "}
                    {f.condition_id} · {f.label}
                  </p>
                  <ScoreBlock f={f} />
                  {f.decision === "abstained" && f.abstention_reason ? (
                    <p className="abstention">
                      {t("findings.abstention", { reason: t(`abst.${f.abstention_reason}` as MessageKey) })}
                    </p>
                  ) : null}
                  {m && m.value !== null ? (
                    <p>{t("findings.measurement", { name: m.name, value: formatScore(m.value, 2), unit: m.unit })}</p>
                  ) : null}
                  <OperatingPoints f={f} />
                  {(f.explanation_ids ?? []).map((id) => (
                    <button
                      key={id}
                      type="button"
                      className={props.activeExplanationId === id ? "btn btn-small active" : "btn btn-small"}
                      onClick={() => props.onShowExplanation(id)}
                      data-testid={`show-map-${it.fullIdx}`}
                    >
                      {t("findings.showHeatmap")}
                    </button>
                  ))}
                  <ProvenanceBlock p={result.provenance[f.provenance_idx]} />
                  {canReview && isReviewable(f) ? (
                    <ReviewControls
                      analysisId={analysis.id}
                      resultSha256={sha}
                      item={it}
                      current={reviews.get(it.fullIdx)}
                      onReviewed={props.onReviewed}
                    />
                  ) : null}
                </li>
              );
            })}
          </ul>
        </section>
      ))}

      {result.measurements.length > 0 ? (
        <details className="measurements">
          <summary>{t("findings.measurements")}</summary>
          <ul className="kv">
            {result.measurements.map((m) => (
              <li key={m.name}>
                {m.value !== null
                  ? t("findings.measurement", { name: m.name, value: formatScore(m.value, 2), unit: m.unit })
                  : t("findings.measurement.abstained", {
                      name: m.name,
                      reason: m.abstention_reason ? t(`abst.${m.abstention_reason}` as MessageKey) : "—",
                    })}
                <span className="muted small"> · {m.method}</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      <details className="runs">
        <summary>{t("findings.runs")}</summary>
        <ul className="kv">
          {result.runs.map((r) => (
            <li key={r.model_id}>
              {t("findings.run", { model: r.model_id, status: t(`run.${r.status}` as MessageKey) })}
              {r.error ? <span className="muted small"> — {r.error}</span> : null}
              {r.qa && !r.qa.applicable ? (
                <span className="muted small">
                  {" "}
                  — {r.qa.reasons.map((x) => t(`abst.${x}` as MessageKey)).join(", ")}
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      </details>
      {result.warnings.length > 0 ? (
        <details className="warnings" open>
          <summary>{t("findings.warnings")}</summary>
          <ul>
            {result.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
