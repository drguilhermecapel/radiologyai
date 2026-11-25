/**
 * Laudo preliminar: gerar instantâneo, assinar (senha + os dois hashes que o
 * médico viu) e baixar PDF/SR/FHIR/SEG. Integridade recalculada pelo servidor
 * é mostrada a cada leitura; violada => sem downloads.
 */
import { useState } from "react";
import type { ReactNode } from "react";

import { api, downloadDocument, unwrap, type Schemas } from "../api/client";
import { useI18n } from "../i18n/i18n";
import type { MessageKey } from "../i18n/messages";
import { useAsync } from "../lib/useAsync";
import { useSession } from "../session";
import { ErrorBox, Hash, Loading, Pill } from "./common";

type ReportOut = Schemas["ReportOut"];

function SignForm({ report, onSigned }: { report: ReportOut; onSigned: () => void }): ReactNode {
  const { t } = useI18n();
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sign = async (): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      unwrap(
        await api.POST("/api/v1/reports/{report_id}/sign", {
          params: { path: { report_id: report.id } },
          body: { password, report_sha256: report.report_sha256, result_sha256: report.result_sha256 },
        }),
      );
      onSigned();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      setPassword("");
    }
  };
  return (
    <form
      className="sign-form"
      onSubmit={(e) => {
        e.preventDefault();
        void sign();
      }}
    >
      <label className="field">
        <span>{t("report.signPassword")}</span>
        <input
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          data-testid="sign-password"
        />
      </label>
      {error ? (
        <p className="error-box" role="alert">
          {error}
        </p>
      ) : null}
      <button type="submit" className="btn btn-primary" disabled={busy || password === ""} data-testid="sign-submit">
        {busy ? t("report.signBusy") : t("report.sign")}
      </button>
    </form>
  );
}

function ReportCard({ report, onChanged }: { report: ReportOut; onChanged: () => void }): ReactNode {
  const { t, dt } = useI18n();
  const { can } = useSession();
  const [dlError, setDlError] = useState<string | null>(null);
  const ok = report.integrity.content_matches_hash && report.integrity.result_matches_hash;
  const c = report.content;
  const docs: { key: string; label: string; url: string; name: string }[] = Object.entries(report.downloads).map(
    ([key, url]) => {
      const seg = /^seg\/(\d+)$/.exec(key);
      const label = seg
        ? t("report.download.seg", { index: seg[1] ?? "" })
        : t(`report.download.${key}` as MessageKey);
      const ext = key === "pdf" ? "pdf" : key === "fhir" ? "json" : "dcm";
      return { key, label, url, name: `laudo-${report.id}-${key.replace("/", "-")}.${ext}` };
    },
  );
  return (
    <article className="report-card" data-testid="report-card">
      <header>
        <h3>
          {t("report.title")} <Hash value={report.report_sha256} />
        </h3>
        <p className="muted small">{t("report.created", { when: dt(report.created_at) })}</p>
        {report.signed ? (
          <Pill tone="ok">
            <span data-testid="report-signed">
              {t("report.signed", { name: report.signed_by_name ?? report.signed_by ?? "?", when: dt(report.signed_at) })}
            </span>
          </Pill>
        ) : (
          <Pill tone="neutral">{t("report.unsigned")}</Pill>
        )}
      </header>
      <p className={ok ? "ok-note" : "error-box"} role={ok ? undefined : "alert"}>
        {t("report.integrity")}: {ok ? t("report.integrity.ok") : t("report.integrity.bad")}
        {report.integrity.signature_valid === null
          ? null
          : ` · ${report.integrity.signature_valid ? t("report.signatureValid") : t("report.signatureInvalid")}`}
      </p>
      <p className="muted small">{c.header}</p>
      <ul className="report-findings">
        {c.findings.map((f) => (
          <li key={f.idx}>
            <strong>{f.condition_name}</strong> — {t(`level.${f.evidence_level.toLowerCase()}.label` as MessageKey)} ·{" "}
            {f.decision_pt}
            {f.review ? ` · ${t(`rv.action.${f.review.action}` as MessageKey)} (${f.review.reviewer_name})` : ""}
          </li>
        ))}
      </ul>
      {c.no_findings_statement ? <p className="no-findings">{c.no_findings_statement}</p> : null}
      {c.excluded_research_findings > 0 ? (
        <p className="muted small">{t("report.excluded", { count: c.excluded_research_findings })}</p>
      ) : null}
      {c.unreviewed_findings > 0 ? (
        <p className="pending-note">{t("report.unreviewed", { count: c.unreviewed_findings })}</p>
      ) : null}
      {!report.signed && can("reports:sign") && c.unreviewed_findings === 0 && ok ? (
        <SignForm report={report} onSigned={onChanged} />
      ) : null}
      {ok ? (
        <div className="downloads">
          <h4>{t("report.downloads")}</h4>
          <div className="btn-row">
            {docs.map((d) => (
              <button
                key={d.key}
                type="button"
                className="btn btn-small"
                data-testid={`download-${d.key.replace("/", "-")}`}
                onClick={() => {
                  setDlError(null);
                  downloadDocument(d.url, d.name).catch((err: unknown) =>
                    setDlError(err instanceof Error ? err.message : String(err)),
                  );
                }}
              >
                {d.label}
              </button>
            ))}
          </div>
          {dlError ? (
            <p className="error-box" role="alert">
              {dlError}
            </p>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

export function ReportPanel({ analysisId, pendingReviews }: { analysisId: string; pendingReviews: number }): ReactNode {
  const { t } = useI18n();
  const { can } = useSession();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reports = useAsync(
    async () =>
      unwrap(
        await api.GET("/api/v1/analyses/{analysis_id}/reports", { params: { path: { analysis_id: analysisId } } }),
      ),
    [analysisId],
  );
  const create = async (): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      unwrap(
        await api.POST("/api/v1/analyses/{analysis_id}/reports", { params: { path: { analysis_id: analysisId } } }),
      );
      reports.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="report card" aria-labelledby="report-title" data-testid="report-panel">
      <h2 id="report-title">{t("report.title")}</h2>
      {can("reports:create") ? (
        <>
          <p className="muted small">{t("report.createHelp")}</p>
          {pendingReviews > 0 ? <p className="pending-note">{t("rv.pending", { count: pendingReviews })}</p> : null}
          <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void create()} data-testid="report-create">
            {t("report.create")}
          </button>
        </>
      ) : (
        <p className="muted small">{t("report.notAllowed")}</p>
      )}
      {error ? (
        <p className="error-box" role="alert">
          {error}
        </p>
      ) : null}
      {reports.error ? <ErrorBox error={reports.error} onRetry={reports.reload} /> : null}
      {reports.loading && !reports.data ? <Loading /> : null}
      {reports.data && reports.data.length > 0 ? (
        <>
          <h3 className="sr-only">{t("report.list")}</h3>
          {reports.data.map((r) => (
            <ReportCard key={r.id} report={r} onChanged={reports.reload} />
          ))}
        </>
      ) : null}
    </section>
  );
}
