/**
 * Trilha de auditoria (somente administradores): eventos encadeados por hash,
 * filtros por evento/sujeito, paginação por ``after_seq`` e verificação da
 * cadeia feita pelo SERVIDOR (o cliente apenas mostra o veredito).
 */
import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { api, unwrap, type Schemas } from "../api/client";
import { ErrorBox, Hash } from "../components/common";
import { useI18n } from "../i18n/i18n";
import { useSession } from "../session";

type AuditEvent = Schemas["AuditEventOut"];
type Verify = Schemas["AuditVerifyOut"];

const PAGE = 100;

export function AuditPage(): ReactNode {
  const { t } = useI18n();
  const { can } = useSession();
  const [event, setEvent] = useState("");
  const [subject, setSubject] = useState("");
  const [items, setItems] = useState<AuditEvent[]>([]);
  const [next, setNext] = useState<number | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [verify, setVerify] = useState<Verify | null>(null);

  const load = async (afterSeq: number | null): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      const out = unwrap(
        await api.GET("/api/v1/audit", {
          params: {
            query: {
              limit: PAGE,
              ...(afterSeq !== null ? { after_seq: afterSeq } : {}),
              ...(event.trim() ? { event: event.trim() } : {}),
              ...(subject.trim() ? { subject_id: subject.trim() } : {}),
            },
          },
        }),
      );
      setItems((prev) => (afterSeq === null ? out.items : [...prev, ...out.items]));
      setNext(out.next_after_seq);
      setLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  const runVerify = async (): Promise<void> => {
    setError(null);
    try {
      setVerify(unwrap(await api.GET("/api/v1/audit/verify")));
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    }
  };

  const allowed = can("audit:read");
  useEffect(() => {
    if (allowed) void load(null);
    // Só a carga inicial; filtros recarregam pelo formulário.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allowed]);

  if (!allowed) return <p className="error-box">{t("audit.forbidden")}</p>;

  return (
    <div className="page audit">
      <h1>{t("audit.title")}</h1>
      <div className="btn-row">
        <button type="button" className="btn" onClick={() => void runVerify()} data-testid="audit-verify">
          {t("audit.verify")}
        </button>
      </div>
      {verify ? (
        <p className={verify.ok ? "ok-note" : "error-box"} role={verify.ok ? "status" : "alert"} data-testid="audit-verify-result">
          {verify.ok
            ? t("audit.verify.ok", { events: verify.events, hash: verify.last_hash.slice(0, 16) })
            : t("audit.verify.broken", { seq: verify.broken_at_seq ?? "?", detail: verify.detail ?? "" })}
        </p>
      ) : null}
      <form
        className="filters"
        onSubmit={(e) => {
          e.preventDefault();
          void load(null);
        }}
      >
        <label className="field">
          <span>{t("audit.filter.event")}</span>
          <input value={event} onChange={(e) => setEvent(e.target.value)} maxLength={64} />
        </label>
        <label className="field">
          <span>{t("audit.filter.subject")}</span>
          <input value={subject} onChange={(e) => setSubject(e.target.value)} maxLength={64} />
        </label>
        <button type="submit" className="btn btn-small" disabled={busy}>
          {t("audit.apply")}
        </button>
      </form>
      {error ? <ErrorBox error={error} onRetry={() => void load(null)} /> : null}
      <div className="table-wrap">
        <table className="table audit-table" data-testid="audit-table">
          <thead>
            <tr>
              <th scope="col">{t("audit.col.seq")}</th>
              <th scope="col">{t("audit.col.time")}</th>
              <th scope="col">{t("audit.col.event")}</th>
              <th scope="col">{t("audit.col.actor")}</th>
              <th scope="col">{t("audit.col.subject")}</th>
              <th scope="col">{t("audit.col.payload")}</th>
              <th scope="col">{t("audit.col.hash")}</th>
            </tr>
          </thead>
          <tbody>
            {loaded && items.length === 0 ? (
              <tr>
                <td colSpan={7} className="muted">
                  {t("audit.empty")}
                </td>
              </tr>
            ) : null}
            {items.map((ev) => (
              <tr key={ev.seq}>
                <td>{ev.seq}</td>
                <td className="nowrap">{ev.timestamp_utc}</td>
                <td>
                  <code>{ev.event}</code>
                </td>
                <td>{ev.actor_username ?? "—"}</td>
                <td className="small">
                  {ev.subject_type ? `${ev.subject_type}: ` : ""}
                  {ev.subject_id ?? "—"}
                </td>
                <td>
                  <details>
                    <summary className="small">JSON</summary>
                    <pre className="payload">{JSON.stringify(ev.payload, null, 2)}</pre>
                  </details>
                </td>
                <td>
                  <Hash value={ev.record_hash} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {next !== null ? (
        <button type="button" className="btn btn-small" disabled={busy} onClick={() => void load(next)}>
          {t("audit.more")}
        </button>
      ) : null}
    </div>
  );
}
