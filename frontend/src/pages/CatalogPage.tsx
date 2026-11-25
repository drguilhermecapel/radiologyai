/**
 * Catálogo de condições (por exame, com melhor nível de evidência e base) e
 * fichas de modelos (licenças, uso comercial, estado dos pesos, referências).
 * Tudo vem de ``GET /catalog`` e ``GET /models``; nenhuma métrica é exibida.
 */
import { useMemo, useState } from "react";
import type { ReactNode } from "react";

import { api, unwrap, type Schemas } from "../api/client";
import { ErrorBox, Hash, LevelBadge, Loading, Pill } from "../components/common";
import { useI18n } from "../i18n/i18n";
import type { MessageKey } from "../i18n/messages";
import { EXAM_TYPES, type ExamType } from "../lib/domain";
import { LEVELS, type EvidenceLevel } from "../lib/evidence";
import { routeHref } from "../lib/route";
import { useAsync } from "../lib/useAsync";
import { useSession } from "../session";

type ModelCard = Schemas["ModelCardOut"];

/** Só links http(s) viram âncoras; o resto aparece como texto. */
function SafeLink({ href, children }: { href: string; children: ReactNode }): ReactNode {
  if (/^https?:\/\//i.test(href)) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  }
  return <span>{children}</span>;
}

function commercialTone(v: ModelCard["commercial_use"]): string {
  return v === "yes" ? "ok" : v === "no" ? "warn" : "neutral";
}

function ConditionsTab(): ReactNode {
  const { t, lang } = useI18n();
  const { research } = useSession();
  const [exam, setExam] = useState<ExamType | "">("");
  const [level, setLevel] = useState<EvidenceLevel | "">("");
  const [q, setQ] = useState("");
  const catalog = useAsync(async () => unwrap(await api.GET("/api/v1/catalog")), []);

  const rows = useMemo(() => {
    const data = catalog.data;
    if (!data) return [];
    const needle = q.trim().toLowerCase();
    return data.conditions.filter((c) => {
      if (!research && c.best_level === "D") return false;
      if (exam && !c.exam_types.includes(exam)) return false;
      if (level && c.best_level !== level) return false;
      if (needle) {
        const hay = `${c.name_pt} ${c.name_en} ${c.id} ${c.code.value}`.toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
  }, [catalog.data, exam, level, q, research]);

  const hiddenD = research ? 0 : (catalog.data?.conditions.filter((c) => c.best_level === "D").length ?? 0);
  if (catalog.error) return <ErrorBox error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <Loading />;
  const data = catalog.data;

  return (
    <>
      {data.protocol_id ? <p className="muted small">{t("catalog.protocol", { id: data.protocol_id })}</p> : null}
      <details className="card levels-help">
        <summary>{t("catalog.levels")}</summary>
        <ul className="level-legend">
          {LEVELS.filter((l) => research || l !== "D").map((l) => (
            <li key={l}>
              <LevelBadge level={l} withMeaning />
            </li>
          ))}
        </ul>
      </details>
      <fieldset className="filters">
        <legend className="sr-only">{t("worklist.filters")}</legend>
        <label className="field">
          <span>{t("catalog.filter.search")}</span>
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} data-testid="catalog-search" />
        </label>
        <label className="field">
          <span>{t("catalog.filter.exam")}</span>
          <select value={exam} onChange={(e) => setExam(e.target.value as ExamType | "")}>
            <option value="">{t("worklist.filter.all")}</option>
            {EXAM_TYPES.map((x) => (
              <option key={x} value={x}>
                {t(`exam.${x}` as MessageKey)}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>{t("catalog.filter.level")}</span>
          <select value={level} onChange={(e) => setLevel(e.target.value as EvidenceLevel | "")}>
            <option value="">{t("worklist.filter.all")}</option>
            {LEVELS.filter((l) => research || l !== "D").map((l) => (
              <option key={l} value={l}>
                {t(`level.${l.toLowerCase()}.label` as MessageKey)}
              </option>
            ))}
          </select>
        </label>
      </fieldset>
      <p className="muted" aria-live="polite">
        {t("catalog.count", { count: rows.length })}
        {hiddenD > 0 ? <span data-testid="catalog-hidden-d"> · {t("catalog.hiddenD", { count: hiddenD })}</span> : null}
      </p>
      <div className="table-wrap">
        <table className="table" data-testid="catalog-table">
          <thead>
            <tr>
              <th scope="col">{t("catalog.col.condition")}</th>
              <th scope="col">{t("catalog.col.exam")}</th>
              <th scope="col">{t("catalog.col.level")}</th>
              <th scope="col">{t("catalog.col.models")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={4} className="muted">
                  {t("catalog.empty")}
                </td>
              </tr>
            ) : null}
            {rows.map((c) => (
              <tr key={c.id}>
                <td>
                  <strong>{lang === "pt" ? c.name_pt : c.name_en}</strong>
                  <br />
                  <span className="muted small">
                    {c.code.scheme}:{c.code.value} · {c.id}
                  </span>
                  {c.research_only ? (
                    <>
                      {" "}
                      <Pill tone="research">{t("catalog.researchOnly")}</Pill>
                    </>
                  ) : null}
                </td>
                <td>{c.exam_types.map((x) => t(`exam.${x}` as MessageKey)).join(", ")}</td>
                <td>
                  <LevelBadge level={c.best_level} />
                </td>
                <td>
                  <ul className="entry-list">
                    {c.entries
                      .filter((e) => research || e.level !== "D")
                      .map((e) => (
                        <li key={`${e.model_id}-${e.label}`}>
                          <LevelBadge level={e.level} /> <code>{e.model_id}</code> · {e.label}
                          {e.headline ? <span className="muted small"> — {e.headline}</span> : null}
                          {e.basis.length > 0 ? (
                            <details>
                              <summary className="small">{t("catalog.basis")}</summary>
                              <ul className="kv">
                                {e.basis.map((b) => (
                                  <li key={b} className="small">
                                    <SafeLink href={b}>{b}</SafeLink>
                                  </li>
                                ))}
                              </ul>
                            </details>
                          ) : null}
                        </li>
                      ))}
                  </ul>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {research && data.uncatalogued.length > 0 ? (
        <details className="card">
          <summary>{t("catalog.uncatalogued")}</summary>
          <ul className="kv">
            {data.uncatalogued.map((u) => (
              <li key={`${u.model_id}-${u.label}`} className="small">
                <code>{u.model_id}</code> · {u.label} ({u.condition_id})
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </>
  );
}

function ModelRow({ m }: { m: ModelCard }): ReactNode {
  const { t } = useI18n();
  const { research } = useSession();
  const labels = m.labels.filter((l) => research || l.evidence_level !== "D");
  return (
    <article className="card model-card" data-testid={`model-${m.id}`}>
      <header className="model-head">
        <h3>
          {m.name} <span className="muted small">v{m.version}</span>
        </h3>
        <Pill tone={m.status === "available" ? "ok" : m.status === "planned" ? "neutral" : "bad"}>
          {t(`models.status.${m.status}` as MessageKey)}
        </Pill>{" "}
        <Pill tone={commercialTone(m.commercial_use)}>
          {t("models.col.commercial")}: {t(`models.commercial.${m.commercial_use}` as MessageKey)}
        </Pill>
      </header>
      <p>{m.summary}</p>
      {m.planned_reason ? <p className="pending-note">{m.planned_reason}</p> : null}
      <dl className="dl">
        <dt>{t("models.col.task")}</dt>
        <dd>
          {m.task} · {t("models.route", { route: m.route })}
        </dd>
        <dt>{t("models.col.exams")}</dt>
        <dd>{m.exam_types.map((x) => t(`exam.${x}` as MessageKey)).join(", ")}</dd>
        <dt>{t("models.col.license")}</dt>
        <dd>
          {m.license_code} / {m.license_weights}
        </dd>
        <dt>{t("models.col.weights")}</dt>
        <dd>
          {m.weights.length === 0 ? "—" : null}
          {m.weights.map((w) => (
            <span key={w.key} className="weight">
              {w.filename} <Pill tone={w.state === "ok" ? "ok" : "warn"}>{w.state}</Pill> <Hash value={w.sha256} />
            </span>
          ))}
        </dd>
        {m.depends_on.length > 0 ? (
          <>
            <dt>{t("models.dependsOn")}</dt>
            <dd>{m.depends_on.join(", ")}</dd>
          </>
        ) : null}
      </dl>
      <details>
        <summary>{t("models.labels", { count: labels.length })}</summary>
        <ul className="entry-list">
          {labels.map((l) => (
            <li key={l.key}>
              <LevelBadge level={l.evidence_level} /> {l.code.meaning}{" "}
              <span className="muted small">
                ({l.source_label} · {l.score_kind})
              </span>
            </li>
          ))}
        </ul>
      </details>
      {m.limitations.length > 0 ? (
        <details>
          <summary>{t("models.limitations")}</summary>
          <ul>
            {m.limitations.map((x) => (
              <li key={x}>{x}</li>
            ))}
          </ul>
        </details>
      ) : null}
      {m.training_data.length > 0 ? (
        <details>
          <summary>{t("models.trainingData")}</summary>
          <ul>
            {m.training_data.map((x) => (
              <li key={x}>{x}</li>
            ))}
          </ul>
          {m.training_data_terms ? <p className="muted small">{m.training_data_terms}</p> : null}
        </details>
      ) : null}
      {m.references.length > 0 ? (
        <details>
          <summary>{t("models.references")}</summary>
          <ul>
            {m.references.map((r, i) => {
              const url = r["url"] ?? r["doi"] ?? null;
              const label = r["title"] ?? r["citation"] ?? url ?? JSON.stringify(r);
              return (
                <li key={i} className="small">
                  {url ? <SafeLink href={url.startsWith("10.") ? `https://doi.org/${url}` : url}>{label}</SafeLink> : label}
                </li>
              );
            })}
          </ul>
        </details>
      ) : null}
    </article>
  );
}

function ModelsTab(): ReactNode {
  const { t } = useI18n();
  const [status, setStatus] = useState<ModelCard["status"] | "">("");
  const models = useAsync(async () => unwrap(await api.GET("/api/v1/models")), []);
  if (models.error) return <ErrorBox error={models.error} onRetry={models.reload} />;
  if (!models.data) return <Loading />;
  const rows = models.data.filter((m) => !status || m.status === status);
  return (
    <>
      <fieldset className="filters">
        <legend className="sr-only">{t("worklist.filters")}</legend>
        <label className="field">
          <span>{t("models.filter.status")}</span>
          <select value={status} onChange={(e) => setStatus(e.target.value as ModelCard["status"] | "")}>
            <option value="">{t("worklist.filter.all")}</option>
            {(["available", "planned", "retired"] as const).map((s) => (
              <option key={s} value={s}>
                {t(`models.status.${s}` as MessageKey)}
              </option>
            ))}
          </select>
        </label>
      </fieldset>
      <p className="muted" aria-live="polite">
        {t("models.count", { count: rows.length })}
      </p>
      <div className="model-grid">
        {rows.map((m) => (
          <ModelRow key={m.id} m={m} />
        ))}
      </div>
    </>
  );
}

export function CatalogPage({ tab }: { tab: "conditions" | "models" }): ReactNode {
  const { t } = useI18n();
  const { can } = useSession();
  return (
    <div className="page catalog">
      <h1>{t("catalog.title")}</h1>
      <nav className="tabs" aria-label={t("catalog.title")}>
        <a
          href={routeHref({ name: "catalog", tab: "conditions" })}
          className={tab === "conditions" ? "tab active" : "tab"}
          aria-current={tab === "conditions" ? "page" : undefined}
        >
          {t("catalog.tab.conditions")}
        </a>
        {can("models:read") ? (
          <a
            href={routeHref({ name: "catalog", tab: "models" })}
            className={tab === "models" ? "tab active" : "tab"}
            aria-current={tab === "models" ? "page" : undefined}
          >
            {t("catalog.tab.models")}
          </a>
        ) : null}
      </nav>
      {tab === "models" && can("models:read") ? <ModelsTab /> : <ConditionsTab />}
    </div>
  );
}
