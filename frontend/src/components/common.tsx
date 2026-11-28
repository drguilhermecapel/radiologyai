/** Componentes pequenos e reutilizáveis: selos, avisos, erros. */
import type { ReactNode } from "react";

import { useI18n } from "../i18n/i18n";
import type { MessageKey } from "../i18n/messages";
import { decisionBadge, levelBadge, type EvidenceLevel, type Finding } from "../lib/evidence";

export function LevelBadge({ level, withMeaning = false }: { level: EvidenceLevel; withMeaning?: boolean }): ReactNode {
  const { t } = useI18n();
  const b = levelBadge(level);
  return (
    <span className="level-wrap">
      <span
        className={`badge ${b.tone}`}
        title={t(b.meaningKey as MessageKey)}
        data-testid={`level-badge-${level}`}
      >
        {t(b.labelKey as MessageKey)}
      </span>
      {withMeaning ? <span className="level-meaning">{t(b.meaningKey as MessageKey)}</span> : null}
    </span>
  );
}

export function DecisionBadge({ decision }: { decision: Finding["decision"] }): ReactNode {
  const { t } = useI18n();
  const b = decisionBadge(decision);
  return <span className={`badge decision-${b.tone}`}>{t(b.labelKey as MessageKey)}</span>;
}

export function Pill({ tone, children }: { tone: string; children: ReactNode }): ReactNode {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

export function ErrorBox({ error, onRetry }: { error: Error; onRetry?: () => void }): ReactNode {
  const { t } = useI18n();
  return (
    <div className="error-box" role="alert">
      <strong>{t("app.error")}:</strong> {error.message}
      {onRetry ? (
        <button type="button" className="btn btn-small" onClick={onRetry}>
          {t("app.retry")}
        </button>
      ) : null}
    </div>
  );
}

export function Loading(): ReactNode {
  const { t } = useI18n();
  return (
    <p className="muted" role="status" aria-live="polite">
      {t("app.loading")}
    </p>
  );
}

/** Hash longo abreviado, com o valor completo acessível. */
export function Hash({ value }: { value: string | null | undefined }): ReactNode {
  if (!value) return <span className="muted">—</span>;
  return (
    <code className="hash" title={value}>
      {value.slice(0, 12)}…
    </code>
  );
}

/** Persistente em todas as telas: o produto é só de pesquisa. */
export function ResearchUseBanner(): ReactNode {
  const { t } = useI18n();
  return (
    <div className="ruo-banner" role="note" data-testid="ruo-banner">
      <strong>{t("app.banner")}</strong>
      <span className="ruo-detail">{t("app.banner.detail")}</span>
    </div>
  );
}

/** Persistente enquanto o servidor estiver no modo silencioso do piloto (REQ-195). */
export function ShadowModeBanner(): ReactNode {
  const { t } = useI18n();
  return (
    <div className="shadow-banner" role="note" data-testid="shadow-banner">
      <strong>{t("shadow.banner")}</strong>
      <span className="ruo-detail">{t("shadow.banner.detail")}</span>
    </div>
  );
}

/** Estado vazio NEUTRO no lugar dos achados: nunca sugere "normal" nem "sem achados". */
export function ShadowEmptyState(): ReactNode {
  const { t } = useI18n();
  return (
    <section className="card shadow-empty" aria-labelledby="shadow-empty-title" data-testid="shadow-empty">
      <h2 id="shadow-empty-title">{t("shadow.empty.title")}</h2>
      <p className="muted">{t("shadow.empty")}</p>
    </section>
  );
}

export function ResearchModeBanner(): ReactNode {
  const { t } = useI18n();
  return (
    <div className="research-banner" role="alert" data-testid="research-banner">
      {t("research.banner")}
    </div>
  );
}
