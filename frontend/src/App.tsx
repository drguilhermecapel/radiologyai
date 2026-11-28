import { useEffect } from "react";
import type { ReactNode } from "react";

import { Loading, ResearchModeBanner, ResearchUseBanner, ShadowModeBanner } from "./components/common";
import { useI18n } from "./i18n/i18n";
import type { MessageKey } from "./i18n/messages";
import { navigate, routeHref, useRoute, type Route } from "./lib/route";
import { AuditPage } from "./pages/AuditPage";
import { CatalogPage } from "./pages/CatalogPage";
import { LoginPage } from "./pages/LoginPage";
import { StudyPage } from "./pages/StudyPage";
import { WorklistPage } from "./pages/WorklistPage";
import { useSession } from "./session";

function Header({ route }: { route: Route }): ReactNode {
  const { t, lang, setLang } = useI18n();
  const { user, can, logout, research, setResearch } = useSession();
  const link = (target: Route, key: MessageKey, active: boolean): ReactNode => (
    <a href={routeHref(target)} className={active ? "nav-link active" : "nav-link"} aria-current={active ? "page" : undefined}>
      {t(key)}
    </a>
  );
  return (
    <header className="app-header">
      <div className="brand">{t("app.name")}</div>
      {user ? (
        <nav aria-label={t("nav.main")} className="main-nav">
          {link({ name: "worklist" }, "nav.worklist", route.name === "worklist" || route.name === "study")}
          {can("catalog:read") ? link({ name: "catalog", tab: "conditions" }, "nav.catalog", route.name === "catalog") : null}
          {can("audit:read") ? link({ name: "audit" }, "nav.audit", route.name === "audit") : null}
        </nav>
      ) : null}
      <div className="header-tools">
        {user && can("research:view") ? (
          <label className="research-toggle" title={t("research.toggle.help")}>
            <input
              type="checkbox"
              role="switch"
              checked={research}
              onChange={(e) => setResearch(e.target.checked)}
              data-testid="research-toggle"
            />
            <span>{t("research.toggle")}</span>
          </label>
        ) : null}
        <label className="lang-select">
          <span className="sr-only">{t("nav.language")}</span>
          <select value={lang} onChange={(e) => setLang(e.target.value === "en" ? "en" : "pt")} aria-label={t("nav.language")}>
            <option value="pt">PT-BR</option>
            <option value="en">EN</option>
          </select>
        </label>
        {user ? (
          <>
            <span className="user-chip" data-testid="user-chip">
              {t("nav.user", { name: user.full_name, role: t(`role.${user.role}` as MessageKey) })}
            </span>
            <button type="button" className="btn btn-small" onClick={() => void logout()}>
              {t("nav.logout")}
            </button>
          </>
        ) : null}
      </div>
    </header>
  );
}

function Page({ route }: { route: Route }): ReactNode {
  const { t } = useI18n();
  switch (route.name) {
    case "login":
      return <LoginPage />;
    case "worklist":
      return <WorklistPage />;
    case "study":
      return <StudyPage key={route.studyId} studyId={route.studyId} analysisId={route.analysisId} />;
    case "catalog":
      return <CatalogPage tab={route.tab} />;
    case "audit":
      return <AuditPage />;
    case "notFound":
      return <p>{t("app.notFound")}</p>;
  }
}

export function App(): ReactNode {
  const { t } = useI18n();
  const route = useRoute();
  const { user, loading, research, shadow } = useSession();

  useEffect(() => {
    if (loading) return;
    if (!user && route.name !== "login") navigate({ name: "login" }, true);
    if (user && route.name === "login") navigate({ name: "worklist" }, true);
  }, [user, loading, route.name]);

  return (
    <div className={research ? "app research-on" : "app"}>
      <a className="skip-link" href="#main">
        {t("app.skip")}
      </a>
      <ResearchUseBanner />
      {shadow ? <ShadowModeBanner /> : null}
      <Header route={route} />
      {research ? <ResearchModeBanner /> : null}
      <main id="main" tabIndex={-1}>
        {loading ? <Loading /> : user || route.name === "login" ? <Page route={route} /> : null}
      </main>
    </div>
  );
}
