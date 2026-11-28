/**
 * Roteador mínimo por hash (``#/caminho``): sem dependência externa e sem
 * configuração de servidor para rotas profundas.
 */
import { useEffect, useState } from "react";

export type Route =
  | { name: "login" }
  | { name: "worklist" }
  | { name: "study"; studyId: string; analysisId: string | null }
  | { name: "catalog"; tab: "conditions" | "models" }
  | { name: "audit" }
  | { name: "notFound" };

/** Interpreta ``#/...`` (sem o ``#``) numa rota tipada. */
export function parseRoute(hash: string): Route {
  const raw = hash.replace(/^#/, "");
  const [path = "", query = ""] = raw.split("?", 2);
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(query);
  if (parts.length === 0) return { name: "worklist" };
  const [head, id] = parts;
  switch (head) {
    case "login":
      return { name: "login" };
    case "worklist":
      return { name: "worklist" };
    case "studies":
      if (id && parts.length === 2) {
        return { name: "study", studyId: id, analysisId: params.get("analysis") };
      }
      return { name: "notFound" };
    case "catalog":
      return { name: "catalog", tab: id === "models" ? "models" : "conditions" };
    case "audit":
      return { name: "audit" };
    default:
      return { name: "notFound" };
  }
}

export function routeHref(route: Route): string {
  switch (route.name) {
    case "login":
      return "#/login";
    case "worklist":
      return "#/worklist";
    case "study": {
      const base = `#/studies/${encodeURIComponent(route.studyId)}`;
      return route.analysisId ? `${base}?analysis=${encodeURIComponent(route.analysisId)}` : base;
    }
    case "catalog":
      return route.tab === "models" ? "#/catalog/models" : "#/catalog";
    case "audit":
      return "#/audit";
    case "notFound":
      return "#/";
  }
}

export function navigate(route: Route, replace = false): void {
  const href = routeHref(route);
  if (replace) window.location.replace(href);
  else window.location.hash = href.slice(1);
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = (): void => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
