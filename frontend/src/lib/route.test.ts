import { describe, expect, it } from "vitest";

import { parseRoute, routeHref, type Route } from "./route";

describe("rotas por hash", () => {
  it("ida e volta", () => {
    const routes: Route[] = [
      { name: "login" },
      { name: "worklist" },
      { name: "study", studyId: "abc", analysisId: null },
      { name: "study", studyId: "a b", analysisId: "x/y" },
      { name: "catalog", tab: "conditions" },
      { name: "catalog", tab: "models" },
      { name: "audit" },
    ];
    for (const r of routes) expect(parseRoute(routeHref(r))).toEqual(r);
  });
  it("vazio vai para a lista e desconhecido para 404", () => {
    expect(parseRoute("")).toEqual({ name: "worklist" });
    expect(parseRoute("#/nada")).toEqual({ name: "notFound" });
    expect(parseRoute("#/studies")).toEqual({ name: "notFound" });
  });
});
