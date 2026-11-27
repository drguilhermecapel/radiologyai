import { describe, expect, it } from "vitest";

import { en, format, pt } from "./messages";

describe("mensagens", () => {
  it("EN cobre exatamente as chaves do PT-BR", () => {
    expect(Object.keys(en).sort()).toEqual(Object.keys(pt).sort());
  });
  it("as variáveis de interpolação coincidem entre idiomas", () => {
    const vars = (s: string): string[] => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1] ?? "").sort();
    for (const k of Object.keys(pt) as (keyof typeof pt)[]) {
      expect(vars(en[k]), k).toEqual(vars(pt[k]));
    }
  });
  it("faixa de pesquisa presente e sem a palavra diagnóstico afirmativa", () => {
    expect(pt["app.banner"]).toContain("não é dispositivo médico");
    expect(en["app.banner"].toLowerCase()).toContain("not a medical device");
  });
  it("format substitui e deixa visível o que falta", () => {
    expect(format("a {x} {y}", { x: 1 })).toBe("a 1 {y}");
  });
  it("significado dos níveis nunca usa 'validado' (linguagem vedada, REQ-191)", () => {
    for (const lv of ["a", "b", "c", "d"] as const) {
      expect(pt[`level.${lv}.meaning`].toLowerCase()).not.toMatch(/validad|validaç/);
      expect(en[`level.${lv}.meaning`].toLowerCase()).not.toMatch(/validat/);
    }
  });
  it("PT nunca chama escore de probabilidade", () => {
    expect(pt["findings.score"].toLowerCase()).not.toContain("probabilidade");
  });
});
