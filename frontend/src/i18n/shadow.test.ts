import { describe, expect, it } from "vitest";

import { en, pt } from "./messages";

// REQ-195: texto exato da faixa do modo silencioso e estado vazio que nunca sugere normalidade.
describe("modo silencioso (REQ-195)", () => {
  it("faixa com o texto do piloto em PT e EN", () => {
    expect(pt["shadow.banner"]).toBe("Modo silencioso: resultados de IA ocultos do cuidado (piloto)");
    expect(en["shadow.banner"]).toBe("Silent mode: AI results hidden from care (pilot)");
  });

  it("estado vazio diz que não é exame normal", () => {
    expect(pt["shadow.empty"]).toMatch(/não indica exame normal/);
    expect(en["shadow.empty"]).toMatch(/does not mean a normal study/);
  });

  it("papel de pesquisa tem rótulo", () => {
    expect(pt["role.researcher"]).toBeTruthy();
    expect(en["role.researcher"]).toBeTruthy();
  });
});
