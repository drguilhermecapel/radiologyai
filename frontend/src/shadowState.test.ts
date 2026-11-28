import { describe, expect, it } from "vitest";

import { en, pt } from "./i18n/messages";
import { shadowStateFrom } from "./shadowState";

// REQ-195: a faixa falha fechada — status ilegível nunca vira "modo desligado".
describe("estado do modo silencioso (REQ-195)", () => {
  it("lê o estado do servidor", () => {
    expect(shadowStateFrom({ shadow_mode: true })).toBe("on");
    expect(shadowStateFrom({ shadow_mode: false })).toBe("off");
  });

  it("falha de leitura ou resposta sem o campo => desconhecido, não desligado", () => {
    expect(shadowStateFrom(null)).toBe("unknown");
    expect(shadowStateFrom(undefined)).toBe("unknown");
    expect(shadowStateFrom({})).toBe("unknown");
    expect(shadowStateFrom({ shadow_mode: "true" })).toBe("unknown");
  });

  it("faixa neutra de estado desconhecido em PT e EN", () => {
    expect(pt["shadow.unknown"]).toBe("Estado do modo silencioso desconhecido");
    expect(en["shadow.unknown"]).toBe("Silent mode status unknown");
    expect(pt["shadow.unknown.detail"]).toBeTruthy();
    expect(en["shadow.unknown.detail"]).toBeTruthy();
  });
});
