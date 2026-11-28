import { describe, expect, it } from "vitest";

import {
  applyLut,
  autoWindow,
  buildLut,
  dragWindow,
  fullRangeWindow,
  normalizeWindow,
  rangeFromPair,
  storedRange,
  voiLinear,
  windowFromPair,
} from "./windowLevel";

describe("voiLinear (DICOM PS3.3 C.11.2.1.2.1)", () => {
  const w = { center: 40, width: 400 };
  it("satura abaixo e acima da janela", () => {
    expect(voiLinear(-1000, w)).toBe(0);
    expect(voiLinear(3000, w)).toBe(255);
    // limites exatos da fórmula: x <= c - 0,5 - (w-1)/2 -> 0
    expect(voiLinear(40 - 0.5 - 399 / 2, w)).toBe(0);
    expect(voiLinear(40 - 0.5 + 399 / 2 + 1e-9, w)).toBe(255);
  });
  it("é monotônica e passa pelo meio no centro", () => {
    let prev = -1;
    for (let x = -200; x <= 260; x += 5) {
      const v = voiLinear(x, w);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
    expect(Math.abs(voiLinear(40, w) - 128)).toBeLessThanOrEqual(1);
  });
  it("largura 1 vira limiar binário", () => {
    expect(voiLinear(0, { center: 10, width: 1 })).toBe(0);
    expect(voiLinear(20, { center: 10, width: 1 })).toBe(255);
  });
});

describe("normalizeWindow", () => {
  it("força largura >= 1 e valores finitos", () => {
    expect(normalizeWindow({ center: Number.NaN, width: -5 })).toEqual({ center: 0, width: 1 });
    expect(normalizeWindow({ center: 3, width: 0.2 })).toEqual({ center: 3, width: 1 });
  });
});

describe("buildLut / applyLut", () => {
  it("aplica slope/intercept (HU) antes da janela", () => {
    // armazenado 0..4095 com intercept -1024 -> HU -1024..3071
    const lut = buildLut(0, 4095, { center: 40, width: 400 }, 1, -1024);
    expect(lut.table.length).toBe(4096);
    expect(lut.table[0]).toBe(0); // -1024 HU
    expect(lut.table[1024 + 40]).toBe(voiLinear(40, { center: 40, width: 400 }));
    expect(lut.table[4095]).toBe(255);
  });
  it("inverte quando pedido", () => {
    const lut = buildLut(0, 10, { center: 5, width: 10 }, 1, 0, true);
    expect(lut.table[0]).toBe(255);
    expect(lut.table[10]).toBe(0);
  });
  it("aceita faixa com negativos (int16) e recorta índices fora da faixa", () => {
    const px = new Int16Array([-2000, -1000, 0, 1000, 5000]);
    const lut = buildLut(-1000, 1000, { center: 0, width: 2000 });
    const out = new Uint8ClampedArray(px.length * 4);
    applyLut(px, lut, out);
    const grays = [0, 1, 2, 3, 4].map((i) => out[i * 4]);
    expect(grays[0]).toBe(0);
    expect(grays[1]).toBe(0);
    expect(Math.abs((grays[2] ?? 0) - 128)).toBeLessThanOrEqual(1);
    expect(grays[3]).toBe(255);
    expect(grays[4]).toBe(255);
    for (let i = 0; i < px.length; i++) {
      expect(out[i * 4]).toBe(out[i * 4 + 1]);
      expect(out[i * 4 + 3]).toBe(255);
    }
  });
  it("rejeita faixa inválida e saída de tamanho errado", () => {
    expect(() => buildLut(10, 0, { center: 0, width: 1 })).toThrow(RangeError);
    expect(() => buildLut(0, 70000, { center: 0, width: 1 })).toThrow(RangeError);
    const lut = buildLut(0, 1, { center: 0, width: 1 });
    expect(() => applyLut(new Uint8Array(2), lut, new Uint8ClampedArray(4))).toThrow(RangeError);
  });
});

describe("storedRange / autoWindow / fullRangeWindow", () => {
  it("acha mínimo e máximo", () => {
    expect(storedRange(new Int16Array([5, -3, 9, 0]))).toEqual([-3, 9]);
    expect(storedRange(new Uint16Array([]))).toEqual([0, 0]);
  });
  it("janela automática cobre o miolo do histograma, ignorando extremos raros", () => {
    const px = new Uint16Array(1000);
    for (let i = 0; i < px.length; i++) px[i] = 100 + (i % 100);
    px[0] = 0; // outlier
    px[1] = 4000; // outlier
    const w = autoWindow(px, 1, 0, 0.01, 0.99);
    expect(w.center - w.width / 2).toBeGreaterThanOrEqual(99);
    expect(w.center + w.width / 2).toBeLessThanOrEqual(200);
  });
  it("imagem constante vira largura 1", () => {
    expect(autoWindow(new Uint8Array([7, 7, 7]), 2, 1)).toEqual({ center: 15, width: 1 });
  });
  it("faixa completa", () => {
    expect(fullRangeWindow(-1024, 3071)).toEqual({ center: 1023.5, width: 4095 });
  });
});

describe("dragWindow e pares do servidor", () => {
  it("horizontal muda largura, vertical muda centro", () => {
    expect(dragWindow({ center: 40, width: 400 }, 10, -20, 2)).toEqual({ center: 0, width: 420 });
    expect(dragWindow({ center: 40, width: 5 }, -100, 0, 1).width).toBe(1);
  });
  it("aceita só pares finitos", () => {
    expect(windowFromPair([40, 400])).toEqual({ center: 40, width: 400 });
    expect(windowFromPair(null)).toBeNull();
    expect(windowFromPair([1])).toBeNull();
    expect(windowFromPair([Number.NaN, 2])).toBeNull();
    expect(rangeFromPair([10, -10])).toEqual([-10, 10]);
    expect(rangeFromPair(undefined)).toBeNull();
  });
});
