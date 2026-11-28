import { describe, expect, it } from "vitest";

import { FrameFormatError, parseFrame, rgbToRgba } from "./frames";

function headers(h: Record<string, string>): { get(name: string): string | null } {
  return { get: (n) => h[n] ?? null };
}

const base = {
  "X-Frame-Rows": "2",
  "X-Frame-Columns": "2",
  "X-Frame-Channels": "1",
  "X-Frame-Dtype": "int16",
  "X-Frame-Scale": "4",
  "X-Frame-Index": "0",
  "X-Frame-Count": "1",
  "X-Rescale-Slope": "1",
  "X-Rescale-Intercept": "-1024",
};

describe("parseFrame", () => {
  it("lê int16 little-endian com geometria e rescale", () => {
    const buf = new ArrayBuffer(8);
    const v = new DataView(buf);
    [-1000, 0, 1, 32767].forEach((x, i) => v.setInt16(i * 2, x, true));
    const f = parseFrame(headers(base), buf);
    expect(Array.from(f.pixels)).toEqual([-1000, 0, 1, 32767]);
    expect(f.scale).toBe(4);
    expect(f.rescaleIntercept).toBe(-1024);
    expect(f.pixels).toBeInstanceOf(Int16Array);
  });
  it("uint16 e uint8", () => {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setUint16(6, 65535, true);
    expect(parseFrame(headers({ ...base, "X-Frame-Dtype": "uint16" }), buf).pixels[3]).toBe(65535);
    const f8 = parseFrame(headers({ ...base, "X-Frame-Dtype": "uint8" }), new Uint8Array([1, 2, 3, 4]).buffer);
    expect(Array.from(f8.pixels)).toEqual([1, 2, 3, 4]);
  });
  it("falha alto com cabeçalho ausente, dtype desconhecido ou tamanho errado", () => {
    const { ["X-Frame-Rows"]: _omit, ...noRows } = base;
    expect(() => parseFrame(headers(noRows), new ArrayBuffer(8))).toThrow(FrameFormatError);
    expect(() => parseFrame(headers({ ...base, "X-Frame-Dtype": "float32" }), new ArrayBuffer(16))).toThrow(
      FrameFormatError,
    );
    expect(() => parseFrame(headers(base), new ArrayBuffer(6))).toThrow(FrameFormatError);
    expect(() => parseFrame(headers({ ...base, "X-Frame-Channels": "2" }), new ArrayBuffer(16))).toThrow(
      FrameFormatError,
    );
  });
  it("slope/intercept ausentes valem 1/0", () => {
    const { ["X-Rescale-Slope"]: _s, ["X-Rescale-Intercept"]: _i, ...h } = base;
    const f = parseFrame(headers(h), new ArrayBuffer(8));
    expect([f.rescaleSlope, f.rescaleIntercept]).toEqual([1, 0]);
  });
});

describe("rgbToRgba", () => {
  it("copia canais e torna opaco", () => {
    const out = new Uint8ClampedArray(8);
    rgbToRgba(new Uint8Array([1, 2, 3, 4, 5, 6]), out);
    expect(Array.from(out)).toEqual([1, 2, 3, 255, 4, 5, 6, 255]);
  });
});
