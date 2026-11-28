import { describe, expect, it } from "vitest";

import {
  bboxToRect,
  colorizeHeatmap,
  colorizeLabels,
  colormapTable,
  fitTransform,
  heatmapScreenRect,
  labelColor,
  presentLabels,
  screenBoxToSource,
  screenToSource,
  sourceSize,
  sourceToFrame,
  sourceToScreen,
  zoomAt,
} from "./overlay";

describe("fitTransform", () => {
  it("encaixa centralizado preservando proporção", () => {
    const t = fitTransform(1000, 500, 400, 400);
    expect(t.k).toBeCloseTo(0.4);
    expect(t.ox).toBeCloseTo(0);
    expect(t.oy).toBeCloseTo(100); // (400 - 500*0.4)/2
  });
  it("aplica zoom e pan", () => {
    const t = fitTransform(100, 100, 200, 200, 2, { x: 10, y: -5 });
    expect(t.k).toBeCloseTo(4);
    expect(t.ox).toBeCloseTo((200 - 400) / 2 + 10);
    expect(t.oy).toBeCloseTo((200 - 400) / 2 - 5);
  });
  it("ida e volta tela <-> origem é identidade", () => {
    const t = fitTransform(2048, 2500, 777, 555, 1.7, { x: 13, y: 21 });
    const p = { x: 123.4, y: 987.6 };
    const back = screenToSource(sourceToScreen(p, t), t);
    expect(back.x).toBeCloseTo(p.x, 9);
    expect(back.y).toBeCloseTo(p.y, 9);
  });
  it("dimensões degeneradas não dividem por zero", () => {
    expect(fitTransform(0, 10, 100, 100).k).toBe(1);
  });
});

describe("quadro reduzido (X-Frame-Scale)", () => {
  it("tamanho de origem e conversão", () => {
    expect(sourceSize(512, 400, 4)).toEqual({ width: 2048, height: 1600 });
    expect(sourceToFrame({ x: 400, y: 80 }, 4)).toEqual({ x: 100, y: 20 });
  });
});

describe("posicionamento do mapa de calor pela bbox", () => {
  it("normaliza cantos em qualquer ordem", () => {
    expect(bboxToRect([
      [300, 50],
      [100, 250],
    ])).toEqual({ x: 100, y: 50, w: 200, h: 200 });
  });
  it("rejeita bbox malformada", () => {
    expect(() => bboxToRect([[1, 2]])).toThrow(RangeError);
    expect(() => bboxToRect([[1, 2, 3], [4, 5]])).toThrow(RangeError);
    expect(() => bboxToRect([[1, Number.NaN], [4, 5]])).toThrow(RangeError);
  });
  it("o retângulo na tela acompanha a transformação (imagem inteira e recorte)", () => {
    const t = fitTransform(2000, 2000, 500, 500); // k = 0.25
    expect(heatmapScreenRect([
      [0, 0],
      [2000, 2000],
    ], t)).toEqual({ x: 0, y: 0, w: 500, h: 500 });
    const r = heatmapScreenRect([
      [400, 800],
      [1200, 1600],
    ], t);
    expect(r).toEqual({ x: 100, y: 200, w: 200, h: 200 });
    // com zoom 2 e pan: canto superior esquerdo segue sourceToScreen
    const t2 = fitTransform(2000, 2000, 500, 500, 2, { x: 30, y: -10 });
    const r2 = heatmapScreenRect([
      [400, 800],
      [1200, 1600],
    ], t2);
    const corner = sourceToScreen({ x: 400, y: 800 }, t2);
    expect(r2.x).toBeCloseTo(corner.x);
    expect(r2.y).toBeCloseTo(corner.y);
    expect(r2.w).toBeCloseTo(800 * t2.k);
  });
});

describe("zoomAt", () => {
  it("mantém parado o ponto sob o cursor", () => {
    const viewW = 600;
    const viewH = 400;
    const srcW = 1000;
    const srcH = 800;
    const zoom = 1;
    const pan = { x: 0, y: 0 };
    const anchor = { x: 150, y: 90 };
    const before = screenToSource(anchor, fitTransform(srcW, srcH, viewW, viewH, zoom, pan));
    const next = zoomAt(zoom, pan, 2.5, anchor, viewW, viewH);
    const after = screenToSource(anchor, fitTransform(srcW, srcH, viewW, viewH, next.zoom, next.pan));
    expect(after.x).toBeCloseTo(before.x, 6);
    expect(after.y).toBeCloseTo(before.y, 6);
  });
  it("respeita limites", () => {
    expect(zoomAt(30, { x: 0, y: 0 }, 10, { x: 0, y: 0 }, 100, 100).zoom).toBe(40);
    expect(zoomAt(0.2, { x: 0, y: 0 }, 0.1, { x: 0, y: 0 }, 100, 100).zoom).toBe(0.1);
  });
});

describe("caixa-prompt desenhada", () => {
  it("converte para origem, ordena e recorta aos limites", () => {
    const t = fitTransform(1000, 1000, 500, 500); // k = 0.5
    expect(screenBoxToSource({ x: 300, y: 50 }, { x: 100, y: 600 }, t, 1000, 1000)).toEqual([
      [200, 100],
      [600, 1000],
    ]);
  });
});

describe("cores", () => {
  it("tabela de 256 cores; gray é identidade", () => {
    const g = colormapTable("gray");
    expect(g.length).toBe(768);
    expect([g[3 * 77], g[3 * 77 + 1], g[3 * 77 + 2]]).toEqual([77, 77, 77]);
    const inf = colormapTable("inferno");
    expect([inf[0], inf[1], inf[2]]).toEqual([0, 0, 4]);
    expect([inf[765], inf[766], inf[767]]).toEqual([252, 255, 164]);
  });
  it("opacidade do mapa proporcional à intensidade e ao alfa", () => {
    const out = new Uint8ClampedArray(8);
    colorizeHeatmap(new Uint8Array([0, 255]), colormapTable("gray"), 0.5, out);
    expect(out[3]).toBe(0);
    expect(out[7]).toBe(128);
    expect(() => colorizeHeatmap(new Uint8Array(2), colormapTable("gray"), 1, new Uint8ClampedArray(4))).toThrow();
  });
  it("rótulos: fundo e ocultos transparentes, cor determinística", () => {
    const labels = new Uint8Array([0, 1, 2, 3]);
    const out = new Uint8ClampedArray(16);
    colorizeLabels(labels, new Set([1, 3]), 1, out);
    expect(out[3]).toBe(0);
    expect(Array.from(out.slice(4, 8))).toEqual([...labelColor(1), 255]);
    expect(out[11]).toBe(0);
    expect(out[15]).toBe(255);
    expect(labelColor(0)).toEqual([0, 0, 0]);
    expect(labelColor(11)).toEqual(labelColor(1));
    expect([...presentLabels(new Uint16Array([0, 5, 5, 7]))].sort()).toEqual([5, 7]);
  });
});
