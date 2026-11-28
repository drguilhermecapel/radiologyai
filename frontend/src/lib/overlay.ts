/**
 * Geometria do visualizador: tudo é guardado em coordenadas de pixel da
 * IMAGEM DE ORIGEM (x = coluna, y = linha), como no contrato ``Primitive``.
 * O quadro recebido pode estar reduzido (``X-Frame-Scale``) e a tela aplica
 * zoom e deslocamento; estas funções fazem as conversões nos dois sentidos.
 */

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Transformação afim sem rotação: tela = origem * k + (ox, oy). */
export interface ViewTransform {
  k: number;
  ox: number;
  oy: number;
}

/** Tamanho da imagem de origem coberta por um quadro reduzido por ``scale``. */
export function sourceSize(frameColumns: number, frameRows: number, scale: number): {
  width: number;
  height: number;
} {
  return { width: frameColumns * scale, height: frameRows * scale };
}

/**
 * Transformação que encaixa a imagem de origem (``srcW`` x ``srcH``) na tela
 * (``viewW`` x ``viewH``), centrada, com ``zoom`` relativo ao encaixe e ``pan`` em px de tela.
 */
export function fitTransform(
  srcW: number,
  srcH: number,
  viewW: number,
  viewH: number,
  zoom = 1,
  pan: Point = { x: 0, y: 0 },
): ViewTransform {
  if (srcW <= 0 || srcH <= 0 || viewW <= 0 || viewH <= 0) {
    return { k: 1, ox: pan.x, oy: pan.y };
  }
  const k = Math.min(viewW / srcW, viewH / srcH) * zoom;
  return {
    k,
    ox: (viewW - srcW * k) / 2 + pan.x,
    oy: (viewH - srcH * k) / 2 + pan.y,
  };
}

export function sourceToScreen(p: Point, t: ViewTransform): Point {
  return { x: p.x * t.k + t.ox, y: p.y * t.k + t.oy };
}

export function screenToSource(p: Point, t: ViewTransform): Point {
  return { x: (p.x - t.ox) / t.k, y: (p.y - t.oy) / t.k };
}

/** Coordenada de origem -> coordenada no quadro reduzido. */
export function sourceToFrame(p: Point, scale: number): Point {
  return { x: p.x / scale, y: p.y / scale };
}

/** Normaliza uma bbox ``[[x0, y0], [x1, y1]]`` (cantos em qualquer ordem) num retângulo. */
export function bboxToRect(xy: readonly (readonly number[])[]): Rect {
  if (xy.length !== 2) throw new RangeError(`bbox exige 2 pontos; recebido ${xy.length}`);
  const [a, b] = xy;
  if (!a || !b || a.length !== 2 || b.length !== 2) {
    throw new RangeError("cada ponto da bbox exige exatamente 2 coordenadas");
  }
  const [ax = 0, ay = 0] = a;
  const [bx = 0, by = 0] = b;
  if (![ax, ay, bx, by].every(Number.isFinite)) throw new RangeError("bbox com coordenada não finita");
  return { x: Math.min(ax, bx), y: Math.min(ay, by), w: Math.abs(bx - ax), h: Math.abs(by - ay) };
}

/**
 * Onde desenhar, na tela, um mapa de calor cuja ``placement`` (bbox na
 * imagem de origem) é ``xy``. O PNG é esticado exatamente sobre esse retângulo.
 */
export function heatmapScreenRect(
  xy: readonly (readonly number[])[],
  t: ViewTransform,
): Rect {
  const r = bboxToRect(xy);
  const p = sourceToScreen({ x: r.x, y: r.y }, t);
  return { x: p.x, y: p.y, w: r.w * t.k, h: r.h * t.k };
}

/** Zoom ancorado num ponto da tela (o ponto sob o cursor fica parado). */
export function zoomAt(
  zoom: number,
  pan: Point,
  factor: number,
  anchor: Point,
  viewW: number,
  viewH: number,
  minZoom = 0.1,
  maxZoom = 40,
): { zoom: number; pan: Point } {
  const next = Math.min(maxZoom, Math.max(minZoom, zoom * factor));
  const real = next / zoom;
  // O centro da tela é a origem do zoom em fitTransform; desloca o pan para manter a âncora.
  const cx = viewW / 2 + pan.x;
  const cy = viewH / 2 + pan.y;
  return {
    zoom: next,
    pan: {
      x: pan.x + (anchor.x - cx) * (1 - real),
      y: pan.y + (anchor.y - cy) * (1 - real),
    },
  };
}

/** Caixa (bbox) desenhada pelo usuário, de tela para origem, recortada aos limites da imagem. */
export function screenBoxToSource(
  a: Point,
  b: Point,
  t: ViewTransform,
  srcW: number,
  srcH: number,
): [[number, number], [number, number]] {
  const clamp = (v: number, hi: number): number => Math.min(hi, Math.max(0, v));
  const pa = screenToSource(a, t);
  const pb = screenToSource(b, t);
  const x0 = clamp(Math.min(pa.x, pb.x), srcW);
  const y0 = clamp(Math.min(pa.y, pb.y), srcH);
  const x1 = clamp(Math.max(pa.x, pb.x), srcW);
  const y1 = clamp(Math.max(pa.y, pb.y), srcH);
  return [
    [Math.round(x0), Math.round(y0)],
    [Math.round(x1), Math.round(y1)],
  ];
}

// --------------------------------------------------------------------------- //
// Cores
// --------------------------------------------------------------------------- //

export type ColormapName = "inferno" | "jet" | "gray";

const INFERNO_STOPS: readonly [number, number, number, number][] = [
  [0.0, 0, 0, 4],
  [0.25, 87, 16, 110],
  [0.5, 188, 55, 84],
  [0.75, 249, 142, 9],
  [1.0, 252, 255, 164],
];

const JET_STOPS: readonly [number, number, number, number][] = [
  [0.0, 0, 0, 131],
  [0.125, 0, 60, 170],
  [0.375, 5, 255, 255],
  [0.625, 255, 255, 0],
  [0.875, 250, 0, 0],
  [1.0, 128, 0, 0],
];

function interpolate(
  stops: readonly [number, number, number, number][],
  t: number,
): [number, number, number] {
  for (let i = 1; i < stops.length; i++) {
    const b = stops[i];
    const a = stops[i - 1];
    if (a !== undefined && b !== undefined && t <= b[0]) {
      const f = (t - a[0]) / (b[0] - a[0] || 1);
      return [
        Math.round(a[1] + (b[1] - a[1]) * f),
        Math.round(a[2] + (b[2] - a[2]) * f),
        Math.round(a[3] + (b[3] - a[3]) * f),
      ];
    }
  }
  const last = stops[stops.length - 1] ?? [1, 255, 255, 255];
  return [last[1], last[2], last[3]];
}

/** Tabela RGB 256x3 do mapa de cores. */
export function colormapTable(name: ColormapName): Uint8Array {
  const out = new Uint8Array(256 * 3);
  for (let v = 0; v < 256; v++) {
    const t = v / 255;
    const rgb: [number, number, number] =
      name === "gray"
        ? [v, v, v]
        : interpolate(name === "jet" ? JET_STOPS : INFERNO_STOPS, t);
    out[v * 3] = rgb[0];
    out[v * 3 + 1] = rgb[1];
    out[v * 3 + 2] = rgb[2];
  }
  return out;
}

/**
 * Colore um mapa 8 bits de um canal (intensidades do PNG do modelo) em RGBA.
 * A opacidade de cada pixel é ``alpha * intensidade/255``: regiões de baixa
 * ativação ficam transparentes e a imagem continua legível.
 */
export function colorizeHeatmap(
  intensities: Uint8Array | Uint8ClampedArray,
  table: Uint8Array,
  alpha: number,
  out: Uint8ClampedArray,
  stride = 1,
): void {
  const n = Math.floor(intensities.length / stride);
  if (out.length !== n * 4) throw new RangeError("tamanho de saída incompatível");
  const a = Math.min(1, Math.max(0, alpha));
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const v = intensities[i * stride] ?? 0;
    out[j] = table[v * 3] ?? 0;
    out[j + 1] = table[v * 3 + 1] ?? 0;
    out[j + 2] = table[v * 3 + 2] ?? 0;
    out[j + 3] = Math.round(a * v);
  }
}

/** Paleta categórica (distinguível, inclusive para daltonismo comum) para rótulos. */
const LABEL_PALETTE: readonly [number, number, number][] = [
  [230, 159, 0],
  [86, 180, 233],
  [0, 158, 115],
  [240, 228, 66],
  [0, 114, 178],
  [213, 94, 0],
  [204, 121, 167],
  [153, 153, 255],
  [120, 200, 80],
  [255, 120, 120],
];

/** Cor determinística de um índice de rótulo (0 = fundo, transparente). */
export function labelColor(index: number): [number, number, number] {
  if (index <= 0) return [0, 0, 0];
  return LABEL_PALETTE[(index - 1) % LABEL_PALETTE.length] ?? [255, 255, 255];
}

/**
 * Pinta um mapa de rótulos em RGBA. Só os índices em ``visible`` aparecem;
 * fundo (0) e rótulos ocultos ficam transparentes.
 */
export function colorizeLabels(
  labels: Uint8Array | Uint16Array,
  visible: ReadonlySet<number>,
  alpha: number,
  out: Uint8ClampedArray,
): void {
  if (out.length !== labels.length * 4) throw new RangeError("tamanho de saída incompatível");
  const a = Math.round(Math.min(1, Math.max(0, alpha)) * 255);
  for (let i = 0, j = 0; i < labels.length; i++, j += 4) {
    const v = labels[i] ?? 0;
    if (v === 0 || !visible.has(v)) {
      out[j + 3] = 0;
      continue;
    }
    const [r, g, b] = labelColor(v);
    out[j] = r;
    out[j + 1] = g;
    out[j + 2] = b;
    out[j + 3] = a;
  }
}

/** Índices de rótulo presentes (≠ 0) num quadro de segmentação. */
export function presentLabels(labels: Uint8Array | Uint16Array): Set<number> {
  const s = new Set<number>();
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i] ?? 0;
    if (v !== 0) s.add(v);
  }
  return s;
}
