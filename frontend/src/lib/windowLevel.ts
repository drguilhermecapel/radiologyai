/**
 * Janela/nível (window/level) aplicada no cliente sobre os quadros inteiros do servidor.
 *
 * Segue a função linear VOI LUT do DICOM PS3.3 C.11.2.1.2.1:
 *   x <= c - 0,5 - (w - 1)/2  -> 0
 *   x >  c - 0,5 + (w - 1)/2  -> 255
 *   senão ((x - (c - 0,5)) / (w - 1) + 0,5) * 255
 * onde x é o valor FÍSICO (armazenado * slope + intercept; HU na TC).
 */

export type FrameDtype = "uint16" | "int16" | "uint8";

export interface WindowLevel {
  /** Centro da janela, em unidades físicas. */
  center: number;
  /** Largura da janela (>= 1), em unidades físicas. */
  width: number;
}

export interface WindowPreset {
  id: string;
  /** Chave i18n do nome do preset. */
  labelKey: string;
  window: WindowLevel;
}

/** Presets clássicos de TC (HU). Valores de referência de livros-texto de radiologia. */
export const CT_PRESETS: readonly WindowPreset[] = [
  { id: "lung", labelKey: "viewer.preset.lung", window: { center: -600, width: 1500 } },
  { id: "mediastinum", labelKey: "viewer.preset.mediastinum", window: { center: 40, width: 400 } },
  { id: "abdomen", labelKey: "viewer.preset.abdomen", window: { center: 40, width: 350 } },
  { id: "liver", labelKey: "viewer.preset.liver", window: { center: 60, width: 160 } },
  { id: "bone", labelKey: "viewer.preset.bone", window: { center: 400, width: 1800 } },
  { id: "brain", labelKey: "viewer.preset.brain", window: { center: 40, width: 80 } },
  { id: "subdural", labelKey: "viewer.preset.subdural", window: { center: 75, width: 215 } },
  { id: "stroke", labelKey: "viewer.preset.stroke", window: { center: 32, width: 8 } },
];

/** Garante largura mínima 1 e valores finitos. */
export function normalizeWindow(w: WindowLevel): WindowLevel {
  const center = Number.isFinite(w.center) ? w.center : 0;
  const width = Number.isFinite(w.width) ? Math.max(1, w.width) : 1;
  return { center, width };
}

/** Valor de exibição (0-255) de UM valor físico pela função linear do DICOM. */
export function voiLinear(x: number, window: WindowLevel): number {
  const { center: c, width: w } = normalizeWindow(window);
  const lower = c - 0.5 - (w - 1) / 2;
  const upper = c - 0.5 + (w - 1) / 2;
  if (x <= lower) return 0;
  if (x > upper) return 255;
  if (w === 1) return 255;
  const y = ((x - (c - 0.5)) / (w - 1) + 0.5) * 255;
  return Math.min(255, Math.max(0, Math.round(y)));
}

export interface Lut {
  /** Menor valor ARMAZENADO coberto (índice 0 da tabela). */
  minStored: number;
  /** Tabela: valor armazenado - minStored -> intensidade 0..255. */
  table: Uint8Array;
}

/**
 * Tabela de consulta indexada pelo valor armazenado, cobrindo [minStored, maxStored].
 *
 * Construir a tabela uma vez por mudança de janela torna a pintura O(pixels)
 * sem aritmética de ponto flutuante por pixel.
 */
export function buildLut(
  minStored: number,
  maxStored: number,
  window: WindowLevel,
  slope = 1,
  intercept = 0,
  invert = false,
): Lut {
  if (!Number.isInteger(minStored) || !Number.isInteger(maxStored) || maxStored < minStored) {
    throw new RangeError(`faixa armazenada inválida: [${minStored}, ${maxStored}]`);
  }
  const size = maxStored - minStored + 1;
  if (size > 65536) throw new RangeError(`faixa armazenada grande demais: ${size}`);
  const table = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    const physical = (minStored + i) * slope + intercept;
    const v = voiLinear(physical, window);
    table[i] = invert ? 255 - v : v;
  }
  return { minStored, table };
}

export type StoredPixels = Uint16Array | Int16Array | Uint8Array;

/** Menor e maior valor armazenado do quadro. */
export function storedRange(pixels: StoredPixels): [number, number] {
  if (pixels.length === 0) return [0, 0];
  let lo = pixels[0] ?? 0;
  let hi = lo;
  for (let i = 1; i < pixels.length; i++) {
    const v = pixels[i] ?? 0;
    if (v < lo) lo = v;
    else if (v > hi) hi = v;
  }
  return [lo, hi];
}

/**
 * Pinta um quadro monocromático em RGBA (opaco) usando a LUT.
 * ``out`` deve ter 4 * pixels.length posições.
 */
export function applyLut(pixels: StoredPixels, lut: Lut, out: Uint8ClampedArray): void {
  if (out.length !== pixels.length * 4) {
    throw new RangeError(`saída com ${out.length} posições; esperado ${pixels.length * 4}`);
  }
  const { table, minStored } = lut;
  const last = table.length - 1;
  for (let i = 0, j = 0; i < pixels.length; i++, j += 4) {
    let k = (pixels[i] ?? 0) - minStored;
    if (k < 0) k = 0;
    else if (k > last) k = last;
    const v = table[k] ?? 0;
    out[j] = v;
    out[j + 1] = v;
    out[j + 2] = v;
    out[j + 3] = 255;
  }
}

/**
 * Janela automática por percentis do histograma dos valores armazenados do quadro.
 * Só usa os pixels reais; não inventa nada sobre o paciente.
 */
export function autoWindow(
  pixels: StoredPixels,
  slope = 1,
  intercept = 0,
  lowPct = 0.005,
  highPct = 0.995,
): WindowLevel {
  const [lo, hi] = storedRange(pixels);
  if (hi === lo) {
    return { center: lo * slope + intercept, width: 1 };
  }
  const hist = new Uint32Array(hi - lo + 1);
  for (let i = 0; i < pixels.length; i++) {
    const idx = (pixels[i] ?? lo) - lo;
    hist[idx] = (hist[idx] ?? 0) + 1;
  }
  const n = pixels.length;
  const lowCount = lowPct * n;
  const highCount = highPct * n;
  let acc = 0;
  let pLo = lo;
  let pHi = hi;
  let foundLo = false;
  for (let i = 0; i < hist.length; i++) {
    acc += hist[i] ?? 0;
    if (!foundLo && acc >= lowCount) {
      pLo = lo + i;
      foundLo = true;
    }
    if (acc >= highCount) {
      pHi = lo + i;
      break;
    }
  }
  const a = pLo * slope + intercept;
  const b = pHi * slope + intercept;
  const low = Math.min(a, b);
  const high = Math.max(a, b);
  return { center: (low + high) / 2, width: Math.max(1, high - low) };
}

/** Janela que cobre toda a faixa física [lo, hi]. */
export function fullRangeWindow(lo: number, hi: number): WindowLevel {
  return { center: (lo + hi) / 2, width: Math.max(1, hi - lo) };
}

/** Ajuste por arrasto do mouse: horizontal muda a largura, vertical o centro. */
export function dragWindow(
  start: WindowLevel,
  dx: number,
  dy: number,
  sensitivity: number,
): WindowLevel {
  return normalizeWindow({
    center: start.center + dy * sensitivity,
    width: start.width + dx * sensitivity,
  });
}

/**
 * Converte o par ``[centro, largura]`` do servidor (``SeriesMeta.window``) em
 * janela; ``null`` se ausente ou malformado (aí vale a janela automática).
 */
export function windowFromPair(pair: readonly number[] | null | undefined): WindowLevel | null {
  if (!pair || pair.length !== 2) return null;
  const [center, width] = pair;
  if (center === undefined || width === undefined || !Number.isFinite(center) || !Number.isFinite(width)) {
    return null;
  }
  return normalizeWindow({ center, width });
}

/** Faixa física ``[mín, máx]`` do servidor (``SeriesMeta.value_range``), ou ``null``. */
export function rangeFromPair(pair: readonly number[] | null | undefined): [number, number] | null {
  if (!pair || pair.length !== 2) return null;
  const [lo, hi] = pair;
  if (lo === undefined || hi === undefined || !Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  return lo <= hi ? [lo, hi] : [hi, lo];
}
