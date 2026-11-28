/**
 * Decodificação do quadro binário de ``GET /series/{id}/frames/{i}`` e
 * ``GET /analyses/{id}/segmentations/{s}/frames/{i}``.
 *
 * O servidor envia os pixels inteiros (little-endian, linha a linha) e a
 * geometria nos cabeçalhos X-Frame-*. Nada aqui altera os valores.
 */
import type { FrameDtype, StoredPixels } from "./windowLevel";

export interface Frame {
  rows: number;
  columns: number;
  channels: 1 | 3;
  dtype: FrameDtype;
  /** Fator inteiro de redução: coordenada no quadro = coordenada de origem / escala. */
  scale: number;
  index: number;
  count: number;
  rescaleSlope: number;
  rescaleIntercept: number;
  pixels: StoredPixels;
}

export class FrameFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FrameFormatError";
  }
}

/** Leitor mínimo de cabeçalhos (compatível com ``Headers`` e com objetos simples). */
export interface HeaderSource {
  get(name: string): string | null;
}

function intHeader(h: HeaderSource, name: string, min: number): number {
  const raw = h.get(name);
  if (raw === null) throw new FrameFormatError(`cabeçalho ${name} ausente`);
  const v = Number(raw);
  if (!Number.isInteger(v) || v < min) {
    throw new FrameFormatError(`cabeçalho ${name} inválido: ${raw}`);
  }
  return v;
}

function floatHeader(h: HeaderSource, name: string, fallback: number): number {
  const raw = h.get(name);
  if (raw === null) return fallback;
  const v = Number(raw);
  if (!Number.isFinite(v)) throw new FrameFormatError(`cabeçalho ${name} inválido: ${raw}`);
  return v;
}

const BYTES: Record<FrameDtype, number> = { uint8: 1, uint16: 2, int16: 2 };

/** Converte corpo + cabeçalhos num :type:`Frame`, conferindo o tamanho em bytes. */
export function parseFrame(headers: HeaderSource, body: ArrayBuffer): Frame {
  const rows = intHeader(headers, "X-Frame-Rows", 1);
  const columns = intHeader(headers, "X-Frame-Columns", 1);
  const channelsRaw = intHeader(headers, "X-Frame-Channels", 1);
  if (channelsRaw !== 1 && channelsRaw !== 3) {
    throw new FrameFormatError(`X-Frame-Channels deve ser 1 ou 3; recebido ${channelsRaw}`);
  }
  const channels: 1 | 3 = channelsRaw;
  const dtypeRaw = headers.get("X-Frame-Dtype");
  if (dtypeRaw !== "uint8" && dtypeRaw !== "uint16" && dtypeRaw !== "int16") {
    throw new FrameFormatError(`X-Frame-Dtype desconhecido: ${String(dtypeRaw)}`);
  }
  const dtype: FrameDtype = dtypeRaw;
  const scale = intHeader(headers, "X-Frame-Scale", 1);
  const index = intHeader(headers, "X-Frame-Index", 0);
  const count = intHeader(headers, "X-Frame-Count", 1);
  const expected = rows * columns * channels * BYTES[dtype];
  if (body.byteLength !== expected) {
    throw new FrameFormatError(
      `corpo com ${body.byteLength} bytes; esperado ${expected} (${rows}x${columns}x${channels} ${dtype})`,
    );
  }
  let pixels: StoredPixels;
  if (dtype === "uint8") {
    pixels = new Uint8Array(body);
  } else {
    // Cópia explícita little-endian: independente da ordem de bytes da máquina.
    const view = new DataView(body);
    const n = rows * columns * channels;
    pixels = dtype === "uint16" ? new Uint16Array(n) : new Int16Array(n);
    for (let i = 0; i < n; i++) {
      pixels[i] = dtype === "uint16" ? view.getUint16(i * 2, true) : view.getInt16(i * 2, true);
    }
  }
  return {
    rows,
    columns,
    channels,
    dtype,
    scale,
    index,
    count,
    rescaleSlope: floatHeader(headers, "X-Rescale-Slope", 1),
    rescaleIntercept: floatHeader(headers, "X-Rescale-Intercept", 0),
    pixels,
  };
}

/** Pinta um quadro RGB (uint8, 3 canais) em RGBA, sem janela. */
export function rgbToRgba(pixels: Uint8Array, out: Uint8ClampedArray): void {
  const n = pixels.length / 3;
  if (out.length !== n * 4) throw new RangeError("tamanho de saída incompatível");
  for (let i = 0, j = 0, k = 0; i < n; i++, j += 4, k += 3) {
    out[j] = pixels[k] ?? 0;
    out[j + 1] = pixels[k + 1] ?? 0;
    out[j + 2] = pixels[k + 2] ?? 0;
    out[j + 3] = 255;
  }
}
