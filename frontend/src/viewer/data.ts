/**
 * Carga e cache dos dados do visualizador: quadros da série, quadros de
 * segmentação e mapas de explicação (PNG). Tudo vem do servidor com o token;
 * nada é sintetizado no cliente.
 */
import { fetchBinary } from "../api/client";
import { parseFrame, type Frame } from "../lib/frames";

const MAX_CACHE = 48;
const cache = new Map<string, Promise<Frame>>();

function remember<T>(map: Map<string, Promise<T>>, key: string, make: () => Promise<T>): Promise<T> {
  const hit = map.get(key);
  if (hit) {
    map.delete(key);
    map.set(key, hit);
    return hit;
  }
  const p = make();
  map.set(key, p);
  p.catch(() => map.delete(key));
  while (map.size > MAX_CACHE) {
    const oldest = map.keys().next().value;
    if (oldest === undefined) break;
    map.delete(oldest);
  }
  return p;
}

async function loadFrame(url: string): Promise<Frame> {
  const response = await fetchBinary(url);
  const body = await response.arrayBuffer();
  return parseFrame(response.headers, body);
}

export function seriesFrameUrl(seriesId: string, index: number, maxDim: number): string {
  return `/api/v1/series/${encodeURIComponent(seriesId)}/frames/${index}?max_dim=${maxDim}`;
}

export function segmentationFrameUrl(
  analysisId: string,
  segIndex: number,
  index: number,
  maxDim: number,
): string {
  return `/api/v1/analyses/${encodeURIComponent(analysisId)}/segmentations/${segIndex}/frames/${index}?max_dim=${maxDim}`;
}

export function getSeriesFrame(seriesId: string, index: number, maxDim: number): Promise<Frame> {
  const url = seriesFrameUrl(seriesId, index, maxDim);
  return remember(cache, url, () => loadFrame(url));
}

export function getSegmentationFrame(
  analysisId: string,
  segIndex: number,
  index: number,
  maxDim: number,
): Promise<Frame> {
  const url = segmentationFrameUrl(analysisId, segIndex, index, maxDim);
  return remember(cache, url, () => loadFrame(url));
}

export interface HeatmapBitmap {
  width: number;
  height: number;
  /** Intensidades 0-255 (um canal) lidas do PNG do modelo. */
  intensities: Uint8ClampedArray;
}

const heatmaps = new Map<string, Promise<HeatmapBitmap>>();

/** PNG 8 bits de um canal guardado no repositório endereçado por conteúdo. */
export function getHeatmap(sha256: string): Promise<HeatmapBitmap> {
  return remember(heatmaps, sha256, async () => {
    const response = await fetchBinary(`/api/v1/artifacts/${sha256}`);
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("canvas 2D indisponível");
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    // PNG de um canal decodifica como cinza: R = G = B. Fica o canal R.
    const intensities = new Uint8ClampedArray(canvas.width * canvas.height);
    for (let i = 0; i < intensities.length; i++) intensities[i] = rgba[i * 4] ?? 0;
    return { width: canvas.width, height: canvas.height, intensities };
  });
}
