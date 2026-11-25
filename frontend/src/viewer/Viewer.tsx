/**
 * Visualizador em canvas: quadros inteiros do servidor, janela/nível no
 * cliente, zoom/deslocamento, rolagem de cortes/quadros e sobreposições
 * (mapa de calor posicionado pela bbox, mapas de rótulos com legenda,
 * geometria de medidas em SVG e caixas-prompt desenhadas pelo usuário).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent, ReactNode, WheelEvent } from "react";

import { api, unwrap, type Schemas } from "../api/client";
import { ErrorBox } from "../components/common";
import { useI18n } from "../i18n/i18n";
import type { MessageKey } from "../i18n/messages";
import { rgbToRgba, type Frame } from "../lib/frames";
import {
  colorizeHeatmap,
  colorizeLabels,
  colormapTable,
  fitTransform,
  heatmapScreenRect,
  labelColor,
  screenBoxToSource,
  sourceToScreen,
  zoomAt,
  type ColormapName,
  type Point,
  type ViewTransform,
} from "../lib/overlay";
import { useAsync } from "../lib/useAsync";
import {
  applyLut,
  autoWindow,
  buildLut,
  CT_PRESETS,
  dragWindow,
  fullRangeWindow,
  normalizeWindow,
  storedRange,
  type WindowLevel,
} from "../lib/windowLevel";
import { getHeatmap, getSegmentationFrame, getSeriesFrame, type HeatmapBitmap } from "./data";

type Primitive = Schemas["Primitive"];
type SeriesOut = Schemas["SeriesOut"];
type AnalysisOut = Schemas["AnalysisOut"];
type Tool = "pan" | "wl" | "box";

export interface ViewerProps {
  series: readonly SeriesOut[];
  seriesId: string;
  onSeriesChange: (id: string) => void;
  analysis: AnalysisOut | null;
  activeExplanationId: string | null;
  onExplanationChange: (id: string | null) => void;
  prompts: readonly Primitive[];
  /** ``null`` quando o usuário não pode pedir análise (sem ferramenta de caixa). */
  onPromptsChange: ((p: Primitive[]) => void) | null;
}

const MAX_DIM = 1024;

function toCanvas(width: number, height: number, rgba: Uint8ClampedArray): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("canvas 2D indisponível");
  const img = new ImageData(width, height);
  img.data.set(rgba);
  ctx.putImageData(img, 0, 0);
  return c;
}

export function Viewer(props: ViewerProps): ReactNode {
  const { series, seriesId, onSeriesChange, analysis, activeExplanationId, onExplanationChange } = props;
  const { t } = useI18n();
  const current = series.find((s) => s.id === seriesId) ?? series[0];
  const meta = useAsync(
    async () =>
      current
        ? unwrap(await api.GET("/api/v1/series/{series_id}", { params: { path: { series_id: current.id } } }))
        : null,
    [current?.id],
  );
  const count = Math.max(1, current?.n_frames ?? 1);
  const [frameIndex, setFrameIndex] = useState(0);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [frameError, setFrameError] = useState<Error | null>(null);
  const [wl, setWl] = useState<WindowLevel | null>(null);
  const [presetId, setPresetId] = useState("auto");
  const [invert, setInvert] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [tool, setTool] = useState<Tool>("pan");
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [alpha, setAlpha] = useState(0.55);
  const [cmap, setCmap] = useState<ColormapName>("inferno");
  const [visibleSegs, setVisibleSegs] = useState<Set<number>>(new Set());
  const [hiddenLabels, setHiddenLabels] = useState<Record<number, Set<number>>>({});
  const [showMeasurements, setShowMeasurements] = useState(true);
  const [segFrames, setSegFrames] = useState<Map<number, Frame>>(new Map());
  const [heatmap, setHeatmap] = useState<HeatmapBitmap | null>(null);
  const [size, setSize] = useState({ w: 640, h: 480 });
  const [drag, setDrag] = useState<{ start: Point; now: Point; wl0: WindowLevel | null; pan0: Point } | null>(null);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Reinicia o estado ao trocar de série.
  useEffect(() => {
    setFrameIndex(0);
    setFrame(null);
    setWl(null);
    setPresetId("auto");
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [current?.id]);

  // Tamanho responsivo.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setSize({ w: Math.max(200, Math.floor(r.width)), h: Math.max(240, Math.floor(r.height)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Quadro da série.
  useEffect(() => {
    if (!current) return;
    let alive = true;
    setFrameError(null);
    getSeriesFrame(current.id, frameIndex, MAX_DIM)
      .then((f) => {
        if (alive) setFrame(f);
      })
      .catch((err: unknown) => {
        if (alive) setFrameError(err instanceof Error ? err : new Error(String(err)));
      });
    return () => {
      alive = false;
    };
  }, [current?.id, frameIndex]);

  // Janela inicial: a do DICOM, se houver; senão automática pelos pixels reais.
  useEffect(() => {
    if (!frame || wl !== null || frame.channels === 3) return;
    const dicomWindow = meta.data?.window;
    if (dicomWindow && meta.data) {
      setWl(normalizeWindow({ center: dicomWindow[0], width: dicomWindow[1] }));
      setPresetId("dicom");
    } else {
      setWl(autoWindow(frame.pixels, frame.rescaleSlope, frame.rescaleIntercept));
      setPresetId("auto");
    }
  }, [frame, wl, meta.data]);

  // Entradas desta série na análise.
  const imageIdx = useMemo(
    () => analysis?.inputs.find((i) => i.series_id === current?.id)?.image_idx ?? null,
    [analysis, current?.id],
  );
  const result = analysis?.result ?? null;
  const explanations = useMemo(
    () => (result && imageIdx !== null ? result.explanations.filter((e) => e.image_idx === imageIdx) : []),
    [result, imageIdx],
  );
  const segmentations = useMemo(
    () =>
      result && imageIdx !== null
        ? result.segmentations.map((s, i) => ({ s, i })).filter(({ s }) => s.image_idx === imageIdx)
        : [],
    [result, imageIdx],
  );
  const measurements = useMemo(
    () => (result && imageIdx !== null ? result.measurements.filter((m) => m.image_idx === imageIdx) : []),
    [result, imageIdx],
  );
  const activeExplanation = explanations.find((e) => e.id === activeExplanationId) ?? null;

  // Mapa de calor ativo.
  useEffect(() => {
    if (!activeExplanation || !showHeatmap) {
      setHeatmap(null);
      return;
    }
    let alive = true;
    getHeatmap(activeExplanation.map.sha256)
      .then((h) => {
        if (alive) setHeatmap(h);
      })
      .catch(() => {
        if (alive) setHeatmap(null);
      });
    return () => {
      alive = false;
    };
  }, [activeExplanation, showHeatmap]);

  // Quadros de segmentação visíveis.
  useEffect(() => {
    if (!analysis || visibleSegs.size === 0) {
      setSegFrames(new Map());
      return;
    }
    let alive = true;
    const ids = [...visibleSegs];
    Promise.all(
      ids.map((i) =>
        getSegmentationFrame(analysis.id, i, frameIndex, MAX_DIM).then(
          (f) => [i, f] as const,
          () => null,
        ),
      ),
    ).then((pairs) => {
      if (!alive) return;
      const m = new Map<number, Frame>();
      for (const p of pairs) if (p) m.set(p[0], p[1]);
      setSegFrames(m);
    });
    return () => {
      alive = false;
    };
  }, [analysis, visibleSegs, frameIndex]);

  // Imagem-base janelada (no tamanho do quadro).
  const base = useMemo(() => {
    if (!frame) return null;
    const rgba = new Uint8ClampedArray(frame.rows * frame.columns * 4);
    if (frame.channels === 3) {
      rgbToRgba(frame.pixels as Uint8Array, rgba);
    } else {
      if (!wl) return null;
      const [lo, hi] = storedRange(frame.pixels);
      applyLut(frame.pixels, buildLut(lo, hi, wl, frame.rescaleSlope, frame.rescaleIntercept, invert), rgba);
    }
    return toCanvas(frame.columns, frame.rows, rgba);
  }, [frame, wl, invert]);

  const srcW = frame ? frame.columns * frame.scale : 1;
  const srcH = frame ? frame.rows * frame.scale : 1;
  const transform: ViewTransform = useMemo(
    () => fitTransform(srcW, srcH, size.w, size.h, zoom, pan),
    [srcW, srcH, size.w, size.h, zoom, pan],
  );

  const heatCanvas = useMemo(() => {
    if (!heatmap) return null;
    const rgba = new Uint8ClampedArray(heatmap.width * heatmap.height * 4);
    colorizeHeatmap(heatmap.intensities, colormapTable(cmap), alpha, rgba);
    return toCanvas(heatmap.width, heatmap.height, rgba);
  }, [heatmap, cmap, alpha]);

  const segCanvases = useMemo(() => {
    const out: { canvas: HTMLCanvasElement; frame: Frame }[] = [];
    for (const [i, f] of segFrames) {
      if (f.channels !== 1 || f.dtype === "int16") continue;
      const all = new Set<number>(Object.keys(result?.segmentations[i]?.labels ?? {}).map(Number));
      const hidden = hiddenLabels[i] ?? new Set<number>();
      const visible = new Set([...all].filter((v) => !hidden.has(v)));
      const rgba = new Uint8ClampedArray(f.rows * f.columns * 4);
      colorizeLabels(f.pixels as Uint8Array | Uint16Array, visible, 0.45, rgba);
      out.push({ canvas: toCanvas(f.columns, f.rows, rgba), frame: f });
    }
    return out;
  }, [segFrames, hiddenLabels, result]);

  const heatmapOnFrame =
    activeExplanation !== null && (count === 1 || activeExplanation.placement.frame === frameIndex);

  // Pintura.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size.w, size.h);
    if (!base || !frame) return;
    ctx.imageSmoothingEnabled = transform.k * frame.scale < 2;
    ctx.drawImage(base, transform.ox, transform.oy, srcW * transform.k, srcH * transform.k);
    ctx.imageSmoothingEnabled = false;
    for (const { canvas: sc, frame: sf } of segCanvases) {
      ctx.drawImage(
        sc,
        transform.ox,
        transform.oy,
        sf.columns * sf.scale * transform.k,
        sf.rows * sf.scale * transform.k,
      );
    }
    if (heatCanvas && activeExplanation && heatmapOnFrame) {
      ctx.imageSmoothingEnabled = true;
      const r = heatmapScreenRect(activeExplanation.placement.xy, transform);
      ctx.drawImage(heatCanvas, r.x, r.y, r.w, r.h);
    }
  }, [base, frame, transform, size, srcW, srcH, segCanvases, heatCanvas, activeExplanation, heatmapOnFrame]);

  // Interação.
  const local = (e: { clientX: number; clientY: number }): Point => {
    const r = canvasRef.current?.getBoundingClientRect();
    return { x: e.clientX - (r?.left ?? 0), y: e.clientY - (r?.top ?? 0) };
  };
  const onWheel = (e: WheelEvent): void => {
    const next = zoomAt(zoom, pan, e.deltaY < 0 ? 1.15 : 1 / 1.15, local(e), size.w, size.h);
    setZoom(next.zoom);
    setPan(next.pan);
  };
  const onPointerDown = (e: PointerEvent): void => {
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const p = local(e);
    setDrag({ start: p, now: p, wl0: wl, pan0: pan });
  };
  const onPointerMove = (e: PointerEvent): void => {
    if (!drag) return;
    const p = local(e);
    setDrag({ ...drag, now: p });
    const dx = p.x - drag.start.x;
    const dy = p.y - drag.start.y;
    if (tool === "pan") setPan({ x: drag.pan0.x + dx, y: drag.pan0.y + dy });
    if (tool === "wl" && drag.wl0) {
      const range = meta.data ? meta.data.value_range[1] - meta.data.value_range[0] : 1000;
      setWl(dragWindow(drag.wl0, dx, dy, Math.max(0.1, range / 800)));
      setPresetId("custom");
    }
  };
  const onPointerUp = (): void => {
    if (drag && tool === "box" && props.onPromptsChange && frame) {
      const xy = screenBoxToSource(drag.start, drag.now, transform, srcW, srcH);
      if (xy[1][0] - xy[0][0] >= 2 && xy[1][1] - xy[0][1] >= 2) {
        props.onPromptsChange([...props.prompts, { type: "bbox", xy, frame: frameIndex }]);
      }
    }
    setDrag(null);
  };
  const step = useCallback(
    (delta: number) => setFrameIndex((i) => Math.min(count - 1, Math.max(0, i + delta))),
    [count],
  );
  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.key === "ArrowUp" || e.key === "PageUp") {
      step(-1);
      e.preventDefault();
    } else if (e.key === "ArrowDown" || e.key === "PageDown") {
      step(1);
      e.preventDefault();
    } else if (e.key === "+" || e.key === "=") setZoom((z) => Math.min(40, z * 1.2));
    else if (e.key === "-") setZoom((z) => Math.max(0.1, z / 1.2));
  };

  const applyPreset = (id: string): void => {
    setPresetId(id);
    if (!frame) return;
    if (id === "auto") setWl(autoWindow(frame.pixels, frame.rescaleSlope, frame.rescaleIntercept));
    else if (id === "full" && meta.data) setWl(fullRangeWindow(meta.data.value_range[0], meta.data.value_range[1]));
    else if (id === "dicom" && meta.data?.window) setWl(normalizeWindow({ center: meta.data.window[0], width: meta.data.window[1] }));
    else {
      const p = CT_PRESETS.find((x) => x.id === id);
      if (p) setWl(p.window);
    }
  };

  const isCT = current?.modality === "CT";
  const monochrome = frame?.channels !== 3;
  const promptsHere = props.prompts.filter((p) => p.frame === frameIndex);
  const geometry = showMeasurements
    ? measurements.flatMap((m) =>
        m.geometry
          .filter((g) => count === 1 || g.frame === frameIndex)
          .map((g, j) => ({ key: `${m.name}-${j}`, g, m })),
      )
    : [];

  const toScreen = (xy: readonly [number, number]): Point => sourceToScreen({ x: xy[0], y: xy[1] }, transform);
  const shape = (g: Primitive, key: string, cls: string, label?: string): ReactNode => {
    const pts = g.xy.map(toScreen);
    const first = pts[0];
    if (!first) return null;
    let el: ReactNode = null;
    if (g.type === "point") el = <circle cx={first.x} cy={first.y} r={4} />;
    else if (g.type === "line" && pts[1]) el = <line x1={first.x} y1={first.y} x2={pts[1].x} y2={pts[1].y} />;
    else if (g.type === "bbox" && pts[1]) {
      el = (
        <rect
          x={Math.min(first.x, pts[1].x)}
          y={Math.min(first.y, pts[1].y)}
          width={Math.abs(pts[1].x - first.x)}
          height={Math.abs(pts[1].y - first.y)}
        />
      );
    } else if (g.type === "polygon") el = <polygon points={pts.map((p) => `${p.x},${p.y}`).join(" ")} />;
    return (
      <g key={key} className={cls}>
        {el}
        {label ? (
          <text x={first.x + 6} y={first.y - 6}>
            {label}
          </text>
        ) : null}
      </g>
    );
  };

  const dragBox =
    drag && tool === "box" ? (
      <rect
        className="prompt-drawing"
        x={Math.min(drag.start.x, drag.now.x)}
        y={Math.min(drag.start.y, drag.now.y)}
        width={Math.abs(drag.now.x - drag.start.x)}
        height={Math.abs(drag.now.y - drag.start.y)}
      />
    ) : null;

  return (
    <section className="viewer card" aria-labelledby="viewer-title">
      <div className="viewer-head">
        <h2 id="viewer-title">{t("viewer.title")}</h2>
        {series.length > 1 ? (
          <label className="field inline">
            <span>{t("study.series")}</span>
            <select value={current?.id} onChange={(e) => onSeriesChange(e.target.value)}>
              {series.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.position + 1}. {s.modality} {s.description ?? ""} ({s.n_frames})
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>

      <div className="viewer-toolbar" role="toolbar" aria-label={t("viewer.tool")}>
        <div className="seg-control" role="radiogroup" aria-label={t("viewer.tool")}>
          {(["pan", "wl", ...(props.onPromptsChange ? (["box"] as const) : [])] as Tool[]).map((x) => (
            <label key={x} className={tool === x ? "seg-option active" : "seg-option"}>
              <input type="radio" name="viewer-tool" value={x} checked={tool === x} onChange={() => setTool(x)} />
              {t(`viewer.tool.${x}` as MessageKey)}
            </label>
          ))}
        </div>
        {monochrome ? (
          <>
            <label className="field inline">
              <span>{t("viewer.presets")}</span>
              <select value={presetId} onChange={(e) => applyPreset(e.target.value)} data-testid="wl-preset">
                {presetId === "custom" ? <option value="custom">—</option> : null}
                <option value="auto">{t("viewer.preset.auto")}</option>
                {meta.data?.window ? <option value="dicom">{t("viewer.preset.dicom")}</option> : null}
                <option value="full">{t("viewer.preset.full")}</option>
                {isCT
                  ? CT_PRESETS.map((p) => (
                      <option key={p.id} value={p.id}>
                        {t(p.labelKey as MessageKey)}
                      </option>
                    ))
                  : null}
              </select>
            </label>
            <label className="field inline narrow">
              <span>{t("viewer.center")}</span>
              <input
                type="number"
                value={wl ? Math.round(wl.center) : ""}
                onChange={(e) => {
                  setWl(normalizeWindow({ center: Number(e.target.value), width: wl?.width ?? 1 }));
                  setPresetId("custom");
                }}
                data-testid="wl-center"
              />
            </label>
            <label className="field inline narrow">
              <span>{t("viewer.width")}</span>
              <input
                type="number"
                min={1}
                value={wl ? Math.round(wl.width) : ""}
                onChange={(e) => {
                  setWl(normalizeWindow({ center: wl?.center ?? 0, width: Number(e.target.value) }));
                  setPresetId("custom");
                }}
                data-testid="wl-width"
              />
            </label>
            <label className="check">
              <input type="checkbox" checked={invert} onChange={(e) => setInvert(e.target.checked)} />
              {t("viewer.invert")}
            </label>
          </>
        ) : null}
        <button type="button" className="btn btn-small" onClick={() => setZoom((z) => Math.min(40, z * 1.25))} aria-label={t("viewer.zoomIn")}>
          +
        </button>
        <button type="button" className="btn btn-small" onClick={() => setZoom((z) => Math.max(0.1, z / 1.25))} aria-label={t("viewer.zoomOut")}>
          −
        </button>
        <button
          type="button"
          className="btn btn-small"
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
          }}
        >
          {t("viewer.reset")}
        </button>
      </div>

      <div
        className={`viewport tool-${tool}`}
        ref={wrapRef}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => setDrag(null)}
        data-testid="viewport"
      >
        <canvas
          ref={canvasRef}
          style={{ width: size.w, height: size.h }}
          role="img"
          aria-label={t("viewer.canvas", { series: (current?.position ?? 0) + 1, frame: frameIndex + 1, count })}
          data-testid="viewer-canvas"
          data-frame-ready={base ? "true" : "false"}
          data-heatmap-drawn={heatCanvas && heatmapOnFrame ? "true" : "false"}
          data-seg-drawn={segCanvases.length > 0 ? "true" : "false"}
        />
        <svg className="overlay-svg" width={size.w} height={size.h} aria-hidden="true">
          {geometry.map(({ key, g, m }) =>
            shape(g, key, "measure", m.value !== null ? `${m.value.toFixed(2)} ${m.unit}` : m.name),
          )}
          {promptsHere.map((p, i) => shape(p, `prompt-${i}`, "prompt"))}
          {dragBox}
        </svg>
        {!frame && !frameError ? <div className="viewport-msg">{t("viewer.loadingFrame")}</div> : null}
      </div>
      {frameError ? <ErrorBox error={new Error(t("viewer.frameError", { detail: frameError.message }))} /> : null}
      {meta.error ? <ErrorBox error={meta.error} onRetry={meta.reload} /> : null}

      <div className="viewer-status">
        {count > 1 ? (
          <label className="field inline grow">
            <span>{t("viewer.frame")}</span>
            <input
              type="range"
              min={0}
              max={count - 1}
              value={frameIndex}
              onChange={(e) => setFrameIndex(Number(e.target.value))}
              aria-valuetext={t("viewer.frameOf", { frame: frameIndex + 1, count })}
            />
            <span>{t("viewer.frameOf", { frame: frameIndex + 1, count })}</span>
          </label>
        ) : null}
        {frame && frame.scale > 1 ? <span className="muted small">{t("viewer.scaleNote", { scale: frame.scale })}</span> : null}
        <span className="muted small">{t("viewer.hint")}</span>
      </div>

      <fieldset className="overlays">
        <legend>{t("viewer.overlays")}</legend>
        <div className="overlay-group">
          <h3>{t("viewer.heatmap")}</h3>
          {explanations.length === 0 ? (
            <p className="muted small">{t("viewer.heatmap.none")}</p>
          ) : (
            <>
              <label className="check">
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                  data-testid="toggle-heatmap"
                />
                {t("viewer.heatmap")}
              </label>
              <label className="field inline">
                <span className="sr-only">{t("viewer.heatmap")}</span>
                <select
                  value={activeExplanationId ?? ""}
                  onChange={(e) => onExplanationChange(e.target.value || null)}
                  data-testid="heatmap-select"
                >
                  <option value="">—</option>
                  {explanations.map((x) => (
                    <option key={x.id} value={x.id}>
                      {t("viewer.heatmap.target", { label: x.target_label, method: x.method })}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field inline">
                <span>{t("viewer.alpha")}</span>
                <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
              </label>
              <label className="field inline">
                <span>{t("viewer.colormap")}</span>
                <select value={cmap} onChange={(e) => setCmap(e.target.value as ColormapName)}>
                  <option value="inferno">inferno</option>
                  <option value="jet">jet</option>
                  <option value="gray">gray</option>
                </select>
              </label>
              <p className="muted small">{t("viewer.heatmap.caveat")}</p>
            </>
          )}
        </div>
        <div className="overlay-group">
          <h3>{t("viewer.segmentation")}</h3>
          {segmentations.length === 0 ? <p className="muted small">{t("viewer.segmentation.none")}</p> : null}
          {segmentations.map(({ s, i }) => {
            const on = visibleSegs.has(i);
            const hidden = hiddenLabels[i] ?? new Set<number>();
            return (
              <div key={s.id} className="seg-entry">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={(e) => {
                      const next = new Set(visibleSegs);
                      if (e.target.checked) next.add(i);
                      else next.delete(i);
                      setVisibleSegs(next);
                    }}
                    data-testid={`toggle-seg-${i}`}
                  />
                  {s.name}
                </label>
                {on ? (
                  <ul className="legend" aria-label={t("viewer.legend")}>
                    {Object.entries(s.labels).map(([k, code]) => {
                      const idx = Number(k);
                      const [r, g, b] = labelColor(idx);
                      return (
                        <li key={k}>
                          <label className="check">
                            <input
                              type="checkbox"
                              checked={!hidden.has(idx)}
                              onChange={(e) => {
                                const h = new Set(hidden);
                                if (e.target.checked) h.delete(idx);
                                else h.add(idx);
                                setHiddenLabels({ ...hiddenLabels, [i]: h });
                              }}
                            />
                            <span className="swatch" style={{ background: `rgb(${r},${g},${b})` }} aria-hidden="true" />
                            {code.meaning}
                            <span className="muted small"> ({s.label_keys[k] ?? k})</span>
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="overlay-group">
          <h3>{t("viewer.measurements")}</h3>
          <label className="check">
            <input type="checkbox" checked={showMeasurements} onChange={(e) => setShowMeasurements(e.target.checked)} />
            {t("viewer.measurements")}
          </label>
          {showMeasurements && geometry.length === 0 ? <p className="muted small">{t("viewer.measurements.none")}</p> : null}
        </div>
        {props.onPromptsChange ? (
          <div className="overlay-group">
            <h3>{t("viewer.prompts")}</h3>
            <p className="muted small">{t("viewer.prompts.help")}</p>
            <p className="small">{t("study.analyze.prompts", { count: props.prompts.length })}</p>
            <button
              type="button"
              className="btn btn-small"
              disabled={props.prompts.length === 0}
              onClick={() => props.onPromptsChange?.([])}
            >
              {t("viewer.prompts.clear")}
            </button>
          </div>
        ) : null}
      </fieldset>
    </section>
  );
}
