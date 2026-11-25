/**
 * Estudo: visualizador + achados + revisão/laudo da análise selecionada.
 *
 * A análise é pedida ao servidor (fila do worker) e acompanhada por consulta
 * periódica enquanto está ``queued``/``running``. Nada é calculado aqui além
 * da exibição: os achados são exatamente os do resultado do servidor.
 */
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { api, unwrap, type Schemas } from "../api/client";
import { ErrorBox, Hash, Loading, Pill } from "../components/common";
import { FindingsPanel } from "../components/FindingsPanel";
import { ReportPanel } from "../components/ReportPanel";
import { useI18n } from "../i18n/i18n";
import type { MessageKey } from "../i18n/messages";
import { groupFindings, pendingReviews } from "../lib/evidence";
import { navigate, routeHref } from "../lib/route";
import { useAsync } from "../lib/useAsync";
import { useSession } from "../session";
import { Viewer } from "../viewer/Viewer";
import { stateTone } from "./WorklistPage";

type Primitive = Schemas["Primitive"];

const POLL_MS = 1500;

function isActive(a: { state: string } | null | undefined): boolean {
  return a?.state === "queued" || a?.state === "running";
}

export function StudyPage({ studyId, analysisId }: { studyId: string; analysisId: string | null }): ReactNode {
  const { t, dt } = useI18n();
  const { can, research } = useSession();
  const study = useAsync(
    async () => unwrap(await api.GET("/api/v1/studies/{study_id}", { params: { path: { study_id: studyId } } })),
    [studyId],
  );
  const [seriesId, setSeriesId] = useState<string>("");
  const [prompts, setPrompts] = useState<Primitive[]>([]);
  const [includeResearch, setIncludeResearch] = useState(false);
  const [busy, setBusy] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [activeExplanationId, setActiveExplanationId] = useState<string | null>(null);

  const analyses = study.data?.analyses ?? [];
  // Sem análise escolhida na rota: a mais recente.
  const selectedId = analysisId ?? analyses[0]?.id ?? null;

  const analysis = useAsync(
    async () =>
      selectedId
        ? unwrap(
            await api.GET("/api/v1/analyses/{analysis_id}", {
              params: { path: { analysis_id: selectedId }, query: { include_research: research } },
            }),
          )
        : null,
    [selectedId, research],
  );

  // Consulta periódica enquanto o worker processa.
  const active = isActive(analysis.data);
  const reloadAnalysis = analysis.reload;
  const reloadStudy = study.reload;
  useEffect(() => {
    if (!active) return;
    const id = window.setTimeout(() => {
      reloadAnalysis();
    }, POLL_MS);
    return () => window.clearTimeout(id);
  }, [active, analysis.data, reloadAnalysis]);

  // Ao terminar, atualiza o resumo do estudo (estado na lista de análises).
  const finishedState = analysis.data?.state;
  useEffect(() => {
    if (finishedState === "completed" || finishedState === "failed") reloadStudy();
  }, [finishedState, reloadStudy]);

  // Primeiro mapa de explicação disponível vira o ativo (o usuário pode trocar/ocultar).
  useEffect(() => {
    const ex = analysis.data?.result?.explanations ?? [];
    setActiveExplanationId((cur) => (cur && ex.some((e) => e.id === cur) ? cur : (ex[0]?.id ?? null)));
  }, [analysis.data?.result]);

  // Visualizar a série da imagem do mapa ativo.
  useEffect(() => {
    const a = analysis.data;
    const ex = a?.result?.explanations.find((e) => e.id === activeExplanationId);
    if (!a || !ex) return;
    const sid = a.inputs.find((i) => i.image_idx === ex.image_idx)?.series_id;
    if (sid) setSeriesId(sid);
  }, [activeExplanationId, analysis.data]);

  const pending = useMemo(() => {
    const a = analysis.data;
    if (!a?.result) return 0;
    const groups = groupFindings(a.result.findings, a.finding_indices, research);
    return pendingReviews(groups, a.reviews, a.result_sha256).length;
  }, [analysis.data, research]);

  if (study.error) return <ErrorBox error={study.error} onRetry={study.reload} />;
  if (!study.data) return <Loading />;
  const s = study.data;
  const canAnalyze = can("analyses:create");
  const needsExamType = s.exam_type === "unknown";

  const createAnalysis = async (): Promise<void> => {
    setBusy(true);
    setCreateError(null);
    try {
      const out = unwrap(
        await api.POST("/api/v1/studies/{study_id}/analyses", {
          params: { path: { study_id: s.id } },
          body: { include_research: includeResearch && research, prompts },
        }),
      );
      setPrompts([]);
      navigate({ name: "study", studyId: s.id, analysisId: out.id });
      study.reload();
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const a = analysis.data ?? null;
  return (
    <div className="page study">
      <p>
        <a href={routeHref({ name: "worklist" })}>← {t("study.back")}</a>
      </p>
      <h1>
        {t("study.title")}: {s.description ?? s.modalities.join(", ")}{" "}
        <span className="muted small">{t(`exam.${s.exam_type}` as MessageKey)}</span>
      </h1>
      <p className="muted small">
        {t("study.pseudoId")}: {s.patient_pseudo_id ?? "—"} · {dt(s.created_at)}
      </p>
      {s.warnings.length > 0 ? (
        <details className="warnings">
          <summary>{t("study.warnings")}</summary>
          <ul>
            {s.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </details>
      ) : null}

      <div className="study-layout">
        <div className="study-main">
          {s.series.length > 0 ? (
            <Viewer
              series={s.series}
              seriesId={seriesId || (s.series[0]?.id ?? "")}
              onSeriesChange={setSeriesId}
              analysis={a && a.state === "completed" ? a : null}
              activeExplanationId={activeExplanationId}
              onExplanationChange={setActiveExplanationId}
              prompts={prompts}
              onPromptsChange={canAnalyze ? setPrompts : null}
            />
          ) : null}
        </div>

        <div className="study-side">
          <section className="card analyses" aria-labelledby="analyses-title">
            <h2 id="analyses-title">{t("study.analyses")}</h2>
            {analyses.length === 0 ? <p className="muted">{t("study.analysis.none")}</p> : null}
            {analyses.length > 0 ? (
              <label className="field">
                <span>{t("study.analysis.select")}</span>
                <select
                  value={selectedId ?? ""}
                  onChange={(e) => navigate({ name: "study", studyId: s.id, analysisId: e.target.value })}
                  data-testid="analysis-select"
                >
                  {analyses.map((x) => (
                    <option key={x.id} value={x.id}>
                      {dt(x.requested_at)} — {t(`state.${x.state}` as MessageKey)}
                      {x.include_research ? " (D)" : ""}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            {canAnalyze ? (
              <div className="analyze-box">
                {needsExamType ? (
                  <p className="pending-note">{t("study.analyze.needsExamType")}</p>
                ) : null}
                {research ? (
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={includeResearch}
                      onChange={(e) => setIncludeResearch(e.target.checked)}
                    />
                    {t("study.analyze.research")}
                  </label>
                ) : null}
                {prompts.length > 0 ? (
                  <p className="small">{t("study.analyze.prompts", { count: prompts.length })}</p>
                ) : null}
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={busy || needsExamType || isActive(a)}
                  onClick={() => void createAnalysis()}
                  data-testid="analyze"
                >
                  {t("study.analyze")}
                </button>
                {createError ? (
                  <p className="error-box" role="alert">
                    {createError}
                  </p>
                ) : null}
              </div>
            ) : null}
            {analysis.error ? <ErrorBox error={analysis.error} onRetry={analysis.reload} /> : null}
            {a ? (
              <div className="analysis-state" aria-live="polite">
                <p>
                  <Pill tone={stateTone(a.state)}>
                    <span data-testid="analysis-state">{t(`state.${a.state}` as MessageKey)}</span>
                  </Pill>{" "}
                  {a.result_status ? (
                    <span className="small">
                      {t("study.analysis.result", { status: t(`result.${a.result_status}` as MessageKey) })}
                    </span>
                  ) : null}
                </p>
                {isActive(a) ? <p className="muted small">{t("study.analysis.polling")}</p> : null}
                {a.error ? (
                  <p className="error-box" role="alert">
                    {t("study.analysis.error", { error: a.error })}
                  </p>
                ) : null}
                {a.result_sha256 ? (
                  <p className="muted small">
                    {t("study.analysis.hash")}: <Hash value={a.result_sha256} />
                  </p>
                ) : null}
              </div>
            ) : null}
          </section>

          {a && a.state === "completed" && a.result ? (
            <>
              <FindingsPanel
                analysis={a}
                researchView={research && a.research_view}
                canReview={can("reviews:write")}
                activeExplanationId={activeExplanationId}
                onShowExplanation={setActiveExplanationId}
                onReviewed={analysis.reload}
              />
              {can("reports:read") || can("reports:create") ? (
                <ReportPanel analysisId={a.id} pendingReviews={pending} />
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
