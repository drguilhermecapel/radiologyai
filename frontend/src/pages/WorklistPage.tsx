import { useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { api, postForm, unwrap, type Schemas } from "../api/client";
import type { operations } from "../api/schema";
import { ErrorBox, Loading, Pill } from "../components/common";
import { useI18n } from "../i18n/i18n";
import type { MessageKey } from "../i18n/messages";
import {
  ANALYSIS_STATES,
  EXAM_TYPES,
  MODALITIES,
  REVIEW_STATUSES,
  type AnalysisStateFilter,
  type ExamType,
  type ReviewStatus,
} from "../lib/domain";
import { navigate, routeHref } from "../lib/route";
import { useAsync } from "../lib/useAsync";
import { useSession } from "../session";

const PAGE = 25;

function ExamTypeSetter({ studyId, onDone }: { studyId: string; onDone: () => void }): ReactNode {
  const { t } = useI18n();
  const [value, setValue] = useState<ExamType | "">("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (): Promise<void> => {
    if (value === "") return;
    setBusy(true);
    setError(null);
    try {
      unwrap(
        await api.PUT("/api/v1/studies/{study_id}/exam-type", {
          params: { path: { study_id: studyId } },
          body: { exam_type: value },
        }),
      );
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <span className="inline-form">
      <label>
        <span className="sr-only">{t("worklist.chooseExamType")}</span>
        <select
          value={value}
          onChange={(e) => setValue(e.target.value as ExamType | "")}
          aria-label={t("worklist.chooseExamType")}
        >
          <option value="">{t("worklist.chooseExamType")}</option>
          {EXAM_TYPES.map((x) => (
            <option key={x} value={x}>
              {t(`exam.${x}` as MessageKey)}
            </option>
          ))}
        </select>
      </label>
      <button type="button" className="btn btn-small" disabled={value === "" || busy} onClick={() => void submit()}>
        {t("worklist.setExamType")}
      </button>
      {error ? (
        <span className="error-inline" role="alert">
          {error}
        </span>
      ) : null}
    </span>
  );
}

function UploadForm({ onUploaded }: { onUploaded: (study: Schemas["StudyOut"]) => void }): ReactNode {
  const { t } = useI18n();
  const [files, setFiles] = useState<FileList | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fields, setFields] = useState({
    exam_type: "",
    modality: "",
    body_part: "",
    view_position: "",
    laterality: "",
    patient_age_years: "",
    patient_sex: "",
  });
  const set = (k: keyof typeof fields, v: string): void => setFields((f) => ({ ...f, [k]: v }));

  const submit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f, f.name));
    for (const [k, v] of Object.entries(fields)) if (v.trim() !== "") form.append(k, v.trim());
    try {
      const study = await postForm<Schemas["StudyOut"]>("/api/v1/studies", form);
      onUploaded(study);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card upload" onSubmit={(e) => void submit(e)} aria-labelledby="upload-title">
      <h2 id="upload-title">{t("upload.title")}</h2>
      <label className="field">
        <span>{t("upload.files")}</span>
        <input
          type="file"
          multiple
          required
          accept=".dcm,.dicom,.zip,.nii,.gz,.png,.jpg,.jpeg,application/dicom,application/zip,image/png,image/jpeg"
          onChange={(e) => setFiles(e.target.files)}
          data-testid="upload-files"
        />
      </label>
      <label className="field">
        <span>{t("upload.examType")}</span>
        <select value={fields.exam_type} onChange={(e) => set("exam_type", e.target.value)} data-testid="upload-exam-type">
          <option value="">{t("upload.examType.infer")}</option>
          {EXAM_TYPES.map((x) => (
            <option key={x} value={x}>
              {t(`exam.${x}` as MessageKey)}
            </option>
          ))}
        </select>
      </label>
      <details data-testid="upload-more">
        <summary>{t("upload.more")}</summary>
        <p className="muted small">{t("upload.meta.help")}</p>
        <div className="grid-2">
          <label className="field">
            <span>{t("upload.modality")}</span>
            <select value={fields.modality} onChange={(e) => set("modality", e.target.value)}>
              <option value="">—</option>
              {MODALITIES.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>{t("upload.bodyPart")}</span>
            <input value={fields.body_part} maxLength={64} onChange={(e) => set("body_part", e.target.value)} />
          </label>
          <label className="field">
            <span>{t("upload.view")}</span>
            <input value={fields.view_position} maxLength={32} onChange={(e) => set("view_position", e.target.value)} />
          </label>
          <label className="field">
            <span>{t("upload.laterality")}</span>
            <select value={fields.laterality} onChange={(e) => set("laterality", e.target.value)}>
              <option value="">—</option>
              <option value="L">L</option>
              <option value="R">R</option>
              <option value="B">B</option>
            </select>
          </label>
          <label className="field">
            <span>{t("upload.age")}</span>
            <input
              type="number"
              min={0}
              max={130}
              step="any"
              value={fields.patient_age_years}
              onChange={(e) => set("patient_age_years", e.target.value)}
              data-testid="upload-age"
            />
          </label>
          <label className="field">
            <span>{t("upload.sex")}</span>
            <select value={fields.patient_sex} onChange={(e) => set("patient_sex", e.target.value)}>
              <option value="">—</option>
              <option value="F">F</option>
              <option value="M">M</option>
              <option value="O">O</option>
            </select>
          </label>
        </div>
      </details>
      <p className="muted small">{t("upload.deident")}</p>
      {error ? (
        <p className="error-box" role="alert">
          {error}
        </p>
      ) : null}
      <button type="submit" className="btn btn-primary" disabled={busy || !files || files.length === 0} data-testid="upload-submit">
        {busy ? t("upload.busy") : t("upload.submit")}
      </button>
    </form>
  );
}

export function stateTone(state: string): string {
  switch (state) {
    case "completed":
      return "ok";
    case "failed":
      return "bad";
    case "running":
    case "queued":
      return "busy";
    default:
      return "neutral";
  }
}

export function WorklistPage(): ReactNode {
  const { t, dt } = useI18n();
  const { can } = useSession();
  const [state, setState] = useState<AnalysisStateFilter | "">("");
  const [examType, setExamType] = useState<ExamType | "">("");
  const [modality, setModality] = useState("");
  const [review, setReview] = useState<ReviewStatus | "">("");
  const [offset, setOffset] = useState(0);

  const list = useAsync(async () => {
    const query: { query: NonNullable<operations["worklist_api_v1_studies_get"]["parameters"]["query"]> } = {
      query: {
        limit: PAGE,
        offset,
        ...(state ? { state } : {}),
        ...(examType ? { exam_type: examType } : {}),
        ...(modality ? { modality } : {}),
        ...(review ? { review_status: review } : {}),
      },
    };
    return unwrap(await api.GET("/api/v1/studies", { params: query }));
  }, [state, examType, modality, review, offset]);

  const resetPage = <T,>(fn: (v: T) => void) => (v: T) => {
    setOffset(0);
    fn(v);
  };

  return (
    <div className="page worklist">
      <h1>{t("worklist.title")}</h1>
      <div className="worklist-layout">
        <section aria-labelledby="wl-list">
          <h2 id="wl-list" className="sr-only">
            {t("worklist.title")}
          </h2>
          <fieldset className="filters">
            <legend>{t("worklist.filters")}</legend>
            <label className="field">
              <span>{t("worklist.filter.state")}</span>
              <select value={state} onChange={(e) => resetPage(setState)(e.target.value as AnalysisStateFilter | "")}>
                <option value="">{t("worklist.filter.all")}</option>
                {ANALYSIS_STATES.map((s) => (
                  <option key={s} value={s}>
                    {t(`state.${s}` as MessageKey)}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>{t("worklist.filter.exam")}</span>
              <select value={examType} onChange={(e) => resetPage(setExamType)(e.target.value as ExamType | "")}>
                <option value="">{t("worklist.filter.all")}</option>
                {[...EXAM_TYPES, "unknown" as const].map((x) => (
                  <option key={x} value={x}>
                    {t(`exam.${x}` as MessageKey)}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>{t("worklist.filter.modality")}</span>
              <select value={modality} onChange={(e) => resetPage(setModality)(e.target.value)}>
                <option value="">{t("worklist.filter.all")}</option>
                {MODALITIES.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>{t("worklist.filter.review")}</span>
              <select value={review} onChange={(e) => resetPage(setReview)(e.target.value as ReviewStatus | "")}>
                <option value="">{t("worklist.filter.all")}</option>
                {REVIEW_STATUSES.map((s) => (
                  <option key={s} value={s}>
                    {t(`review.${s}` as MessageKey)}
                  </option>
                ))}
              </select>
            </label>
          </fieldset>
          {list.error ? <ErrorBox error={list.error} onRetry={list.reload} /> : null}
          {list.loading && !list.data ? <Loading /> : null}
          {list.data ? (
            <>
              <p className="muted" aria-live="polite">
                {t("worklist.total", { total: list.data.total })}
              </p>
              <div className="table-wrap">
                <table className="table" data-testid="worklist-table">
                  <thead>
                    <tr>
                      <th scope="col">{t("worklist.col.created")}</th>
                      <th scope="col">{t("worklist.col.modalities")}</th>
                      <th scope="col">{t("worklist.col.description")}</th>
                      <th scope="col">{t("worklist.col.exam")}</th>
                      <th scope="col">{t("worklist.col.series")}</th>
                      <th scope="col">{t("worklist.col.analysis")}</th>
                      <th scope="col">{t("worklist.col.review")}</th>
                      <th scope="col">{t("worklist.col.actions")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.data.items.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="muted">
                          {t("worklist.empty")}
                        </td>
                      </tr>
                    ) : null}
                    {list.data.items.map((s) => {
                      const st = s.latest_analysis?.state ?? "no_analysis";
                      return (
                        <tr key={s.id}>
                          <td>{dt(s.created_at)}</td>
                          <td>{s.modalities.join(", ")}</td>
                          <td>{s.description ?? "—"}</td>
                          <td>
                            {s.exam_type === "unknown" ? (
                              <div>
                                <Pill tone="warn">{t("worklist.needsExamType")}</Pill>
                                {can("studies:set_exam_type") ? (
                                  <ExamTypeSetter studyId={s.id} onDone={list.reload} />
                                ) : null}
                              </div>
                            ) : (
                              <span title={t(`worklist.examTypeSource.${s.exam_type_source}` as MessageKey)}>
                                {t(`exam.${s.exam_type}` as MessageKey)}
                              </span>
                            )}
                          </td>
                          <td>{s.n_series}</td>
                          <td>
                            <Pill tone={stateTone(st)}>{t(`state.${st}` as MessageKey)}</Pill>
                          </td>
                          <td>
                            {s.latest_analysis?.shadow ? (
                              <span className="muted">{t("shadow.review")}</span>
                            ) : (
                              <Pill tone={s.review_status === "signed" ? "ok" : "neutral"}>
                                {t(`review.${s.review_status}` as MessageKey)}
                              </Pill>
                            )}
                          </td>
                          <td>
                            <a className="btn btn-small" href={routeHref({ name: "study", studyId: s.id, analysisId: null })}>
                              {t("worklist.open")}
                            </a>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="pager">
                <button type="button" className="btn btn-small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
                  {t("worklist.prev")}
                </button>
                <button
                  type="button"
                  className="btn btn-small"
                  disabled={offset + PAGE >= list.data.total}
                  onClick={() => setOffset(offset + PAGE)}
                >
                  {t("worklist.next")}
                </button>
              </div>
            </>
          ) : null}
        </section>
        {can("studies:upload") ? (
          <aside>
            <UploadForm
              onUploaded={(study) => navigate({ name: "study", studyId: study.id, analysisId: null })}
            />
          </aside>
        ) : null}
      </div>
    </div>
  );
}
