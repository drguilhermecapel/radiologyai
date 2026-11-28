/** Enumerações do domínio espelhadas do contrato OpenAPI (para seletores). */
import type { components } from "../api/schema";

export type ExamType = components["schemas"]["ExamType"];

export const EXAM_TYPES: readonly ExamType[] = [
  "cxr",
  "msk_xr",
  "ct_head",
  "ct_chest",
  "ct_abdomen",
  "ct_other",
  "mr_brain",
  "mr_body",
  "mg",
  "us_breast",
  "us_abdomen",
  "us_other",
  "echo",
];

export const ANALYSIS_STATES = ["no_analysis", "queued", "running", "completed", "failed"] as const;
export type AnalysisStateFilter = (typeof ANALYSIS_STATES)[number];

export const REVIEW_STATUSES = ["no_analysis", "pending_review", "in_review", "signed"] as const;
export type ReviewStatus = (typeof REVIEW_STATUSES)[number];

export const MODALITIES = ["CR", "DX", "CT", "MR", "MG", "US", "OT"] as const;
export type UploadModality = (typeof MODALITIES)[number];
