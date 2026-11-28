/**
 * Lógica de exibição de achados: níveis de evidência, escore x ponto de
 * operação e agrupamento. Regras do produto codificadas aqui:
 *
 * * Nível D é exploratório: só aparece no modo pesquisa e nunca é revisável.
 * * "Probabilidade" só é dita quando o achado traz ``probability`` E
 *   ``calibration_id`` (calibração ajustada). Caso contrário é "escore".
 * * A decisão (positivo/negativo/...) é a que o backend calculou; o cliente
 *   só mostra a relação do escore com o limiar quando a escala é a mesma
 *   (ou uma conversão matemática exata, logit -> sigmoide).
 */
import type { Schemas } from "../api/client";

export type Finding = Schemas["Finding"];
export type OperatingPoint = Schemas["OperatingPoint"];
export type EvidenceLevel = Schemas["EvidenceLevel"];
export type Review = Schemas["ReviewOut"];

export const LEVELS: readonly EvidenceLevel[] = ["A", "B", "C", "D"];

export interface LevelBadge {
  level: EvidenceLevel;
  /** Chave i18n do rótulo curto (ex.: "Nível A"). */
  labelKey: string;
  /** Chave i18n do significado em linguagem simples. */
  meaningKey: string;
  /** Classe CSS do tom do selo. */
  tone: "level-a" | "level-b" | "level-c" | "level-d";
  /** Pode entrar no laudo? (A-C sim; D nunca sozinho). */
  reportable: boolean;
  /** Só no modo pesquisa? */
  researchOnly: boolean;
}

export function levelBadge(level: EvidenceLevel): LevelBadge {
  const lower = level.toLowerCase() as "a" | "b" | "c" | "d";
  return {
    level,
    labelKey: `level.${lower}.label`,
    meaningKey: `level.${lower}.meaning`,
    tone: `level-${lower}`,
    reportable: level !== "D",
    researchOnly: level === "D",
  };
}

export type ScoreDisplay =
  | { kind: "probability"; value: number; calibrationId: string }
  | { kind: "score"; raw: number; scoreKind: Finding["score_kind"]; opNormalized: number | null }
  | { kind: "category"; category: string; classScores: Record<string, number> }
  | { kind: "none" };

/** Como mostrar o número de um achado sem chamar escore de probabilidade. */
export function scoreDisplay(f: Finding): ScoreDisplay {
  if (f.decision === "abstained" || f.raw_score === null) return { kind: "none" };
  if (f.kind === "categorical" && f.category && f.class_scores) {
    return { kind: "category", category: f.category, classScores: f.class_scores };
  }
  if (
    typeof f.probability === "number" &&
    typeof f.calibration_id === "string" &&
    f.calibration_id.length > 0
  ) {
    return { kind: "probability", value: f.probability, calibrationId: f.calibration_id };
  }
  return {
    kind: "score",
    raw: f.raw_score,
    scoreKind: f.score_kind,
    opNormalized: typeof f.op_normalized === "number" ? f.op_normalized : null,
  };
}

export interface OperatingPointRelation {
  op: OperatingPoint;
  /** Valor do escore na escala do ponto de operação, quando comparável. */
  value: number | null;
  /** O escore atinge o limiar (segundo ``direction``)? ``null`` se não comparável. */
  meets: boolean | null;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Valor do escore cru na escala ``on`` do ponto de operação, se a conversão é exata. */
export function scoreOnScale(f: Finding, on: OperatingPoint["on"]): number | null {
  if (f.raw_score === null) return null;
  if (on === f.score_kind) return f.raw_score;
  if (on === "sigmoid" && f.score_kind === "logit") return sigmoid(f.raw_score);
  if (on === "calibrated" && typeof f.probability === "number" && f.calibration_id) {
    return f.probability;
  }
  return null;
}

export function operatingPointRelations(f: Finding): OperatingPointRelation[] {
  return (f.operating_points ?? []).map((op) => {
    const value = scoreOnScale(f, op.on);
    let meets: boolean | null = null;
    if (value !== null) meets = op.direction === "le" ? value <= op.threshold : value >= op.threshold;
    return { op, value, meets };
  });
}

export interface DecisionBadge {
  labelKey: string;
  tone: "pos" | "neg" | "ind" | "cat" | "score" | "abst";
}

export function decisionBadge(decision: Finding["decision"]): DecisionBadge {
  switch (decision) {
    case "positive":
      return { labelKey: "decision.positive", tone: "pos" };
    case "negative":
      return { labelKey: "decision.negative", tone: "neg" };
    case "indeterminate":
      return { labelKey: "decision.indeterminate", tone: "ind" };
    case "category":
      return { labelKey: "decision.category", tone: "cat" };
    case "score_only":
      return { labelKey: "decision.score_only", tone: "score" };
    case "abstained":
      return { labelKey: "decision.abstained", tone: "abst" };
  }
}

export interface IndexedFinding {
  /** Índice no resultado COMPLETO (o que ``ReviewIn.finding_idx`` espera). */
  fullIdx: number;
  finding: Finding;
}

export interface FindingGroup {
  level: EvidenceLevel;
  badge: LevelBadge;
  items: IndexedFinding[];
}

/**
 * Agrupa por nível (A, B, C e, só no modo pesquisa, D), preservando a ordem do
 * resultado. ``indices[i]`` é o índice completo do i-ésimo achado exibido.
 * Achados de nível D recebidos fora do modo pesquisa são descartados (defesa em
 * profundidade: o servidor já os omite).
 */
export function groupFindings(
  findings: readonly Finding[],
  indices: readonly number[] | null | undefined,
  researchView: boolean,
): FindingGroup[] {
  const groups = new Map<EvidenceLevel, IndexedFinding[]>();
  findings.forEach((finding, i) => {
    const level = finding.evidence_level;
    if (level === "D" && !researchView) return;
    if (finding.visibility === "research" && !researchView) return;
    const fullIdx = indices?.[i] ?? i;
    const list = groups.get(level) ?? [];
    list.push({ fullIdx, finding });
    groups.set(level, list);
  });
  return LEVELS.filter((l) => groups.has(l)).map((level) => ({
    level,
    badge: levelBadge(level),
    items: groups.get(level) ?? [],
  }));
}

/** Só achados da visão principal (níveis A-C) são revisáveis e entram no laudo. */
export function isReviewable(f: Finding): boolean {
  return f.visibility === "main" && f.evidence_level !== "D";
}

/** Revisão vigente de cada achado (a mais recente; revisões são só inserção). */
export function latestReviews(reviews: readonly Review[]): Map<number, Review> {
  const out = new Map<number, Review>();
  for (const r of reviews) {
    const prev = out.get(r.finding_idx);
    if (prev === undefined || Date.parse(r.created_at) >= Date.parse(prev.created_at)) {
      out.set(r.finding_idx, r);
    }
  }
  return out;
}

/** Achados revisáveis ainda sem revisão vigente para ESTE resultado (hash). */
export function pendingReviews(
  groups: readonly FindingGroup[],
  reviews: readonly Review[],
  resultSha256: string | null | undefined,
): IndexedFinding[] {
  const latest = latestReviews(reviews.filter((r) => r.result_sha256 === resultSha256));
  return groups
    .flatMap((g) => g.items)
    .filter((it) => isReviewable(it.finding) && !latest.has(it.fullIdx));
}

/** Formata um número com dígitos significativos razoáveis para tela. */
export function formatScore(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return "—";
  if (v !== 0 && Math.abs(v) < 0.001) return v.toExponential(2);
  return v.toFixed(digits);
}
