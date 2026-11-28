import { describe, expect, it } from "vitest";

import {
  decisionBadge,
  formatScore,
  groupFindings,
  isReviewable,
  latestReviews,
  levelBadge,
  operatingPointRelations,
  pendingReviews,
  scoreDisplay,
  scoreOnScale,
  type Finding,
  type OperatingPoint,
  type Review,
} from "./evidence";

function finding(over: Partial<Finding> = {}): Finding {
  return {
    code: { scheme: "SCT", value: "36118008", meaning: "Pneumotórax" },
    condition_id: "pneumothorax",
    decision: "score_only",
    evidence_level: "B",
    explanation_ids: [],
    image_idx: 0,
    kind: "binary",
    label: "Pneumothorax",
    operating_points: [],
    provenance_idx: 0,
    raw_score: 0.2,
    score_kind: "sigmoid",
    visibility: "main",
    ...over,
  };
}

function op(over: Partial<OperatingPoint> = {}): OperatingPoint {
  return {
    direction: "ge",
    name: "rule_in",
    on: "sigmoid",
    role: "rule_in",
    source: "artifacts/eval/exemplo.json",
    threshold: 0.5,
    ...over,
  };
}

function review(idx: number, created: string, over: Partial<Review> = {}): Review {
  return {
    action: "accept",
    amended_decision: null,
    amended_text: null,
    analysis_id: "a1",
    comment: null,
    created_at: created,
    finding_idx: idx,
    id: `r-${idx}-${created}`,
    result_sha256: "h1",
    reviewer_id: "u1",
    reviewer_username: "medico",
    ...over,
  };
}

describe("levelBadge", () => {
  it("A-C entram no laudo; D é só pesquisa", () => {
    for (const l of ["A", "B", "C"] as const) {
      const b = levelBadge(l);
      expect(b.reportable).toBe(true);
      expect(b.researchOnly).toBe(false);
      expect(b.tone).toBe(`level-${l.toLowerCase()}`);
      expect(b.labelKey).toBe(`level.${l.toLowerCase()}.label`);
      expect(b.meaningKey).toBe(`level.${l.toLowerCase()}.meaning`);
    }
    const d = levelBadge("D");
    expect(d.reportable).toBe(false);
    expect(d.researchOnly).toBe(true);
  });
});

describe("scoreDisplay: nunca chamar escore de probabilidade sem calibração", () => {
  it("escore cru quando não há calibração", () => {
    expect(scoreDisplay(finding({ probability: 0.7 })).kind).toBe("score");
    expect(scoreDisplay(finding({ probability: 0.7, calibration_id: "" })).kind).toBe("score");
  });
  it("probabilidade só com probability E calibration_id", () => {
    const s = scoreDisplay(finding({ probability: 0.7, calibration_id: "platt-v1" }));
    expect(s).toEqual({ kind: "probability", value: 0.7, calibrationId: "platt-v1" });
  });
  it("abstenção e escore nulo não mostram número", () => {
    expect(scoreDisplay(finding({ decision: "abstained", abstention_reason: "out_of_distribution" })).kind).toBe(
      "none",
    );
    expect(scoreDisplay(finding({ raw_score: null })).kind).toBe("none");
  });
  it("categórico mostra a categoria", () => {
    const s = scoreDisplay(finding({ kind: "categorical", decision: "category", category: "C", class_scores: { A: 0.1, C: 0.9 } }));
    expect(s.kind).toBe("category");
  });
  it("normalização do ponto de operação é repassada", () => {
    const s = scoreDisplay(finding({ op_normalized: 0.42 }));
    expect(s).toMatchObject({ kind: "score", opNormalized: 0.42 });
  });
});

describe("escore x ponto de operação", () => {
  it("mesma escala compara direto, nos dois sentidos", () => {
    const f = finding({ raw_score: 0.6, operating_points: [op(), op({ name: "rule_out", direction: "le", threshold: 0.1 })] });
    const rel = operatingPointRelations(f);
    expect(rel.map((r) => r.meets)).toEqual([true, false]);
  });
  it("logit -> sigmoide é conversão exata", () => {
    const f = finding({ raw_score: 0, score_kind: "logit" });
    expect(scoreOnScale(f, "sigmoid")).toBeCloseTo(0.5);
    expect(operatingPointRelations({ ...f, operating_points: [op({ threshold: 0.5 })] })[0]?.meets).toBe(true);
  });
  it("escalas incompatíveis não são comparadas", () => {
    const f = finding({ score_kind: "softmax", operating_points: [op({ on: "measurement" })] });
    const [r] = operatingPointRelations(f);
    expect(r?.value).toBeNull();
    expect(r?.meets).toBeNull();
    expect(scoreOnScale(finding({ score_kind: "sigmoid" }), "logit")).toBeNull();
  });
  it("calibrada só com calibração presente", () => {
    expect(scoreOnScale(finding({ probability: 0.3 }), "calibrated")).toBeNull();
    expect(scoreOnScale(finding({ probability: 0.3, calibration_id: "iso" }), "calibrated")).toBe(0.3);
  });
});

describe("decisionBadge", () => {
  it("cobre todas as decisões", () => {
    const all = ["positive", "negative", "indeterminate", "category", "score_only", "abstained"] as const;
    expect(new Set(all.map((d) => decisionBadge(d).tone)).size).toBe(all.length);
  });
});

describe("groupFindings", () => {
  const fs = [
    finding({ evidence_level: "C" }),
    finding({ evidence_level: "A" }),
    finding({ evidence_level: "D", visibility: "research" }),
    finding({ evidence_level: "A" }),
    finding({ evidence_level: "B", visibility: "research" }),
  ];
  it("agrupa em ordem A,B,C e esconde D/pesquisa fora do modo pesquisa", () => {
    const g = groupFindings(fs, null, false);
    expect(g.map((x) => x.level)).toEqual(["A", "C"]);
    expect(g[0]?.items.map((i) => i.fullIdx)).toEqual([1, 3]);
  });
  it("modo pesquisa mostra D", () => {
    const g = groupFindings(fs, null, true);
    expect(g.map((x) => x.level)).toEqual(["A", "B", "C", "D"]);
  });
  it("usa os índices completos do servidor", () => {
    const g = groupFindings([finding({ evidence_level: "B" })], [7], false);
    expect(g[0]?.items[0]?.fullIdx).toBe(7);
  });
});

describe("revisões", () => {
  it("só A-C da visão principal são revisáveis", () => {
    expect(isReviewable(finding())).toBe(true);
    expect(isReviewable(finding({ evidence_level: "D" }))).toBe(false);
    expect(isReviewable(finding({ visibility: "research" }))).toBe(false);
  });
  it("vale a revisão mais recente de cada achado", () => {
    const m = latestReviews([
      review(0, "2026-01-01T00:00:00Z", { action: "reject" }),
      review(0, "2026-01-02T00:00:00Z", { action: "amend" }),
      review(1, "2026-01-01T00:00:00Z"),
    ]);
    expect(m.get(0)?.action).toBe("amend");
    expect(m.size).toBe(2);
  });
  it("pendências consideram só revisões do MESMO hash de resultado", () => {
    const groups = groupFindings([finding(), finding({ evidence_level: "A" })], null, false);
    const reviews = [review(0, "2026-01-01T00:00:00Z"), review(1, "2026-01-01T00:00:00Z", { result_sha256: "outro" })];
    expect(pendingReviews(groups, reviews, "h1").map((p) => p.fullIdx)).toEqual([1]);
  });
});

describe("formatScore", () => {
  it("formata com segurança", () => {
    expect(formatScore(0.12345)).toBe("0.123");
    expect(formatScore(0.00001234)).toBe("1.23e-5");
    expect(formatScore(Number.NaN)).toBe("—");
    expect(formatScore(0)).toBe("0.000");
  });
});
