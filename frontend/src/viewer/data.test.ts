import { describe, expect, it } from "vitest";

import { segmentationFrameUrl } from "./data";

// REQ-169: máscara exploratória só é pedida explicitamente em modo pesquisa.
describe("segmentationFrameUrl", () => {
  it("não pede modo pesquisa por padrão", () => {
    expect(segmentationFrameUrl("a1", 3, 0, 512)).toBe(
      "/api/v1/analyses/a1/segmentations/3/frames/0?max_dim=512",
    );
  });
  it("pede include_research só para segmentação exploratória", () => {
    expect(segmentationFrameUrl("a1", 3, 2, 512, true)).toBe(
      "/api/v1/analyses/a1/segmentations/3/frames/2?max_dim=512&include_research=true",
    );
  });
});
