#!/usr/bin/env node
/**
 * Teste de ponta a ponta contra o backend REAL (uvicorn + SQLite + worker em
 * processo + modelos reais) e a interface compilada (vite preview).
 *
 * Fluxo: login -> envio de um DICOM de tórax real -> análise -> visualizador
 * com sobreposições -> revisão de cada achado -> laudo -> assinatura -> PDF.
 *
 * Sem @playwright/test: usa a biblioteca ``playwright`` já instalada no
 * sistema (nunca rode ``playwright install``). Um único navegador.
 *
 * Variáveis:
 *   E2E_DICOM          arquivo DICOM de tórax (padrão: primeiro de samples-cxr-dicom)
 *   E2E_MODELS         ids de modelos separados por vírgula (padrão abaixo); "auto" = roteamento
 *   E2E_PATIENT_AGE    idade informada no envio (padrão 60; "" = não informar)
 *   E2E_SCREENS_DIR    onde salvar capturas (padrão: <workdir>/screens)
 *   E2E_WORKDIR        diretório descartável (padrão: mkdtemp)
 *   E2E_BACKEND_PORT / E2E_UI_PORT
 *   E2E_ANALYSIS_TIMEOUT_MS (padrão 600000)
 *   E2E_EXPECT_HOLD    "1" = exige ao menos um achado "Só escore" com nota de decisão suspensa (REQ-193)
 *   E2E_SHADOW         "1" = modo silencioso do piloto (REQ-195): sobe o backend com
 *                      MEDAI_SHADOW_MODE=true e verifica, na interface real, que a médica e o
 *                      administrador não recebem saída de IA (nem na tela, nem em resposta HTTP),
 *                      que a pesquisadora (papel researcher, shadow:view) vê os achados e que
 *                      ninguém exporta laudo ao PACS. Único erro HTTP aceito: 403 ShadowModeError.
 *
 * Os modelos pedidos em E2E_MODELS são EXATAMENTE os da análise: modelo de rota
 * "research" (ex.: xrv-densenet121-res224-chex) só aparece no seletor com o modo
 * pesquisa ligado, que o teste liga para escolhê-lo; modelo que não aparece mesmo
 * assim é falha (nunca cai em silêncio no roteamento automático). O corpo do POST
 * da análise é conferido.
 *
 * Saída 0 = fluxo completo; qualquer falha encerra com código 1 e mensagem.
 * Ausência do DICOM ou dos pesos => código 77 (pulado), nunca sucesso falso.
 */
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const FRONTEND = join(ROOT, "frontend");
const SKIP = 77;

const BACKEND_PORT = Number(process.env.E2E_BACKEND_PORT ?? 8765);
const UI_PORT = Number(process.env.E2E_UI_PORT ?? 4173);
const ANALYSIS_TIMEOUT = Number(process.env.E2E_ANALYSIS_TIMEOUT_MS ?? 600_000);
const DEFAULT_MODELS = "xrv-densenet121-res224-pc,xrv-pspnet-chestxdet,xr-probe-tb";
const PATIENT_AGE = process.env.E2E_PATIENT_AGE ?? "60";
const SHADOW = process.env.E2E_SHADOW === "1";
const EXPECT_HOLD = process.env.E2E_EXPECT_HOLD === "1";
const MODELS = (process.env.E2E_MODELS ?? DEFAULT_MODELS)
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

function log(msg) {
  process.stdout.write(`[e2e ${new Date().toISOString().slice(11, 19)}] ${msg}\n`);
}

function findDicom() {
  if (process.env.E2E_DICOM) return process.env.E2E_DICOM;
  const base = "/root/medai-cache/datasets/samples-cxr-dicom";
  if (!existsSync(base)) return null;
  for (const d of readdirSync(base).sort()) {
    const dir = join(base, d);
    if (!statSync(dir).isDirectory()) continue;
    const f = readdirSync(dir).find((x) => x.endsWith(".dcm"));
    if (f) return join(dir, f);
  }
  return null;
}

async function waitHttp(url, timeoutMs, proc) {
  const t0 = Date.now();
  for (;;) {
    if (proc && proc.exitCode !== null) throw new Error(`processo encerrou (código ${proc.exitCode}) antes de ${url}`);
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {
      // ainda subindo
    }
    if (Date.now() - t0 > timeoutMs) throw new Error(`tempo esgotado esperando ${url}`);
    await new Promise((r) => setTimeout(r, 500));
  }
}

function start(name, cmd, args, env, cwd) {
  const p = spawn(cmd, args, { cwd, env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] });
  const tail = [];
  const keep = (chunk) => {
    for (const line of String(chunk).split("\n")) {
      if (!line) continue;
      tail.push(line);
      if (tail.length > 200) tail.shift();
      if (process.env.E2E_VERBOSE) process.stdout.write(`  [${name}] ${line}\n`);
    }
  };
  p.stdout.on("data", keep);
  p.stderr.on("data", keep);
  p.tail = tail;
  return p;
}

function stop(p) {
  if (p && p.exitCode === null) p.kill("SIGTERM");
}


async function login(page, username, password) {
  await page.locator('input[name="username"]').waitFor();
  await page.locator('input[name="username"]').fill(username);
  await page.locator('input[name="password"]').fill(password);
  await page.locator('button[type="submit"]').click();
  await page.getByTestId("worklist-table").waitFor();
}

async function logout(page) {
  await page.getByRole("button", { name: /Sair|Sign out/ }).click();
  await page.locator('input[name="username"]').waitFor();
}

/** Envia o DICOM real pela interface e espera o primeiro quadro. */
async function uploadStudy(page, dicom) {
  await page.getByTestId("upload-files").setInputFiles(dicom);
  await page.getByTestId("upload-exam-type").selectOption("cxr");
  if (PATIENT_AGE) {
    // Metadado informado pelo usuário (o DICOM de exemplo vem sem idade e os
    // modelos de adulto se abstêm, corretamente, sem ela).
    await page.getByTestId("upload-more").locator("summary").click();
    await page.getByTestId("upload-age").fill(PATIENT_AGE);
  }
  await page.getByTestId("upload-submit").click();
  await page.waitForURL(/#\/studies\//, { timeout: 120_000 });
  await page.locator('[data-testid="viewer-canvas"][data-frame-ready="true"]').waitFor({ timeout: 60_000 });
}

/**
 * Marca no seletor EXATAMENTE os modelos de ``MODELS``. Modelo de rota "research"
 * só é listado com o modo pesquisa ligado: liga-o se preciso. Devolve se ligou.
 */
async function chooseModels(page) {
  if (MODELS.length === 0 || MODELS[0] === "auto") return false;
  const chooser = page.locator("details.model-chooser");
  await chooser.waitFor({ timeout: 30_000 });
  const openChooser = () => page.evaluate(() => document.querySelector("details.model-chooser")?.setAttribute("open", ""));
  await openChooser();
  let researchOn = false;
  const missing = [];
  for (const id of MODELS) {
    if ((await page.getByTestId(`choose-model-${id}`).count()) === 0) missing.push(id);
  }
  const toggle = page.getByTestId("research-toggle");
  if (missing.length > 0 && (await toggle.count()) > 0 && !(await toggle.isChecked())) {
    await toggle.check();
    researchOn = true;
    await openChooser();
    log(`modo pesquisa ligado para listar modelo(s) de rota exploratória: ${missing.join(", ")}`);
  }
  for (const id of MODELS) {
    const box = page.getByTestId(`choose-model-${id}`);
    if ((await box.count()) === 0) {
      throw new Error(`modelo ${id} não está disponível/roteável no seletor (E2E_MODELS não seria respeitado)`);
    }
    await box.check();
  }
  // Nada além do pedido (defensivo: um padrão pré-marcado mudaria a análise).
  const checked = await page
    .locator('details.model-chooser input[type="checkbox"]:checked')
    .evaluateAll((els) => els.map((e) => e.value));
  for (const id of checked) {
    if (!MODELS.includes(id)) await page.getByTestId(`choose-model-${id}`).uncheck();
  }
  const finalSet = await page
    .locator('details.model-chooser input[type="checkbox"]:checked')
    .evaluateAll((els) => els.map((e) => e.value).sort());
  if (JSON.stringify(finalSet) !== JSON.stringify([...MODELS].sort())) {
    throw new Error(`seleção de modelos divergente: ${finalSet.join(",")} != ${MODELS.join(",")}`);
  }
  log(`seleção: ${(await chooser.locator("summary").innerText()).trim()} — ${finalSet.join(", ")}`);
  return researchOn;
}

/** Clica em Analisar, confere o corpo do POST e espera o fim. Devolve a análise criada. */
async function runAnalysis(page) {
  const t0 = Date.now();
  const [req] = await Promise.all([
    page.waitForRequest((r) => r.method() === "POST" && /\/api\/v1\/studies\/[^/]+\/analyses$/.test(r.url())),
    page.getByTestId("analyze").click(),
  ]);
  const body = JSON.parse(req.postData() ?? "{}");
  if (MODELS.length > 0 && MODELS[0] !== "auto") {
    const sent = [...(body.model_ids ?? [])].sort();
    if (JSON.stringify(sent) !== JSON.stringify([...MODELS].sort())) {
      throw new Error(`POST da análise com model_ids=${JSON.stringify(body.model_ids)}; esperado ${MODELS.join(",")}`);
    }
  } else if (body.model_ids) {
    throw new Error(`roteamento automático pedido, mas o POST enviou model_ids=${JSON.stringify(body.model_ids)}`);
  }
  const created = await (await req.response()).json();
  log(`análise ${created.id} pedida com model_ids=${JSON.stringify(created.model_ids)}`);
  await page.waitForFunction(
    () => {
      const el = document.querySelector('[data-testid="analysis-state"]');
      return el !== null && /concluída|falhou/.test(el.textContent ?? "");
    },
    null,
    { timeout: ANALYSIS_TIMEOUT, polling: 1000 },
  );
  const state = await page.getByTestId("analysis-state").innerText();
  log(`análise terminou em ${((Date.now() - t0) / 1000).toFixed(1)} s: ${state}`);
  if (!state.includes("concluída")) throw new Error(`análise não concluiu: ${state}`);
  return created;
}

/** ``fetch`` dentro da página com o token da sessão (mesmo caminho da interface). */
async function apiCall(page, method, path) {
  return page.evaluate(
    async ({ method, path }) => {
      const token = sessionStorage.getItem("radiologyai.token");
      const r = await fetch(path, { method, headers: token ? { Authorization: `Bearer ${token}` } : {} });
      const text = await r.text();
      let json = null;
      try {
        json = JSON.parse(text);
      } catch {
        // corpo não JSON
      }
      return { status: r.status, json, text };
    },
    { method, path },
  );
}

/** Chaves que só existem com saída de IA; valor não vazio numa resposta ao cuidado = vazamento. */
const AI_KEYS = new Set([
  "result",
  "findings",
  "explanations",
  "segmentations",
  "raw_score",
  "probability",
  "result_sha256",
  "result_status",
  "n_main_findings",
  "n_research_findings",
  "n_findings",
  "finding_indices",
  "segmentation_indices",
  "research_segmentation_indices",
  "research_measurement_names",
  "reviews",
  "hidden_research_findings",
  "hidden_research_artifacts",
  "unreviewed",
  "demoted",
]);

function aiLeaks(value, path = "$", out = []) {
  if (Array.isArray(value)) {
    value.forEach((v, i) => aiLeaks(v, `${path}[${i}]`, out));
  } else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) {
      const empty =
        v === null ||
        v === false ||
        (Array.isArray(v) && v.length === 0) ||
        (typeof v === "object" && v !== null && !Array.isArray(v) && Object.keys(v).length === 0) ||
        (v === 0 && k.startsWith("hidden_"));
      if (AI_KEYS.has(k) && !empty) out.push(`${path}.${k}=${JSON.stringify(v).slice(0, 80)}`);
      aiLeaks(v, `${path}.${k}`, out);
    }
  }
  return out;
}

/**
 * Modo silencioso (REQ-195). A médica pede a análise e não vê saída de IA; nenhuma
 * resposta HTTP a ela (ou ao administrador) carrega saída de IA; a pesquisadora
 * vê os achados; ninguém exporta laudo ao PACS.
 */
async function shadowFlow({ page, base, screens, dicom, adminPass, docPass, resPass }) {
  let role = "anon";
  const responses = [];
  const consoleErrors = [];
  page.on("response", (r) => {
    const url = r.url();
    if (!url.includes("/api/")) return;
    const entry = { role, url, method: r.request().method(), status: r.status(), text: null };
    responses.push(entry);
    const ct = r.headers()["content-type"] ?? "";
    if (ct.includes("json")) {
      entry.body = r.text().then(
        (t) => {
          entry.text = t;
        },
        () => undefined,
      );
    }
  });
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  const PROBE = "e2e_probe=1";

  // 1. Médica: faixa do modo silencioso, envio, análise.
  await page.goto(`${base}/#/login`);
  await page.getByTestId("ruo-banner").waitFor();
  await login(page, "e2e-medico", docPass);
  role = "physician";
  const shadowBanner = await page.getByTestId("shadow-banner").innerText();
  if (!shadowBanner.includes("Modo silencioso")) throw new Error(`faixa do modo silencioso ausente: ${shadowBanner}`);
  log(`médica: faixa "${shadowBanner.split("\n")[0]}"`);
  await uploadStudy(page, dicom);
  const studyUrl = page.url();
  const researchForChoice = await chooseModels(page);
  const created = await runAnalysis(page);
  if (researchForChoice) await page.getByTestId("research-toggle").uncheck();
  await page.getByTestId("shadow-empty").waitFor();
  await page.waitForTimeout(1500); // deixa o visualizador pedir o que pediria
  const physicianUi = {
    findings: await page.locator('[data-testid^="finding-"]').count(),
    findingsPanel: await page.getByTestId("findings-panel").count(),
    reviews: await page.locator('[data-testid^="review-"]').count(),
    seg: await page.locator('[data-testid^="toggle-seg-"]').count(),
    heatmap: await page.getByTestId("heatmap-select").count(),
    heatDrawn: await page.locator('[data-testid="viewer-canvas"][data-heatmap-drawn="true"]').count(),
    segDrawn: await page.locator('[data-testid="viewer-canvas"][data-seg-drawn="true"]').count(),
    reportCreate: await page.getByTestId("report-create").count(),
    reportCard: await page.getByTestId("report-card").count(),
    decisionNotes: await page.getByTestId("decision-note").count(),
  };
  log(`médica, tela do estudo: ${JSON.stringify(physicianUi)}`);
  for (const [k, v] of Object.entries(physicianUi)) {
    if (v !== 0) throw new Error(`modo silencioso: a médica vê ${k}=${v}`);
  }
  const stateText = (await page.locator(".analysis-state").innerText()).replace(/\s+/g, " ");
  if (/achado|positivo|negativo|abstenção|Resultado/i.test(stateText)) {
    throw new Error(`modo silencioso: o estado da análise revela o resultado: ${stateText}`);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: join(screens, "00-medico-modo-silencioso.png"), fullPage: true });
  await page.goto(`${base}/#/worklist`);
  await page.getByTestId("worklist-table").locator("tbody tr").first().waitFor();
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(screens, "01-medico-lista-de-trabalho.png"), fullPage: false });

  // Sondas diretas (o mesmo token da interface): laudo e envio ao PACS retidos.
  const rep = await apiCall(page, "POST", `/api/v1/analyses/${created.id}/reports?${PROBE}`);
  if (rep.status !== 403 || rep.json?.error !== "ShadowModeError") {
    throw new Error(`médica criou laudo no modo silencioso: HTTP ${rep.status} ${rep.text.slice(0, 200)}`);
  }
  const expDoc = await apiCall(page, "POST", `/api/v1/reports/qualquer/export?${PROBE}`);
  if (expDoc.status !== 403 || expDoc.json?.error !== "ShadowModeError") {
    throw new Error(`envio ao PACS (médica) não bloqueado: HTTP ${expDoc.status} ${expDoc.text.slice(0, 200)}`);
  }
  log("médica: criar laudo => 403 ShadowModeError; exportar ao PACS => 403 ShadowModeError");
  await logout(page);

  // 2. Pesquisadora (shadow:view): vê os achados; também não exporta.
  role = "anon";
  await login(page, "e2e-pesquisa", resPass);
  role = "researcher";
  await page.getByTestId("shadow-banner").waitFor();
  const researcherView = await apiCall(page, "GET", `/api/v1/analyses/${created.id}?${PROBE}`);
  if (researcherView.status !== 200 || !researcherView.json?.result) {
    throw new Error(`pesquisadora sem o resultado: HTTP ${researcherView.status}`);
  }
  await page.goto(studyUrl);
  await page.getByTestId("findings-panel").waitFor({ timeout: 30_000 });
  const nResFindings = await page.locator('[data-testid^="finding-"]').count();
  if (nResFindings === 0) throw new Error("pesquisadora não vê achados no modo silencioso");
  const resUi = {
    reviews: await page.locator('[data-testid^="review-"] button, [data-testid^="rv-accept-"]').count(),
    reportCreate: await page.getByTestId("report-create").count(),
    exportControls: await page.locator('[data-testid*="export"]').count(),
  };
  log(`pesquisadora: ${nResFindings} achado(s) exibido(s); controles ${JSON.stringify(resUi)}`);
  if (resUi.reportCreate !== 0 || resUi.exportControls !== 0 || resUi.reviews !== 0) {
    throw new Error(`pesquisadora com controles de cuidado no modo silencioso: ${JSON.stringify(resUi)}`);
  }
  const heat = page.getByTestId("heatmap-select");
  if ((await heat.count()) > 0) {
    await page.locator('[data-testid="viewer-canvas"][data-heatmap-drawn="true"]').waitFor({ timeout: 30_000 });
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(screens, "02-pesquisadora-achados.png"), fullPage: false });
  const expRes = await apiCall(page, "POST", `/api/v1/reports/qualquer/export?${PROBE}`);
  if (expRes.status !== 403) throw new Error(`envio ao PACS (pesquisadora) não bloqueado: HTTP ${expRes.status}`);
  log(`pesquisadora: exportar ao PACS => HTTP ${expRes.status} (${expRes.json?.error ?? expRes.json?.detail ?? ""})`);
  await logout(page);

  // 3. Administrador: trilha íntegra, contagens da IA redigidas, bloqueios auditados.
  role = "anon";
  await login(page, "e2e-admin", adminPass);
  role = "admin";
  await page.goto(`${base}/#/audit`);
  await page.getByTestId("audit-table").locator("tbody tr").first().waitFor();
  await page.getByTestId("audit-verify").click();
  const verdict = await page.getByTestId("audit-verify-result").innerText();
  if (!/íntegra|intact/i.test(verdict)) throw new Error(`cadeia de auditoria: ${verdict}`);
  await page.screenshot({ path: join(screens, "03-admin-auditoria.png"), fullPage: false });
  const completed = await apiCall(page, "GET", "/api/v1/audit?event=analysis.completed");
  const ev = completed.json?.items?.find((e) => e.subject_id === created.id);
  if (!ev) throw new Error("analysis.completed ausente da trilha");
  if (ev.payload.n_findings !== null || !ev.payload.shadow_redacted?.includes("n_findings")) {
    throw new Error(`administrador vê n_findings no modo silencioso: ${JSON.stringify(ev.payload).slice(0, 300)}`);
  }
  if (ev.record_hash !== "") throw new Error("administrador recebe o record_hash do evento redigido");
  const blocked = (await apiCall(page, "GET", "/api/v1/audit?event=shadow.blocked&limit=500")).json?.items ?? [];
  const careExports = blocked.filter((e) => e.payload.care_export === true);
  if (careExports.length === 0) throw new Error("bloqueio de envio ao PACS não auditado");
  log(`administrador: ${verdict.trim()}; n_findings redigido; ${blocked.length} shadow.blocked (${careExports.length} de envio ao PACS)`);

  // 4. Nenhuma resposta à médica ou ao administrador carrega saída de IA.
  await Promise.all(responses.map((r) => r.body).filter(Boolean));
  const aiResult = researcherView.json.result;
  const secrets = [
    researcherView.json.result_sha256,
    ...(aiResult.explanations ?? []).map((e) => e.map?.sha256),
    ...(aiResult.segmentations ?? []).map((g) => g.mask?.sha256),
  ].filter(Boolean);
  // Autoverificação do detector: a resposta da pesquisadora TEM saída de IA.
  if (aiLeaks(researcherView.json).length === 0) throw new Error("detector de saída de IA inerte");
  const problems = [];
  let inspected = 0;
  for (const r of responses) {
    if (r.role !== "physician" && r.role !== "admin") continue;
    const probe = r.url.includes(PROBE);
    if (r.status >= 400 && !probe) {
      let err = null;
      try {
        err = JSON.parse(r.text ?? "null")?.error;
      } catch {
        // corpo não JSON
      }
      if (!(r.status === 403 && err === "ShadowModeError")) {
        problems.push(`HTTP ${r.status} ${r.method} ${r.url} (${(r.text ?? "").slice(0, 120)})`);
      }
    }
    if (r.status < 300 && /\/segmentations\//.test(r.url)) problems.push(`quadro de segmentação servido: ${r.url}`);
    for (const sec of secrets) {
      if (r.url.includes(sec) && r.status < 300) problems.push(`artefato de IA servido a ${r.role}: ${r.url}`);
      if (r.text?.includes(sec)) problems.push(`hash da saída de IA na resposta a ${r.role}: ${r.url}`);
    }
    if (r.text && r.status < 300) {
      inspected += 1;
      let json = null;
      try {
        json = JSON.parse(r.text);
      } catch {
        // não JSON
      }
      for (const leak of aiLeaks(json)) problems.push(`${r.role} ${r.method} ${r.url}: ${leak}`);
    }
  }
  const expected403 = responses.filter(
    (r) => (r.role === "physician" || r.role === "admin") && r.status === 403 && !r.url.includes(PROBE),
  );
  log(
    `respostas inspecionadas (médica+admin): ${inspected} JSON de ${responses.filter((r) => r.role === "physician" || r.role === "admin").length}; ` +
      `403 ShadowModeError esperados: ${expected403.length}${expected403.length ? ` (${[...new Set(expected403.map((r) => new URL(r.url).pathname.replace(/[0-9a-f-]{20,}/g, "…")))].join(", ")})` : ""}`,
  );
  if (problems.length > 0) throw new Error(`modo silencioso violado:\n  ${problems.slice(0, 20).join("\n  ")}`);
  const otherConsole = consoleErrors.filter((e) => !/favicon|403/i.test(e));
  if (otherConsole.length > 0) log(`avisos de console (${otherConsole.length}): ${otherConsole.slice(0, 5).join(" | ")}`);
  log(`OK (modo silencioso) — capturas em ${screens}`);
  return 0;
}

async function main() {
  const dicom = findDicom();
  if (!dicom || !existsSync(dicom)) {
    log("SKIP: nenhum DICOM de tórax disponível (E2E_DICOM ou samples-cxr-dicom)");
    return SKIP;
  }
  const weights = process.env.MEDAI_WEIGHTS_DIR ?? "/root/medai-cache/weights/registry";
  if (!existsSync(weights)) {
    log(`SKIP: diretório de pesos ausente: ${weights}`);
    return SKIP;
  }
  let playwright;
  try {
    playwright = require("playwright");
  } catch {
    log("SKIP: biblioteca playwright indisponível");
    return SKIP;
  }
  if (!existsSync(join(FRONTEND, "dist", "index.html"))) {
    throw new Error("frontend/dist ausente: rode `npm run build` em frontend/ antes");
  }

  const work = process.env.E2E_WORKDIR ?? mkdtempSync(join(tmpdir(), "radiologyai-e2e-"));
  const screens = process.env.E2E_SCREENS_DIR ?? join(work, "screens");
  mkdirSync(screens, { recursive: true });
  const adminPass = `Adm-${randomBytes(9).toString("base64url")}`;
  const docPass = `Med-${randomBytes(9).toString("base64url")}`;
  const resPass = `Pes-${randomBytes(9).toString("base64url")}`;
  log(
    `workdir ${work}; DICOM ${dicom}; modelos ${MODELS.join(", ") || "auto"}` +
      (SHADOW ? "; MODO SILENCIOSO (REQ-195)" : ""),
  );

  const procs = [];
  let browser = null;
  try {
    const backend = start(
      "api",
      "bash",
      [join(HERE, "start-backend.sh")],
      {
        E2E_WORKDIR: work,
        E2E_BACKEND_PORT: String(BACKEND_PORT),
        E2E_ADMIN_PASSWORD: adminPass,
        E2E_PHYSICIAN_PASSWORD: docPass,
        E2E_RESEARCHER_PASSWORD: SHADOW ? resPass : "",
        MEDAI_SHADOW_MODE: SHADOW ? "true" : "false",
        MEDAI_WEIGHTS_DIR: weights,
      },
      ROOT,
    );
    procs.push(backend);
    await waitHttp(`http://127.0.0.1:${BACKEND_PORT}/api/v1/healthz`, 120_000, backend).catch((e) => {
      throw new Error(`${e.message}\n--- backend ---\n${backend.tail.join("\n")}`);
    });
    log("backend no ar");

    const ui = start(
      "ui",
      join(FRONTEND, "node_modules", ".bin", "vite"),
      ["preview", "--host", "127.0.0.1", "--port", String(UI_PORT), "--strictPort"],
      { RADIOLOGYAI_BACKEND: `http://127.0.0.1:${BACKEND_PORT}` },
      FRONTEND,
    );
    procs.push(ui);
    await waitHttp(`http://127.0.0.1:${UI_PORT}/`, 60_000, ui);
    log("interface no ar");

    browser = await playwright.chromium.launch({ headless: true });
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
      deviceScaleFactor: 2,
      acceptDownloads: true,
      locale: "pt-BR",
    });
    const page = await context.newPage();
    const consoleErrors = [];
    page.on("response", (r) => {
      if (r.status() >= 400) consoleErrors.push(`HTTP ${r.status()} ${r.request().method()} ${r.url()}`);
    });
    page.on("console", (m) => {
      if (m.type() === "error") consoleErrors.push(m.text());
    });
    page.setDefaultTimeout(30_000);
    const base = `http://127.0.0.1:${UI_PORT}`;
    if (SHADOW) return await shadowFlow({ page, base, screens, dicom, adminPass, docPass, resPass });

    // 1. Login (médica).
    await page.goto(`${base}/#/login`);
    await page.getByTestId("ruo-banner").waitFor();
    const banner = await page.getByTestId("ruo-banner").innerText();
    if (!banner.includes("não é dispositivo médico")) throw new Error(`faixa de pesquisa ausente: ${banner}`);
    await login(page, "e2e-medico", docPass);
    log("login ok");

    // 2. Envio do DICOM real.
    await uploadStudy(page, dicom);
    log("estudo enviado; quadro exibido");

    // 3. Análise com EXATAMENTE os modelos pedidos (ou roteamento automático).
    const researchForChoice = await chooseModels(page);
    await runAnalysis(page);
    if (researchForChoice) {
      // Volta à visão principal: o resto do fluxo é o do médico assistente.
      await page.getByTestId("research-toggle").uncheck();
      await page.getByTestId("research-banner").waitFor({ state: "detached" });
    }
    await page.getByTestId("findings-panel").waitFor();
    const resultText = await page.locator(".analysis-state").innerText();
    const nFindings = await page.locator('[data-testid^="finding-"]').count();
    log(`resultado: ${resultText.replace(/\s+/g, " ")}; achados exibidos: ${nFindings}`);
    // Achado com decisão suspensa (REQ-193) mostra o motivo no próprio cartão.
    const notes = page.getByTestId("decision-note");
    const nNotes = await notes.count();
    if (nNotes > 0) log(`nota de decisão: ${(await notes.first().innerText()).replace(/\s+/g, " ").slice(0, 160)}`);
    // Achado "só escore" sem motivo no cartão é regressão (REQ-193): todo score_only explica.
    const scoreOnly = await page.locator('[data-testid^="finding-"]').filter({ hasText: "Só escore" }).count();
    if (scoreOnly > nNotes) {
      throw new Error(`${scoreOnly - nNotes} achado(s) só com escore sem nota de motivo (REQ-193)`);
    }
    if (EXPECT_HOLD) {
      const held = page
        .locator('[data-testid^="finding-"]')
        .filter({ hasText: "Só escore" })
        .filter({ has: page.locator('[data-testid="decision-note"]', { hasText: "decisão suspensa" }) });
      const nHeld = await held.count();
      if (nHeld === 0) throw new Error("E2E_EXPECT_HOLD: nenhum achado \"Só escore\" com decisão suspensa");
      for (let i = 0; i < nHeld; i++) {
        log(`decisão suspensa exibida: ${(await held.nth(i).innerText()).replace(/\s+/g, " ").slice(0, 240)}`);
      }
    }
    if (/abstenção/.test(resultText)) {
      throw new Error("abstenção total: o fluxo não exercitou inferência real");
    }
    // Achados de nível D só aparecem no modo pesquisa (papel com research:view).
    const hidden = page.getByTestId("hidden-research");
    let researchFindings = 0;
    // Visão padrão (sem modo pesquisa): captura e conta as camadas de segmentação expostas.
    const defaultSeg = await page.locator('[data-testid^="toggle-seg-"]').count();
    log(`visão principal: ${defaultSeg} camada(s) de segmentação disponível(is)`);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: join(screens, "00-visao-principal.png"), fullPage: true });
    if ((await hidden.count()) > 0) {
      log(`visão principal: ${(await hidden.innerText()).replace(/\s+/g, " ")}`);
      await page.getByTestId("research-toggle").check();
      await page.getByTestId("research-banner").waitFor();
      await page.locator(".group-level-d").first().waitFor({ timeout: 30_000 });
      researchFindings = await page.locator('.group-level-d [data-testid^="finding-"]').count();
      log(`modo pesquisa: ${researchFindings} achado(s) de nível D exibidos sob a faixa de pesquisa`);
      if (await page.locator('.group-level-d [data-testid^="review-"]').count()) {
        throw new Error("achado de nível D com controles de revisão");
      }
    }
    if (nFindings + researchFindings === 0) {
      throw new Error("nenhum achado calculado: o fluxo não exercitou inferência real");
    }

    // 4. Sobreposições: mapa de calor (se o modelo gerou) e segmentação (se houver).
    const heatSelect = page.getByTestId("heatmap-select");
    let heatmapShown = false;
    if ((await heatSelect.count()) > 0) {
      await page.locator('[data-testid="viewer-canvas"][data-heatmap-drawn="true"]').waitFor({ timeout: 30_000 });
      heatmapShown = true;
    }
    const segToggles = page.locator('[data-testid^="toggle-seg-"]');
    let segShown = false;
    if ((await segToggles.count()) > 0) {
      await segToggles.first().check();
      await page.locator('[data-testid="viewer-canvas"][data-seg-drawn="true"]').waitFor({ timeout: 30_000 });
      segShown = true;
    }
    log(`sobreposições: mapa de calor=${heatmapShown} segmentação=${segShown}`);
    // Fecha o seletor de modelos e enquadra o visualizador para a captura.
    await page.evaluate(() => document.querySelector("details.model-chooser")?.removeAttribute("open"));
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(300);
    await page.screenshot({ path: join(screens, "01-visualizador-achados.png"), fullPage: false });
    if (researchFindings > 0) {
      await page.getByTestId("research-toggle").uncheck();
      await page.getByTestId("research-banner").waitFor({ state: "detached" });
    }

    // 5. Revisão: aceita cada achado pendente (o servidor exige o hash do resultado).
    let reviewed = 0;
    for (let guard = 0; guard < 200; guard++) {
      const pendingBtns = page.locator('[data-testid^="review-"]').filter({
        hasNot: page.locator(".review-current"),
      });
      if ((await pendingBtns.count()) === 0) break;
      const ctl = pendingBtns.first();
      const testId = await ctl.getAttribute("data-testid");
      const idx = (testId ?? "").replace("review-", "");
      await page.getByTestId(`rv-accept-${idx}`).click();
      await page.getByTestId(`rv-submit-${idx}`).click();
      await page.locator(`[data-testid="review-${idx}"] .review-current`).waitFor();
      reviewed += 1;
    }
    log(`achados revisados: ${reviewed}`);

    // 6. Laudo: gerar, assinar e baixar o PDF.
    await page.getByTestId("report-create").click();
    await page.getByTestId("report-card").first().waitFor();
    await page.getByTestId("sign-password").fill(docPass);
    await page.getByTestId("sign-submit").click();
    await page.getByTestId("report-signed").waitFor({ timeout: 60_000 });
    log("laudo assinado");
    await page.getByTestId("report-card").first().evaluate((el) => el.scrollIntoView({ block: "center" }));
    await page.waitForTimeout(200);
    await page.screenshot({ path: join(screens, "02-revisao-laudo-assinado.png"), fullPage: false });

    const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("download-pdf").click()]);
    const pdfPath = join(work, "laudo.pdf");
    await download.saveAs(pdfPath);
    const head = readFileSync(pdfPath).subarray(0, 5).toString("latin1");
    const size = statSync(pdfPath).size;
    if (head !== "%PDF-" || size < 1000) throw new Error(`PDF inválido (${head}, ${size} B)`);
    log(`PDF baixado: ${size} B`);

    // 7. Catálogo.
    await page.goto(`${base}/#/catalog`);
    await page.getByTestId("catalog-table").waitFor();
    const hiddenD = page.getByTestId("catalog-hidden-d");
    if ((await hiddenD.count()) > 0) {
      log(`catálogo: ${(await hiddenD.innerText()).trim()}`);
      await page.getByTestId("research-toggle").check();
      await page.getByTestId("research-banner").waitFor();
    }
    const nCond = await page.getByTestId("catalog-table").locator("tbody tr").count();
    log(`catálogo: ${nCond} linha(s) exibida(s)`);
    await page.screenshot({ path: join(screens, "03-catalogo.png"), fullPage: false });
    if ((await hiddenD.count()) === 0 && (await page.getByTestId("research-banner").count()) > 0) {
      await page.getByTestId("research-toggle").uncheck();
    }

    // 8. Auditoria (administrador) com verificação da cadeia pelo servidor.
    await logout(page);
    await login(page, "e2e-admin", adminPass);
    await page.goto(`${base}/#/audit`);
    await page.getByTestId("audit-table").locator("tbody tr").first().waitFor();
    await page.getByTestId("audit-verify").click();
    const verdict = await page.getByTestId("audit-verify-result").innerText();
    if (!/íntegra|intact/i.test(verdict)) throw new Error(`cadeia de auditoria: ${verdict}`);
    await page.screenshot({ path: join(screens, "04-auditoria.png"), fullPage: false });
    log(`auditoria: ${verdict}`);

    const relevant = consoleErrors.filter((e) => !/favicon/i.test(e));
    if (relevant.length > 0) log(`avisos de console (${relevant.length}): ${relevant.slice(0, 5).join(" | ")}`);
    const httpErrors = relevant.filter((e) => e.startsWith("HTTP "));
    if (httpErrors.length > 0) throw new Error(`respostas HTTP de erro durante o fluxo: ${httpErrors.join(" | ")}`);
    log(`OK — capturas em ${screens}`);
    return 0;
  } catch (err) {
    for (const p of procs) {
      if (p.tail.length) process.stderr.write(`--- últimas linhas ---\n${p.tail.slice(-40).join("\n")}\n`);
    }
    throw err;
  } finally {
    if (browser) await browser.close();
    for (const p of procs.reverse()) stop(p);
  }
}

main().then(
  (code) => process.exit(code),
  (err) => {
    process.stderr.write(`[e2e] FALHA: ${err instanceof Error ? err.stack : String(err)}\n`);
    process.exit(1);
  },
);
