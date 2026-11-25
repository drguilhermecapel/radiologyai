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
  log(`workdir ${work}; DICOM ${dicom}; modelos ${MODELS.join(", ") || "auto"}`);

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
    page.on("console", (m) => {
      if (m.type() === "error") consoleErrors.push(m.text());
    });
    page.setDefaultTimeout(30_000);
    const base = `http://127.0.0.1:${UI_PORT}`;

    // 1. Login (médica).
    await page.goto(`${base}/#/login`);
    await page.getByTestId("ruo-banner").waitFor();
    const banner = await page.getByTestId("ruo-banner").innerText();
    if (!banner.includes("não é dispositivo médico")) throw new Error(`faixa de pesquisa ausente: ${banner}`);
    await page.locator('input[name="username"]').fill("e2e-medico");
    await page.locator('input[name="password"]').fill(docPass);
    await page.locator('button[type="submit"]').click();
    await page.getByTestId("worklist-table").waitFor();
    log("login ok");

    // 2. Envio do DICOM real.
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
    log("estudo enviado; quadro exibido");

    // 3. Análise (modelos escolhidos, se roteáveis; senão automático).
    if (MODELS.length > 0 && MODELS[0] !== "auto") {
      const chooser = page.locator("details.model-chooser");
      await chooser.waitFor({ timeout: 30_000 }).catch(() => undefined);
      if ((await chooser.count()) === 0) {
        log("aviso: nenhum modelo roteável listado; análise em modo automático");
      } else {
        await chooser.locator("summary").click();
        for (const id of MODELS) {
          const box = page.getByTestId(`choose-model-${id}`);
          if ((await box.count()) > 0) await box.check();
          else log(`aviso: modelo ${id} não está disponível/roteável; ignorado`);
        }
      }
    }
    if (MODELS[0] !== "auto") {
      const summary = await page.locator("details.model-chooser summary").innerText().catch(() => "");
      log(`seleção: ${summary}`);
    }
    const t0 = Date.now();
    await page.getByTestId("analyze").click();
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
    await page.getByTestId("findings-panel").waitFor();
    const resultText = await page.locator(".analysis-state").innerText();
    const nFindings = await page.locator('[data-testid^="finding-"]').count();
    log(`resultado: ${resultText.replace(/\s+/g, " ")}; achados exibidos: ${nFindings}`);
    if (/abstenção/.test(resultText)) {
      throw new Error("abstenção total: o fluxo não exercitou inferência real");
    }
    // Achados de nível D só aparecem no modo pesquisa (papel com research:view).
    const hidden = page.getByTestId("hidden-research");
    let researchFindings = 0;
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
    await page.getByRole("button", { name: /Sair|Sign out/ }).click();
    await page.locator('input[name="username"]').waitFor();
    await page.locator('input[name="username"]').fill("e2e-admin");
    await page.locator('input[name="password"]').fill(adminPass);
    await page.locator('button[type="submit"]').click();
    await page.getByTestId("worklist-table").waitFor();
    await page.goto(`${base}/#/audit`);
    await page.getByTestId("audit-table").locator("tbody tr").first().waitFor();
    await page.getByTestId("audit-verify").click();
    const verdict = await page.getByTestId("audit-verify-result").innerText();
    if (!/íntegra|intact/i.test(verdict)) throw new Error(`cadeia de auditoria: ${verdict}`);
    await page.screenshot({ path: join(screens, "04-auditoria.png"), fullPage: false });
    log(`auditoria: ${verdict}`);

    const relevant = consoleErrors.filter((e) => !/favicon/i.test(e));
    if (relevant.length > 0) log(`avisos de console (${relevant.length}): ${relevant.slice(0, 5).join(" | ")}`);
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
