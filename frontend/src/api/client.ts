/**
 * Cliente HTTP tipado (tipos gerados de docs/OPENAPI_SPEC.yaml).
 *
 * O token de sessão é opaco e vive só em ``sessionStorage`` (some ao fechar a
 * aba). Toda resposta 401 encerra a sessão local. Endpoints binários (quadros,
 * artefatos, documentos) usam :func:`fetchBinary` com o mesmo cabeçalho.
 */
import createClient, { type Middleware } from "openapi-fetch";

import type { Readable } from "openapi-typescript-helpers";

import type { components, paths } from "./schema";

type RawSchemas = components["schemas"];
/**
 * Tipos como o openapi-fetch os ENTREGA nas respostas (``Readable``): tuplas
 * como ``[number, number]`` viram ``number[]``. Usar estes tipos (e não os
 * crus) evita conversões inseguras entre resposta e componentes.
 */
export type Schemas = { [K in keyof RawSchemas]: Readable<RawSchemas[K]> };
export type UserOut = Schemas["UserOut"];

const TOKEN_KEY = "radiologyai.token";

let token: string | null = readToken();
const listeners = new Set<() => void>();

function readToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function getToken(): string | null {
  return token;
}

export function setToken(value: string | null): void {
  token = value;
  try {
    if (value === null) sessionStorage.removeItem(TOKEN_KEY);
    else sessionStorage.setItem(TOKEN_KEY, value);
  } catch {
    // armazenamento indisponível: a sessão vale só enquanto a página estiver aberta
  }
  listeners.forEach((fn) => fn());
}

/** Avisado quando o token muda (login, logout, 401). */
export function onTokenChange(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

const authMiddleware: Middleware = {
  onRequest({ request }) {
    if (token) request.headers.set("Authorization", `Bearer ${token}`);
    return request;
  },
  onResponse({ response, request }) {
    if (response.status === 401 && !request.url.endsWith("/auth/login") && token !== null) {
      setToken(null);
    }
    return response;
  },
};

export const api = createClient<paths>({ baseUrl: "" });
api.use(authMiddleware);

/** Erro de API com a mensagem do servidor (``ErrorOut.detail``). */
export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Extrai uma mensagem legível de um corpo de erro do servidor. */
export function errorMessage(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) =>
          d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d),
        )
        .join("; ");
    }
    return JSON.stringify(detail);
  }
  return `HTTP ${status}`;
}

/** Desembrulha o resultado do openapi-fetch ou lança :class:`ApiError`. */
export function unwrap<T>(res: { data?: T; error?: unknown; response: Response }): T {
  if (res.error !== undefined || res.data === undefined) {
    if (res.response.ok && res.data === undefined && res.error === undefined) {
      return undefined as T;
    }
    throw new ApiError(res.response.status, errorMessage(res.error, res.response.status));
  }
  return res.data;
}

/** GET binário autenticado (quadros, artefatos, PDF/SR/FHIR). */
export async function fetchBinary(url: string, signal?: AbortSignal): Promise<Response> {
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const init: RequestInit = { headers };
  if (signal) init.signal = signal;
  const response = await fetch(url, init);
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    if (response.status === 401 && token !== null) setToken(null);
    throw new ApiError(response.status, errorMessage(body, response.status));
  }
  return response;
}

/** POST multipart autenticado (envio de estudos); devolve o JSON da resposta. */
export async function postForm<T>(url: string, form: FormData): Promise<T> {
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(url, { method: "POST", headers, body: form });
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    if (response.status === 401 && token !== null) setToken(null);
    throw new ApiError(response.status, errorMessage(body, response.status));
  }
  return body as T;
}

/** Baixa um documento autenticado e dispara o "salvar como" do navegador. */
export async function downloadDocument(url: string, fallbackName: string): Promise<void> {
  const response = await fetchBinary(url);
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = /filename="([^"]+)"/.exec(disposition);
  const name = match?.[1] ?? fallbackName;
  const blob = await response.blob();
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 30_000);
}
