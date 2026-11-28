/**
 * Estado do modo silencioso visto pela interface (REQ-195).
 *
 * Falha fechada: se ``/api/v1/status`` não puder ser lido (erro de rede, resposta sem
 * ``shadow_mode``), o estado é ``"unknown"`` e a interface mostra uma faixa neutra —
 * nunca some com a faixa como se o modo estivesse desligado. A redação da saída de IA
 * continua sendo aplicada pela API em qualquer caso.
 */
export type ShadowState = "on" | "off" | "unknown";

/** Estado a partir da leitura de ``/status``; ``null`` = a leitura falhou. */
export function shadowStateFrom(status: { shadow_mode?: unknown } | null | undefined): ShadowState {
  if (status === null || status === undefined) return "unknown";
  if (status.shadow_mode === true) return "on";
  if (status.shadow_mode === false) return "off";
  return "unknown";
}
