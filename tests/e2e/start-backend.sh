#!/usr/bin/env bash
# Sobe o backend REAL (uvicorn + SQLite) para os testes de ponta a ponta.
#
# Cria um diretório de trabalho descartável, aplica as migrações, cria dois
# usuários pela CLI (sem credencial padrão no produto: as senhas vêm do
# ambiente do teste) e executa `radiologyai serve` em primeiro plano.
#
# Variáveis aceitas:
#   E2E_WORKDIR        diretório de trabalho (padrão: mktemp)
#   E2E_BACKEND_PORT   porta da API (padrão: 8765)
#   E2E_ADMIN_PASSWORD / E2E_PHYSICIAN_PASSWORD   senhas dos usuários de teste
#   E2E_RESEARCHER_PASSWORD  opcional: cria também e2e-pesquisa (papel researcher, shadow:view)
#   MEDAI_SHADOW_MODE  true = modo silencioso do piloto (REQ-195); padrão false
#   MEDAI_WEIGHTS_DIR  pesos verificados (padrão: /root/medai-cache/weights/registry)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${E2E_PYTHON:-$ROOT/.venv/bin/python}"
WORK="${E2E_WORKDIR:-$(mktemp -d -t radiologyai-e2e-XXXXXX)}"
PORT="${E2E_BACKEND_PORT:-8765}"
mkdir -p "$WORK/store"

export MEDAI_DATABASE_URL="sqlite:///$WORK/platform.sqlite"
export MEDAI_STORE_DIR="$WORK/store"
# Chave HMAC de des-identificação só deste banco descartável.
export MEDAI_DEIDENT_KEY="${MEDAI_DEIDENT_KEY:-$("$PY" -c 'import secrets; print(secrets.token_hex(32))')}"
export MEDAI_WEIGHTS_DIR="${MEDAI_WEIGHTS_DIR:-/root/medai-cache/weights/registry}"
export MEDAI_DEVICE="${MEDAI_DEVICE:-cpu}"
export MEDAI_WORKER_POLL_SECONDS="${MEDAI_WORKER_POLL_SECONDS:-0.5}"
# Explícito: um MEDAI_SHADOW_MODE herdado por acaso não muda o fluxo normal.
export MEDAI_SHADOW_MODE="${MEDAI_SHADOW_MODE:-false}"

: "${E2E_ADMIN_PASSWORD:?defina E2E_ADMIN_PASSWORD}"
: "${E2E_PHYSICIAN_PASSWORD:?defina E2E_PHYSICIAN_PASSWORD}"

cd "$WORK"
"$PY" -m radiologyai.cli.main migrate
printf '%s\n' "$E2E_ADMIN_PASSWORD" | "$PY" -m radiologyai.cli.main users create \
  --username e2e-admin --full-name "Administrador E2E" --role admin --password-stdin
printf '%s\n' "$E2E_PHYSICIAN_PASSWORD" | "$PY" -m radiologyai.cli.main users create \
  --username e2e-medico --full-name "Médica E2E" --role physician \
  --professional-id "CRM-TESTE-0000" --password-stdin
if [[ -n "${E2E_RESEARCHER_PASSWORD:-}" ]]; then
  printf '%s\n' "$E2E_RESEARCHER_PASSWORD" | "$PY" -m radiologyai.cli.main users create \
    --username e2e-pesquisa --full-name "Pesquisadora E2E" --role researcher --password-stdin
fi

exec "$PY" -m radiologyai.cli.main serve --host 127.0.0.1 --port "$PORT"
