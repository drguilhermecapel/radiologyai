# RadiologyAI — imagem de inferência em CPU.
#
# Diferenças em relação ao Dockerfile do v1 (agora em legacy/):
#   - python:3.11-slim, não 3.9. O pacote exige >=3.11,<3.12; a imagem do v1
#     não conseguiria sequer instalá-lo.
#   - --workers 1, não 4. Cada worker uvicorn carrega sua própria cópia do
#     modelo: 4 workers custam 4x a RAM sem ganho no alvo CPU, e tornam a
#     ordenação do log de auditoria não-determinística entre processos.
#   - usuário não-root.
#   - nenhum segredo embutido; tudo vem do ambiente.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Torch de CPU: a imagem de GPU é ~5x maior e não serve ao alvo de implantação.
RUN pip install --no-cache-dir --upgrade pip build \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      ".[api,imaging,eval,ml]" \
 && pip install --no-cache-dir --no-deps .

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="RadiologyAI" \
      org.opencontainers.image.description="Software de pesquisa. NAO e dispositivo medico." \
      org.opencontainers.image.source="https://github.com/drguilhermecapel/radiologyai"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEDAI_ARTIFACTS_DIR=/app/artifacts

RUN groupadd --system --gid 1001 radiologyai \
 && useradd --system --uid 1001 --gid radiologyai --create-home radiologyai

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/radiologyai /usr/local/bin/radiologyai

WORKDIR /app
RUN mkdir -p /app/artifacts && chown -R radiologyai:radiologyai /app

USER radiologyai

# Falha alto se o núcleo estiver incompleto — nunca sobe degradado.
RUN radiologyai selftest

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health').read()"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "--factory", "radiologyai.api:create_app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
