---
doc_id: REG-05
title: Plano de Desenvolvimento de Software (IEC 62304 §5.1)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-02, REG-03]
---

# Plano de Desenvolvimento de Software

> Software de pesquisa, não é dispositivo médico. Este plano descreve o processo que o
> repositório **já executa** e o transforma em evidência de ciclo de vida (IEC 62304 Ed. 1.1,
> Classe C). Onde o processo ainda não existe, está marcado como pendente.

## 1. Modelo de ciclo de vida

Desenvolvimento incremental orientado por portões de evidência (G0–G6,
[`ROADMAP.md`](../../ROADMAP.md) §4). Cada incremento passa por: requisito → projeto → implementação
→ verificação de unidade e integração → verificação de sistema → liberação. Não há liberação
para uso clínico em nenhum portão anterior a G5.

## 2. Entregáveis por atividade (IEC 62304 §5)

| Atividade 62304 | Entregável neste repositório |
|---|---|
| §5.1 Planejamento | este documento; [`ROADMAP.md`](../../ROADMAP.md) |
| §5.2 Requisitos | [`06-requirements/REQ-*.md`](06-requirements/) com frontmatter (`id`, `title`, `type`, `risk_controls`, `status`, `verified_by`, `depends_on`) e seções "Requisito" (DEVE/NÃO DEVE) e "Justificativa" |
| §5.3 Arquitetura | [`07-architecture.md`](07-architecture.md); segregação de SOUP por extras opcionais do `pyproject.toml` |
| §5.4 Projeto detalhado | docstrings em português por unidade; tipos estritos (`mypy --strict`); contratos pydantic congelados |
| §5.5 Implementação e verificação de unidade | `tests/unit/<área>/test_*.py`; cobertura mínima de 85% imposta no CI |
| §5.6 Integração | `tests/integration/` |
| §5.7 Verificação de sistema | [`09-verification-plan.md`](09-verification-plan.md); matriz [`13-traceability.md`](13-traceability.md) gerada |
| §5.8 Liberação | etiqueta git + artefatos de build + relatório de verificação do commit (pendente: ainda não há release) |
| §6 Manutenção | [`14-change-control.md`](14-change-control.md) |
| §7 Gestão de risco | [`03-risk-management-plan.md`](03-risk-management-plan.md), [`04-risk-file.md`](04-risk-file.md) |
| §8 Gestão de configuração | git; `uv.lock`; lista SOUP gerada ([`08-soup-list.md`](08-soup-list.md)); sha256 de pesos nos model cards |
| §9 Resolução de problemas | issues do repositório, descritas como processo em [`14-change-control.md`](14-change-control.md) |

## 3. Padrões de codificação

- Python ≥ 3.11, < 3.14; `src`-layout, pacote único `radiologyai`.
- `from __future__ import annotations`; `mypy --strict`; ruff (linha de 100).
- Identificadores em inglês; docstrings, comentários, mensagens de erro e documentos em português
  do Brasil.
- Importações pesadas (numpy, torch, pydicom, SimpleITK, scikit-learn, fastapi) só dentro de
  funções ou sob `TYPE_CHECKING`; `import radiologyai` não importa torch, numpy, SimpleITK nem
  cv2 (verificado no CI).
- Dependência opcional ausente: `importlib.util.find_spec` + importação tardia + erro que nomeia o
  extra. `except ImportError` é proibido em `src/` (REQ-001, `scripts/check_honesty.py`).
- Tipos de resultado pydantic congelados com `extra="forbid"`.
- Exceções derivam de `radiologyai.errors.RadiologyAIError`; nada de `except` nu.
- **Regra de ouro:** nenhum caminho de código emite predição, métrica, mapa de calor ou alegação
  regulatória que um modelo ou algoritmo real não tenha calculado sobre a entrada real. Falha ⇒
  exceção explícita ou abstenção com motivo. Modelo que não roda de verdade fica `planned`, sem
  stub.
- `np.random`/`random` só na estatística de avaliação (REQ-002).

## 4. Portões de integração contínua

Definidos em `.github/workflows/ci.yml` e `security.yml`. Toda PR e todo push em `main` ou
`claude/**`:

| Portão | Comando |
|---|---|
| Lint e formatação | `ruff check src/ tests/ scripts/`; `ruff format --check src/ tests/` |
| Tipos | `mypy src/radiologyai` (modo estrito pela configuração) |
| Testes e cobertura | `pytest --cov=radiologyai --cov-fail-under=85` em Python 3.11, 3.12 e 3.13 |
| Honestidade | `python scripts/check_honesty.py` |
| SOUP | `python scripts/soup.py --check` |
| Rastreabilidade | `python scripts/trace.py --check` |
| Importação leve | asserção de que `import radiologyai` não carrega torch/numpy/SimpleITK/cv2 |
| Sem pixel no git | `git ls-files` sem `.dcm`, `.nii`, `.mhd`, `.raw`, `.nrrd` |
| Pesos reais | job `ml`: testes do backend TorchXRayVision e verificação do sha256 do card |
| Segurança | gitleaks no histórico completo, bandit, pip-audit, SBOM CycloneDX |

Testes que exigem pesos ou dados baixados pulam de forma limpa no CI sem eles
(`pytest.mark.skipif` por `find_spec` ou existência de arquivo) e **são executados** no ambiente
onde os pesos existem; o relatório de verificação registra onde rodaram.

## 5. Política de branches e revisão

- `main` é o ramo de integração; mudanças entram por PR com CI verde.
- Mensagens de commit em português com prefixo convencional (`feat`, `fix`, `docs`, `chore`…).
- Nunca commitar pesos, datasets, bancos de dados ou segredos.
- **Pendências de governança (G0):** proteção de branch em `main`; poda dos ramos remotos
  abandonados; expurgo de `medai_security.db` do histórico com `git filter-repo` e rotação do
  que ele possa ter tocado.

## 6. Ambientes e ferramentas

| Item | Versão / fonte |
|---|---|
| Gerenciador | `uv` com `uv.lock` (artefato de configuração e fonte da lista SOUP) |
| Python | 3.11 primário; 3.12 e 3.13 testados |
| PyTorch | conforme extra `ml` do `pyproject.toml` |
| CI | GitHub Actions |
| Execução de inferência pesada em desenvolvimento | CPU, sem GPU; trabalhos pesados serializados |

## 7. Desenvolvimento assistido por agentes

Parte da engenharia é feita por agentes de IA trabalhando em paralelo. Regras: cada agente só
altera os arquivos que lhe foram atribuídos; mudanças de contrato compartilhado são pedidas ao
responsável, não contornadas; todo código passa pelos mesmos portões de CI; nenhum agente edita a
matriz de rastreabilidade gerada. A responsabilidade pelo conteúdo é do autor humano do projeto.

## 8. Documentação de unidade (Classe C)

Cada módulo de `src/radiologyai/` tem docstring de módulo com propósito, entradas, saídas e o
requisito ou perigo que implementa; funções públicas documentam exceções levantadas. A
verificação de unidade tem critério de aceitação declarado no próprio teste (asserção) e, quando
controla risco, o marcador `@pytest.mark.requirement`.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação, descrevendo o processo já em execução |
