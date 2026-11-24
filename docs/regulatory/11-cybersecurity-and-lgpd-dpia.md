doc_id: REG-11
title: Cibersegurança e Relatório de Impacto à Proteção de Dados (LGPD / DPIA)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
# Cibersegurança e LGPD / DPIA

> **Estado atual:** software em estágio de pesquisa — não é dispositivo médico e não processa dados clínicos em produção assistencial. Este documento estabelece a arquitetura de cibersegurança (ISO/IEC 81001-5-1, IEC 62304, RDC 657/2022) e a avaliação preliminar de proteção de dados pessoais sensíveis de saúde sob a **LGPD (Lei nº 13.709/2018)**. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

## 1. Superfície de ataque e modelo de ameaças

| Ativo / Fronteira | Ameaça principal | Controle técnico implementado / planejado | Referência |
|---|---|---|---|
| **Pesos de modelo e artefatos** | Adulteração de pesos (*supply-chain tampering*) ou substituição silenciosa de modelo | Verificação obrigatória de SHA-256 na carga com falha fechada (`REQ-020`, `REQ-030`) | `src/radiologyai/models/` |
| **Dependências SOUP** | Vulnerabilidades conhecidas (CVEs) ou pacotes maliciosos | `uv.lock`, inventário verificado (`scripts/soup.py --check`), `pip-audit` semanal e SBOM CycloneDX em `.github/workflows/security.yml` | [`08-soup-list.md`](08-soup-list.md) |
| **Código-fonte e segredos** | Vazamento de credenciais em commits ou configuração | `gitleaks` no histórico completo e análise estática `bandit` em CI; segredos apenas por variáveis de ambiente (`.env.example`) | `.github/workflows/security.yml` |
| **API HTTP (`/api/v1/*`)** | Entrada malformada, abuso de payload ou exposição de dados em mensagens de erro | Validação estrita por `pydantic`, gate de escopo antes da inferência (`REQ-031`), container `non-root` (`Dockerfile`) | `src/radiologyai/api/` |
| **Trilha de auditoria** | Adulteração retroativa de registros de decisão | Log append-only encadeado por hash criptográfico com detecção de adulteração (`REQ-070`, `REQ-071`) | `src/radiologyai/audit/` |

## 2. Proteção de Dados Pessoais Sensíveis (LGPD — DPIA preliminar)

### 2.1 Natureza dos dados e minimização
- **Dados de saúde (Art. 5º, II e Art. 11 da LGPD):** imagens DICOM e metadados clínicos constituem dados pessoais sensíveis.
- **Princípio da minimização:** na etapa de pesquisa, o repositório proíbe o versionamento de qualquer dado de pixel médico (portão de CI bloqueando `.dcm`, `.nii`, `.nii.gz`, `.mhd`, `.raw`, `.nrrd`).

### 2.2 Des-identificação técnica (DICOM PS3.15)
- O módulo `src/radiologyai/io/` implementa remoção verificada de tags identificadoras conforme perfil **DICOM PS3.15** (`REQ-011`) e geração de pseudo-identificadores determinísticos (`REQ-010`) sem expor identificadores diretos de paciente em logs ou mensagens de erro (`RISK-009` / `H-09`).

### 2.3 Governança para operação futura (Fases 5–6)
Antes de qualquer implantação com dados reais de pacientes em ambiente hospitalar ou ambulatorial, serão obrigatórios:
1. Definição formal de Controlador, Operador e Encarregado (DPO) no veículo societário (Fase 6).
2. Base legal documentada (Art. 11 da LGPD) e política de retenção/descarte seguro compatível com a trilha de auditoria regulatória.
3. Criptografia em trânsito (TLS 1.3) e em repouso, controle de acesso baseado em papéis (RBAC) e testes de intrusão (*pentest*) pré-produção.

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação do documento de Cibersegurança e LGPD/DPIA preliminar (Fase 3) |
