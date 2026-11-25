---
doc_id: REG-11
title: Cibersegurança e Proteção de Dados (LGPD / DPIA)
status: RASCUNHO — v0.2
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-04, REG-07]
---

# Cibersegurança e Proteção de Dados

> Software de pesquisa; não é dispositivo médico. Base normativa: RDC 848/2024 Art. 53 (nível
> adequado de cibersegurança para dispositivos com software), IEC 81001-5-1:2021 (segurança no
> ciclo de vida), LGPD (Lei 13.709/2018), DICOM PS3.15 Annex E. Nada aqui é parecer jurídico; os
> pontos marcados "a confirmar" precisam de leitura por profissional habilitado.

## 1. Ativos

| Ativo | Por que importa |
|---|---|
| Imagens e metadados de exames | dado pessoal sensível quando identificável ou pseudonimizado (LGPD Art. 5 II; Art. 13 §4) |
| Chave de pseudonimização (HMAC) | permite reidentificar; quem a tem transforma dado pseudonimizado em identificado |
| Resultados, laudos preliminares, assinaturas | integridade é controle de risco clínico (RISK-018) |
| Trilha de auditoria | prova de quem fez o quê; precisa ser à prova de adulteração (REQ-070) |
| Pesos de modelos | integridade (RISK-004) e licença (RISK-014) |
| Segredos de configuração | credenciais de banco, chaves de sessão |
| Registros de evidência e `levels.json` | adulterar um registro altera o nível exibido |

## 2. Modelo de ameaças (STRIDE resumido)

| Ameaça | Exemplo | Controle | Estado |
|---|---|---|---|
| Falsificação de identidade | uso de credencial de outro médico para assinar | autenticação por usuário (argon2id, sessão opaca revogável — REQ-150); RBAC por rota (REQ-151); assinatura vinculada a usuário e hash (REQ-156) | código presente; verificação em andamento |
| Adulteração | alterar resultado ou registro de evidência | auditoria encadeada por hash (REQ-070; no banco, REQ-155); `levels.json` gerado e reproduzível (REQ-116); sha256 de pesos; artefatos endereçados por conteúdo (REQ-107) | implementado no núcleo; plataforma em verificação |
| Repúdio | médico nega ter assinado | trilha com usuário, hash do conteúdo e carimbo de tempo | parcial |
| Divulgação de informação | PHI em log, em erro, em texto gravado no pixel, em backup | des-identificação na ingestão; logs sem identificador nem nome de arquivo; limpeza de pixel em US/eco (pendente) | parcial |
| Negação de serviço | upload gigante, volume malformado | limites medidos no fluxo da ingestão (REQ-152); decodificação que recusa em vez de adivinhar (`UnsupportedInputError`) | parcial |
| Elevação de privilégio | pesquisador acessa função de administrador | RBAC por papel imposto rota a rota; testes de autorização (REQ-151) | código presente; verificação em andamento |
| Cadeia de suprimentos | dependência com vulnerabilidade; pesos trocados no servidor de origem | `uv.lock`; lista SOUP gerada; pip-audit; SBOM CycloneDX; sha256 fixado por manifesto; sem rede na inferência | implementado (pip-audit ainda não bloqueia o build — ver §5) |

## 3. Des-identificação (DICOM PS3.15 Annex E)

Implementação atual (`io/deident.py`): subconjunto do perfil básico com pseudo-ID determinístico
por HMAC-SHA256 (REQ-010) e cobertura de tags verificada (REQ-011). O perfil completo e as opções
são os da edição corrente do PS3.15.

Configuração recomendada para a fábrica de evidência e para dados de parceiros:

| Opção PS3.15 | Uso |
|---|---|
| Basic Application Level Confidentiality Profile | sempre |
| Clean Pixel Data | **obrigatória** em ultrassom, ecocardiografia, captura secundária e radiografia digitalizada — o próprio padrão observa que US rotineiramente tem anotação gravada no pixel |
| Clean Descriptors | sempre |
| Retain Patient Characteristics | idade e sexo, para análise de subgrupo |
| Retain Device Identity | fabricante e modelo, para análise de mudança de equipamento |
| Retain Longitudinal Temporal Information with Modified Dates | quando o seguimento importa |
| Clean Recognizable Visual Features | considerar para TC/RM de crânio antes de qualquer compartilhamento externo (reconstrução facial) |

O perfil e as opções usados são registrados no datasheet de cada dataset.

## 4. LGPD — esboço de DPIA (Relatório de Impacto à Proteção de Dados)

| Item | Conteúdo |
|---|---|
| Finalidade | pesquisa em avaliação de modelos de IA sobre imagens médicas |
| Natureza dos dados | dado de saúde (sensível, Art. 5 II); imagem e metadados |
| Base legal | para dado hospitalar: Art. 11 II c (estudos por **órgão de pesquisa**), que exige pessoa jurídica de pesquisa (Art. 5 XVIII) — portanto o tratamento ocorre **sob a instituição parceira**, não sob a pessoa física do autor; para datasets públicos: os termos de cada dataset |
| Controlador | a instituição parceira (hospital/universidade) para dados hospitalares — a confirmar contratualmente |
| Minimização | des-identificação na origem pelo controlador; só os campos necessários (idade, sexo, equipamento) |
| Anonimização vs. pseudonimização | dado com chave de reidentificação é pseudonimizado e continua dado pessoal (Art. 12, Art. 13 §4); a chave fica separada, com o controlador |
| Transferência | nenhuma transferência a terceiro (Art. 13 §2 para estudos de saúde pública); nenhum envio de imagem a API ou nuvem de terceiro; modelos rodam localmente |
| Retenção | pelo prazo do protocolo CEP; registros de validação de SaMD *in house* teriam de ser guardados por 10 anos (RDC 657/2022 Art. 5 §3) — a confirmar se aplicável |
| Direitos do titular | exercidos perante o controlador |
| Incidentes | procedimento de comunicação a definir com o encarregado do controlador |
| Ética | protocolo CEP; Res. CNS 738/2024 para bancos de dados |
| Riscos residuais | reidentificação por características visuais (crânio), por texto no pixel (US), por combinação de metadados retidos |

## 5. Pendências conhecidas

1. **`medai_security.db` no histórico git.** O arquivo SQLite do legado foi removido da árvore,
   mas continua no histórico (commits `3f5ae26`, `c80c22e`, `0b89c95`). Ação: expurgo com
   `git filter-repo`, *force-push* coordenado, invalidação de forks/caches conhecidos e rotação de
   qualquer credencial que ele possa ter guardado. Até lá, tratar o conteúdo como exposto.
   O mesmo vale para segredos do legado (`medai_config.json`, senhas padrão do
   `docker-compose.yml`) presentes no histórico.
2. `pip-audit` roda com `|| true` no workflow de segurança: reporta mas não bloqueia. Decidir
   política (bloquear vulnerabilidade com correção disponível).
3. Limpeza de texto gravado no pixel para US/eco ainda não implementada.
4. RBAC, autenticação e sessão da plataforma: escritos (REQ-150, REQ-151); falta política de expiração/rotação de sessão documentada e revisão de segurança independente.
5. SBOM é gerado no CI; falta anexá-lo a cada liberação.
6. Plano de tratamento de vulnerabilidades (recebimento, triagem, prazo, divulgação) a escrever,
   conforme IEC 81001-5-1.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação |
| v0.2 | 2026-11-25 | Estado dos controles da plataforma ligado aos requisitos REQ-150…REQ-155 |
