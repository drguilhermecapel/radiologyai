---
doc_id: REG-04
title: Arquivo de Risco / Análise de Perigos
status: RASCUNHO VIVO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-03]
---

# Arquivo de Risco

> Documento vivo. Critérios de severidade (S1–S5), probabilidade (P1–P5) e aceitabilidade em
> [`03-risk-management-plan.md`](03-risk-management-plan.md). A probabilidade é qualitativa até
> existirem dados de G2/G4; por isso **nenhum risco residual está declarado aceitável** nesta
> versão — a coluna diz o que falta para avaliá-lo.

`RISK-001`…`RISK-010` correspondem um a um aos perigos-semente `H-01`…`H-10` do
[`ROADMAP.md`](../../ROADMAP.md) §8.3 e já são citados pelos requisitos em
[`06-requirements/`](06-requirements/). `RISK-011`…`RISK-020` vêm da multimodalidade, do
zero-shot, da medida assistida, dos pesos não comerciais e da plataforma web.

## 1. Resumo

| ID | Perigo | S | P (antes) | Controles (requisitos) | Situação |
|---|---|---|---|---|---|
| RISK-001 | Falso negativo em achado crítico | S5 | P3 | abstenção (REQ-062); nível D fora da visão principal (REQ-091); decisão só com ponto de operação de origem declarada (REQ-092); IFU proíbe *rule-out* | controles implementados; residual a estimar em G2 |
| RISK-002 | Viés de automação | S5 | P3 | nunca afirmar normalidade (REQ-063); incerteza e nível visíveis; revisão obrigatória | parcial — interface ainda em construção |
| RISK-003 | Entrada fora de distribuição processada em silêncio | S4 | P4 | gate de escopo antes do modelo (REQ-031, REQ-042); falha alta (REQ-001) | implementado para RX; demais modalidades em construção |
| RISK-004 | Descasamento entre pesos e código | S4 | P2 | sha256 verificado (REQ-020); cabeças não treinadas não reportadas (REQ-021); falha fechada (REQ-030); API não fabrica métrica (REQ-080) | implementado |
| RISK-005 | Má interpretação de DICOM | S4 | P3 | Modality LUT (REQ-005); MONOCHROME1 (REQ-006) | implementado para RX; VOI LUT em mamografia e HU em TC a verificar por modalidade |
| RISK-006 | Erro de lateralidade | S4 | P2 | lateralidade propagada do DICOM, nunca inferida do pixel | **sem requisito ainda** — lacuna |
| RISK-007 | Desempenho real diverge do declarado (número sem lastro, deriva) | S5 | P3 | sem aleatoriedade (REQ-002); toda verificação executada (REQ-003); avaliação reprodutível (REQ-050); sem vazamento por paciente (REQ-051); auditoria (REQ-070, REQ-071); nível calculado (REQ-090); medição própria prevalece (REQ-093) | implementado; deriva pós-implantação só em G6 |
| RISK-008 | Explicabilidade enganosa | S4 | P2 | mapa derivado do modelo real (REQ-060) | implementado |
| RISK-009 | Vazamento de dado pessoal | S3 | P3 | pseudo-ID determinístico (REQ-010); cobertura PS3.15 (REQ-011) | parcial — pixel com texto gravado (US, eco, captura secundária) pendente |
| RISK-010 | Confiança mal calibrada | S4 | P4 | calibração preserva ordenação (REQ-061); IFU: escore não é probabilidade de doença | implementado; ECE reportado por avaliação |
| RISK-011 | Biomarcador quantitativo fora das pré-condições de aquisição | S3 | P4 | verificar fase, kVp, kernel, espessura, inspiração; senão abstenção com motivo | em implementação — requisito a criar |
| RISK-012 | Saída exploratória (zero-shot, nível D) lida como validada | S4 | P3 | modo pesquisa separado e desligado por padrão; nível D nunca sozinho no laudo (REQ-091) | parcial |
| RISK-013 | Medida assistida aceita sem revisão (contorno, FE) | S4 | P3 | contorno exibido e confirmado pelo médico; FE só em modo pesquisa; caixa/prompt sempre do usuário ou declarado | em implementação — requisito a criar |
| RISK-014 | Uso de pesos fora da licença (não comercial, só acadêmico, sem licença) | S2 | P3 | `commercial_use` no manifesto; modelos sem licença excluídos; pesos de licença intransferível não redistribuídos | parcial — inventário em [`MODEL_LICENSES.md`](../../MODEL_LICENSES.md) |
| RISK-015 | Vazamento treino/teste na evidência ⇒ nível superestimado | S4 | P4 | campo `leakage` obrigatório; medição com vazamento no máximo C; política do [protocolo](../protocolo-evidencia.md) §8 | implementado no esquema; verificação por dataset em andamento |
| RISK-016 | Modelo aplicado fora da população ou tarefa de treino | S4 | P3 | pré-condições por condição (ex.: glioma conhecido; T1w + FLAIR; diagnóstico, não rastreamento) | em implementação |
| RISK-017 | Roteamento errado de exame (região, sequência ou janela ecocardiográfica) | S3 | P3 | roteamento por metadado DICOM validado; desconhecido ⇒ abstenção | em implementação |
| RISK-018 | Assinatura do médico desvinculada do conteúdo analisado | S4 | P2 | assinatura vinculada ao hash do conteúdo; mudança após assinatura invalida a assinatura; auditoria encadeada | planejado na plataforma |
| RISK-019 | Evidência publicada transcrita errado ou não aplicável ao modelo | S4 | P3 | citação literal conferida na fonte (`verified_at_source`); mesmo modelo e versão; revisão por pares ou diretriz | implementado no esquema |
| RISK-020 | Acesso não autorizado ou adulteração | S4 | P3 | controle de acesso por papel; auditoria encadeada; segredos só por ambiente; expurgo de `medai_security.db` do histórico | parcial — ver [REG-11](11-cybersecurity-lgpd.md) |

## 2. Detalhamento

Formato: perigo → sequência de eventos → situação perigosa → dano.

### RISK-001 — Falso negativo em achado crítico (H-01)
Modelo não sinaliza pneumotórax, hemorragia intracraniana, derrame pericárdico ou câncer → leitor
não vê um achado sugerido e interpreta a ausência como tranquilização → tratamento tardio →
morte ou deterioração irreversível. Controles: banda "não avaliável"; nenhuma condição sem ponto
de operação vira "negativo"; achado de nível D não aparece como conclusão; a IFU proíbe uso para
exclusão. Verificação: REQ-062, REQ-091, REQ-092.

### RISK-002 — Viés de automação (H-02)
Saída negativa confiante ancora o leitor, que passa a perder achados que veria sem assistência
(evidência de que saída errada da IA piora leitores, inclusive experientes, compilada na pesquisa
verificada de novembro/2026). Controles: nunca afirmar normalidade (REQ-063); mostrar nível,
fonte e incerteza; sem rótulo *top-1* isolado; revisão e assinatura obrigatórias por achado.
Residual: só um estudo MRMC (G3) mede o efeito.

### RISK-003 — Entrada fora de distribuição (H-03)
Incidência lateral, criança, AP de leito, *scout* de TC, sequência de RM errada, imagem que não é
médica → modelo produz escore sem significado → achado espúrio ou falso negativo. Controles: gate
de escopo **antes** do modelo (REQ-031), limites de RX (REQ-042), gate de distribuição por
autoencoder no RX, dependência ausente falha alto (REQ-001).

### RISK-004 — Descasamento pesos/código (H-04)
Pesos de outra versão ou corrompidos carregados em silêncio; cabeça de saída não treinada
reportada como achado. Controles: sha256 verificado na carga (REQ-020); cabeças não treinadas
excluídas (REQ-021); falha fechada (REQ-030); a API não fabrica métrica (REQ-080).

### RISK-005 — Má interpretação de DICOM (H-05)
MONOCHROME1 não invertido, RescaleSlope/Intercept ignorado, VOI LUT ausente na mamografia →
imagem com contraste invertido ou HU errado → saída errada. Controles: REQ-005, REQ-006; testes
com DICOMs sintéticos.

### RISK-006 — Lateralidade (H-06)
Achado reportado no lado errado → procedimento no lado errado. Controle previsto: lateralidade
(0020,0060)/(0020,0062) propagada do DICOM e asserida; nunca inferida do pixel. **Lacuna: não há
requisito nem teste ainda.** Prioridade antes de qualquer laudo com lado (mamografia, fratura).

### RISK-007 — Desempenho real diverge do declarado (H-07)
Número sem lastro (como no v1), avaliação irreprodutível, vazamento por paciente, ou deriva após
implantação → confiança indevida → dano por erro não esperado. Controles: REQ-002, REQ-003,
REQ-050, REQ-051, REQ-070, REQ-071, REQ-090, REQ-093; `scripts/check_honesty.py` no CI.
Deriva: [REG-15](15-post-market-surveillance.md).

### RISK-008 — Explicabilidade enganosa (H-08)
Mapa de saliência que não reflete o modelo (o v1 servia bordas Canny) → clínico atribui
significado a artefato. Controle: Grad-CAM por *hooks* de gradiente; teste de modelo com
gradiente nulo ⇒ mapa nulo (REQ-060). Sobreposição de máscara só a partir de segmentação real.

### RISK-009 — Vazamento de dado pessoal (H-09)
Des-identificação incompleta, nome de arquivo ou texto livre DICOM em log, texto gravado no pixel
(rotina em ultrassom) → exposição de dado sensível (LGPD Art. 5 II). Controles: REQ-010, REQ-011;
redação de log. Lacuna: limpeza de pixel (opção *Clean Pixel Data* do PS3.15) para US/eco.

### RISK-010 — Confiança mal calibrada (H-10)
Escore 0,85 lido como 85% de probabilidade de doença. Controles: calibração por temperatura que
preserva a ordenação (REQ-061); ECE em toda avaliação; IFU declara que escore não é probabilidade.

### RISK-011 — Biomarcador fora das pré-condições
Atenuação hepática medida em fase portal com limiar de TC sem contraste; %LAA-950 em exame
contrastado ou expiratório; diâmetro aórtico só axial → valor fora da faixa de validade do
limiar → falso achado de esteatose, enfisema ou aneurisma. Controle: cada biomarcador declara
pré-condições e abstém com motivo quando não verificáveis (fase de contraste desconhecida ⇒
abstenção, porque etiquetas DICOM sozinhas não são confiáveis para a fase).

### RISK-012 — Exploratório lido como validado
Saída zero-shot (CheXzero, BiomedCLIP, Merlin) ou condição de nível D vista fora do contexto de
pesquisa → tratada como achado validado. Controles: modo pesquisa desligado por padrão, marcado
visualmente; REQ-091; nível sempre exibido ao lado do achado.

### RISK-013 — Medida assistida aceita sem revisão
Contorno do VE propagado automaticamente com viés sistemático, ou caixa gerada sem o usuário →
volumes e FE errados → conduta errada. Controles: o contorno é sempre exibido e confirmado ou
corrigido pelo médico; FE só em modo pesquisa até passar na barra de concordância do protocolo;
a origem do prompt (usuário vs. automático) é registrada.

### RISK-014 — Licença violada
Pesos só acadêmicos ou não comerciais usados ou redistribuídos fora dos termos → retirada
obrigatória do modelo no meio de um estudo, invalidando o estudo. Severidade baixa para o
paciente, alta para o projeto. Controles: campo `commercial_use` por manifesto; EchoPrime não
integrado (licença intransferível); PanEcho excluído (sem licença).

### RISK-015 — Vazamento na evidência
Avaliar pesos no dado em que foram treinados (ex.: TorchXRayVision `all`/`nih` no teste oficial
do NIH; mammoscreen no CBIS-DDSM; desafios SIIM/RSNA derivados do NIH) → nível A indevido.
Controles: campo `leakage` obrigatório; medição com vazamento no máximo C (`compute_level`);
política do protocolo §8.

### RISK-016 — Fora da população ou tarefa de treino
Bundle BraTS usado em cérebro sem glioma conhecido; LST-AI lido como diagnóstico de esclerose
múltipla; mammoscreen tratado como rastreamento; Mirai (risco futuro) lido como detecção.
Controles: pré-condições por condição no catálogo; redação do achado restrita ao que o modelo
mede ("carga lesional", não "esclerose múltipla").

### RISK-017 — Roteamento errado
TC de crânio enviada a modelo de abdome, janela ecocardiográfica errada → saída sem significado.
Controles: roteamento por metadado validado; exame desconhecido ⇒ abstenção.

### RISK-018 — Assinatura desvinculada
Médico assina uma versão e o laudo exportado é outra → decisão sobre conteúdo não revisado.
Controle: assinatura sobre o hash do conteúdo; qualquer mudança posterior invalida a assinatura
e é registrada na auditoria.

### RISK-019 — Evidência publicada errada ou não aplicável
Número de artigo transcrito errado, de outro modelo ou versão, ou de preprint → nível B indevido.
Controles: `quote` literal com `verified_at_source`; barra de publicação do protocolo §6.

### RISK-020 — Acesso não autorizado ou adulteração
Credencial vazada, segredo no histórico git (`medai_security.db` ainda presente no histórico),
alteração de resultado → exposição de dados ou resultado adulterado. Controles: RBAC,
auditoria encadeada por hash, segredos só por variável de ambiente, expurgo do histórico
([REG-11](11-cybersecurity-lgpd.md)).

## 3. Lacunas abertas

1. RISK-006 sem requisito nem teste.
2. RISK-011, RISK-013, RISK-016, RISK-017, RISK-018 sem requisito próprio ainda — precisam de
   `REQ-NNN` antes do fechamento de G1 para as modalidades afetadas.
3. RISK-009: limpeza de texto gravado no pixel para US/eco.
4. RISK-020: expurgo do histórico git pendente.
5. Nenhum risco residual avaliado — depende de frequências de G2/G4.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação: RISK-001…RISK-010 a partir de H-01…H-10; RISK-011…RISK-020 da multimodalidade |
