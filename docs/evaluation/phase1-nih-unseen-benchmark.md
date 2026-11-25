# Fase 1 — Benchmark em Coorte Inédita do NIH ChestX-ray14 (Research-Only)

> **Aviso regulatório:** Este documento registra exclusivamente resultados de pesquisa computacional (*research-only*). O sistema **não é um dispositivo médico** e não se destina a uso clínico, diagnóstico ou tomada de decisão médica.

## 1. Contexto e Desenho Experimental

- **Data da execução:** 25/11/2026
- **Ambiente de hardware:** GPU Tesla T4 (Google Colab)
- **Coorte inédita (NIH ChestX-ray14):** 1.000 novos pacientes distintos (1 radiografia por paciente), com exclusão explícita dos 1.044 pacientes históricos (64 do piloto inicial + 1.000 do conjunto NIH-1000, totalizando 0 sobreposição de pacientes).
- **Particionamento disjunto:** 250 pacientes para calibração e 750 pacientes para teste (0 pacientes compartilhados entre calibração e teste).
- **Inferência estatística:** 1.000 reamostragens *bootstrap* pareadas por paciente.

## 2. Discriminação (6 Achados Elegíveis)

Dos 14 achados comuns avaliados no conjunto de teste (750 pacientes), **6 achados** atenderam ao critério de elegibilidade (`>= 10` positivos e `>= 10` negativos no teste):

| Modelo | Macro-AUROC (6 achados) | IC 95% (Bootstrap Pareado) |
| --- | --- | --- |
| `xrv-densenet121-pc` | 0,7938 | 0,7508 – 0,8331 |
| `densenet121-res224-all` | 0,8306 | 0,7970 – 0,8615 |

- **Diferença pareada (`All - PC`):** `+0,0368` (IC 95%: `+0,0041` a `+0,0713`).

## 3. Transferibilidade e Calibração (Platt Scaling)

Avaliação do *Expected Calibration Error* (ECE) no conjunto de teste (`n = 750` pacientes):

- **Transferibilidade sem reajuste (`xrv-densenet121-pc` com Platt anterior congelado):**
  - ECE sem calibração: `0,22526`
  - ECE com Platt congelado: `0,06013` (redução de ~73%; IC 95% de delta: `-0,17270` a `-0,15549`)
  - Observação: o valor residual (`0,06013`) permaneceu acima do alvo de `0,05`.
- **Recalibração na coorte inédita (calibradores ajustados nos 250 pacientes de calibração e avaliados nos 750 de teste):**
  - `xrv-densenet121-pc`: ECE reduzido para `0,00797`
  - `densenet121-res224-all`: ECE reduzido para `0,00755`

## 4. Portão de Regressão (*Fail-Closed*)

A execução do comando `radiologyai check-regression` retornou bloqueio *fail-closed*:

- **Status:** `passed: false`
- **Motivo:** `current_metrics.dataset.external_to_training_data must be True`
- **Interpretação:** como o modelo `densenet121-res224-all` inclui o conjunto NIH ChestX-ray14 entre suas fontes de pré-treinamento, o portão impede tratar o NIH como validação externa independente para o modelo `All`.

## 5. Limitações Explícitas e Próximo Passo Científico

### Limitações
1. **Cobertura parcial de achados:** apenas 6 de 14 achados comuns atingiram o mínimo de `>= 10` positivos e `>= 10` negativos no conjunto de teste.
2. **Amostragem determinística:** a seleção seguiu a ordem determinística da lista oficial após exclusão dos 1.044 pacientes históricos.
3. **Rastreabilidade de artefatos locais:** ausência de download recuperado do arquivo ZIP local gerado no ambiente Colab.

### Próximo Passo Científico
- Executar **validação externa verdadeira** no conjunto **VinDr-CXR 1.0.0** (PhysioNet, ausente das fontes de pré-treinamento do `densenet121-res224-all`), após conclusão do credenciamento CITI/PhysioNet.
