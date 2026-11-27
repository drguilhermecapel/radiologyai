# Fase 1 — Benchmark em Coorte Inédita do NIH ChestX-ray14 (Research-Only)

> **Aviso regulatório:** Este documento registra exclusivamente resultados de pesquisa computacional (*research-only*). O sistema **não é um dispositivo médico** e não se destina a uso clínico, diagnóstico ou tomada de decisão médica.

> **Situação da proveniência (conferida em 2026-11-27): NÃO VERIFICÁVEL.** Os números abaixo foram
> transcritos de uma execução em Google Colab cujo artefato (`metrics.json`, predições, manifest
> da amostra) **não está commitado** em `artifacts/eval/` nem em `evidence/`. Nenhum deles pode
> ser reproduzido ou conferido a partir deste repositório, e nenhum sustenta nível de evidência,
> critério de aceitação ou alegação em outro documento. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md)
> §"Relatos sem artefato". Os únicos números de desempenho com artefato são os de
> `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json` e
> `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`.

> **Observações da conferência:** (1) o próprio relato declara que o ZIP com os artefatos locais
> do Colab não foi recuperado (§5, limitação 3); (2) `densenet121-res224-all` inclui o NIH
> ChestX-ray14 no treino, portanto a comparação `All − PC` não é externa para `All` — o portão de
> regressão bloqueou corretamente (§4); com o split aleatório por paciente do TorchXRayVision,
> pacientes do NIH podem estar na parte de treino dos pesos `all` ([`ROADMAP.md`](../../ROADMAP.md)
> §9.4); (3) a consistência aritmética interna foi conferida (diferença pareada e redução
> relativa de ECE batem com os valores relatados), o que não substitui o artefato.

## 1. Contexto e Desenho Experimental

- **Data da execução:** 25/11/2026
- **Ambiente de hardware:** GPU Tesla T4 (Google Colab)
- **Coorte inédita (NIH ChestX-ray14):** 1.000 novos pacientes distintos (1 radiografia por paciente), com exclusão explícita dos 1.044 pacientes históricos (64 do piloto inicial + 1.000 do conjunto NIH-1000, totalizando 0 sobreposição de pacientes).
- **Particionamento disjunto:** 250 pacientes para calibração e 750 pacientes para teste (0 pacientes compartilhados entre calibração e teste).
- **Inferência estatística:** 1.000 reamostragens *bootstrap* pareadas por paciente.

## 2. Discriminação (6 Achados Elegíveis)

Dos 14 achados comuns avaliados no conjunto de teste (750 pacientes), **6 achados** atenderam ao critério de elegibilidade (`>= 10` positivos e `>= 10` negativos no teste):

| Modelo | Macro-AUROC (6 achados) — NÃO VERIFICÁVEL | IC 95% (Bootstrap Pareado) |
| --- | --- | --- |
| `xrv-densenet121-pc` | 0,7938 (NÃO VERIFICÁVEL) | 0,7508 – 0,8331 |
| `densenet121-res224-all` | 0,8306 (NÃO VERIFICÁVEL) | 0,7970 – 0,8615 |

- **Diferença pareada (`All - PC`):** `+0,0368` (IC 95%: `+0,0041` a `+0,0713`) — NÃO VERIFICÁVEL.

## 3. Transferibilidade e Calibração (Platt Scaling)

Avaliação do *Expected Calibration Error* (ECE) no conjunto de teste (`n = 750` pacientes):

- **Transferibilidade sem reajuste (`xrv-densenet121-pc` com Platt anterior congelado):**
  - ECE sem calibração: `0,22526` — NÃO VERIFICÁVEL
  - ECE com Platt congelado: `0,06013` (redução de ~73%; IC 95% de delta: `-0,17270` a `-0,15549`) — NÃO VERIFICÁVEL
  - Observação: o valor residual (`0,06013`) permaneceu acima do alvo de `0,05` — NÃO VERIFICÁVEL.
- **Recalibração na coorte inédita (calibradores ajustados nos 250 pacientes de calibração e avaliados nos 750 de teste):**
  - `xrv-densenet121-pc`: ECE reduzido para `0,00797` — NÃO VERIFICÁVEL
  - `densenet121-res224-all`: ECE reduzido para `0,00755` — NÃO VERIFICÁVEL

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
- Commitar o artefato desta execução (ou refazê-la com `radiologyai evaluate` / `.github/workflows/eval.yml`) antes de citar qualquer número dela.
- Executar **validação externa verdadeira** no conjunto **VinDr-CXR 1.0.0** (PhysioNet, ausente das fontes de pré-treinamento do `densenet121-res224-all`), após conclusão do credenciamento CITI/PhysioNet.
