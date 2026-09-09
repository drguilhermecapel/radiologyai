# RadiologyAI (MedAI Radiologia)

Plataforma de pesquisa em interpretação de imagens radiológicas por inteligência artificial.

> ## ⚠️ Software de pesquisa — não é dispositivo médico
>
> **Não existe nenhum modelo treinado neste repositório. Nenhuma métrica de desempenho foi medida. Não houve validação clínica. Não use para nenhuma decisão sobre um paciente real.**
>
> Versões anteriores deste README alegavam acurácia de 88–95% e "validação clínica". **Essas alegações eram fabricadas e estão retratadas.** Ver **[HONEST_STATUS.md](HONEST_STATUS.md)** para a retratação completa e a lista de defeitos conhecidos do código legado.
>
> O plano para transformar isto em algo real está em **[ROADMAP.md](ROADMAP.md)**.

---

## O que existe hoje

| Camada | Estado |
|---|---|
| Leitura DICOM (pydicom + SimpleITK), windowing por modalidade | **funcional**, com defeitos conhecidos ([HONEST_STATUS.md](HONEST_STATUS.md) §6, §10) |
| Normalização por modalidade (HU para TC, etc.) | **funcional** |
| Contrato de API REST (FastAPI, `/api/v1/*`) | **esqueleto**; a implementação atual fabrica métricas por requisição |
| Modelos treinados | **nenhum** |
| Métricas de desempenho | **nenhuma medida** |
| Validação clínica | **nenhuma** |
| CI, testes automatizados, empacotamento | **nenhum** |
| Documentação regulatória (IEC 62304, ISO 14971) | **em construção** — `docs/regulatory/` |

O restante do código em `src/` (67 módulos) está em processo de arquivamento. Ver a tabela de disposição em [`ROADMAP.md` §1.14](ROADMAP.md).

## Objetivo

Plataforma multimodal (RX, TC, RM, US) extensível por plugin de modalidade, com médico no circuito, trilha de auditoria, explicabilidade real, calibração, abstenção por baixa confiança e documentação compatível com SaMD, em trajetória de registro ANVISA.

**Nenhuma alegação de desempenho será publicada neste repositório sem um artefato de avaliação reproduzível em `artifacts/eval/` que a sustente.** Um verificador de CI (`scripts/check_honesty.py`) impõe essa regra automaticamente.

## Estado dos dados

Os dados de exemplo previamente incluídos em `data/nih_chest_xray/` eram **imagens sintéticas desenhadas com OpenCV**, gravadas com nomes de arquivos do NIH ChestX-ray14 (`00000001_000.png`). Foram removidos por risco de proveniência. Nenhum dado de paciente jamais esteve neste repositório.

## Referência técnica

- **[ROADMAP.md](ROADMAP.md)** — diagnóstico do estado atual, arquitetura-alvo, decisão de framework, fases, trilha regulatória, orçamento e cronograma
- **[HONEST_STATUS.md](HONEST_STATUS.md)** — retratação das alegações anteriores e defeitos conhecidos
- `docs/OPENAPI_SPEC.yaml` — contrato de API alvo
- `docs/CLINICAL_VALIDATION.md` — relatório da v1; é o único documento historicamente honesto do repositório (registra 20% de acurácia em 5 imagens sintéticas)

## Licença e contato

- Repositório: https://github.com/drguilhermecapel/radiologyai
- Autor: Dr. Guilherme Capel Pasqua — CRM-SP 175873
