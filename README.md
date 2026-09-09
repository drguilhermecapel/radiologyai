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

O núcleo v2 (`src/radiologyai/`) está em construção. O código v1 foi arquivado
em `legacy/` — está preservado, mas nada o importa e ele é excluído de build,
lint, tipos e CI.

| Camada | Estado |
|---|---|
| Leitura DICOM — VOI LUT, MONOCHROME1, RescaleSlope/Intercept, sem quantização | **funcional, testada** |
| Des-identificação PS3.15 (subconjunto) com pseudo-IDs determinísticos HMAC | **funcional, testada** |
| Janelamento por modalidade (presets de TC em HU) | **funcional, testada** |
| Plugin de modalidade + gate de escopo (o controle de risco H-03) | **funcional, testada** |
| Modalidades TC / RM / US | **declaradas**, não implementadas — levantam `NotImplementedError` |
| Model card com sha256 validado e verificação de integridade | **funcional, testada** |
| Motor de inferência falha-fechada | **funcional** — sem backend de pesos ainda |
| Métricas (AUROC com IC bootstrap, ponto de operação, ECE, vazamento) | **funcional, testada** |
| CLI (`radiologyai version / selftest / modalities / inspect / cards`) | **funcional** |
| Backend de pesos reais, calibração, abstenção, Grad-CAM, API | Fase 2 |
| **Modelos treinados** | **nenhum** |
| **Métricas de desempenho** | **nenhuma medida** |
| **Validação clínica** | **nenhuma** |

## Instalação

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev,eval]"
radiologyai selftest
```

## Verificação

```bash
ruff check src/ tests/ scripts/     # lint
mypy src/radiologyai               # tipos, modo strict
pytest --cov=radiologyai           # 154 testes
python scripts/check_honesty.py    # guardião de honestidade
python scripts/trace.py --check    # rastreabilidade RISCO→REQ→TESTE
python scripts/soup.py --check     # lista SOUP (IEC 62304 §8.1.2)
```

Três portões de CI existem especificamente para impedir a recorrência do modo de
falha do v1:

- **`check_honesty.py`** — quebra o build em qualquer alegação numérica de
  desempenho sem artefato de avaliação que a sustente, qualquer sha256 inválido,
  qualquer `except ImportError` que degrade em silêncio, e qualquer marcador de
  simulação em código de produção.
- **`trace.py`** — quebra o build se um requisito ficar sem teste verificador ou
  se um teste citar um requisito inexistente.
- **`soup.py`** — quebra o build se a lista SOUP divergir das dependências.

O problema do v1 nunca foi falta de competência técnica. Foi que nada no sistema
jamais objetou quando um número foi inventado. A objeção agora é automática.

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
