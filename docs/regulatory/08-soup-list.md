---
doc_id: REG-08
title: Lista SOUP (Software of Unknown Provenance)
status: GERADO AUTOMATICAMENTE — não editar à mão
generated_by: scripts/soup.py
source_commit: ec4b79c
---

# Lista SOUP

IEC 62304 §5.3.3 e §8.1.2. **Documento gerado** — regenere com
`python scripts/soup.py --write` e verifique no CI com `--check`.

SOUP é todo software de terceiros incorporado ao produto cuja qualificação
não foi feita por nós. Para cada item a norma exige identificação, versão,
propósito no sistema e lista de anomalias publicadas.

**Relevante para a segurança** marca os itens cuja falha pode contribuir
para uma situação perigosa (ISO 14971). Esses exigem monitoramento ativo de
anomalias publicadas (CVE/OSV) e avaliação de impacto a cada atualização.

| Item | Especificação | Instalado | Extra | Segurança | Propósito no sistema |
|---|---|---|---|---|---|
| `fastapi` | `fastapi>=0.110,<1` | 0.141.1 | api | não | Camada de API REST |
| `python-multipart` | `python-multipart>=0.0.9` | 0.0.32 | api | não | Upload de arquivos na API |
| `uvicorn` | `uvicorn[standard]>=0.29` | 0.52.4 | api | não | Servidor ASGI |
| `numpy` | `numpy>=1.26,<2.2` | 2.1.3 | core | **sim** | Aritmética de arrays em todo o pipeline de imagem e métricas |
| `pydantic` | `pydantic>=2.7,<3` | 2.13.5 | core | **sim** | Validação de contratos de dados (model cards, metadados, resultados) |
| `pydantic-settings` | `pydantic-settings>=2.3,<3` | 2.15.0 | core | não | Carga de configuração a partir do ambiente |
| `pydicom` | `pydicom==2.4.4` | 2.4.4 | core | **sim** | Parsing de DICOM, acesso a tags e des-identificação |
| `pyyaml` | `PyYAML>=6.0,<7` | 6.0.3 | core | não | Leitura de model cards em YAML |
| `typer` | `typer>=0.12,<0.16` | 0.15.4 | core | não | Interface de linha de comando |
| `scikit-learn` | `scikit-learn>=1.4,<2` | 1.9.0 | eval | não | Métricas de referência e utilidades de avaliação |
| `scipy` | `scipy>=1.11,<2` | 1.17.1 | eval | não | Estatística usada nos intervalos de confiança |
| `pillow` | `Pillow>=10.0` | 12.3.0 | imaging | não | Decodificação de imagem não-DICOM |
| `simpleitk` | `SimpleITK>=2.3,<3` | — | imaging | **sim** | Leitura de séries e reamostragem volumétrica (TC/RM) |
| `monai` | `monai>=1.3,<2` | — | ml | **sim** | Transformações e arquiteturas de imagem médica (2D e 3D) |
| `onnxruntime` | `onnxruntime>=1.18` | — | ml | **sim** | Runtime de inferência para implantação em CPU |
| `timm` | `timm>=1.0` | — | ml | **sim** | Arquiteturas de backbone pré-treinadas |
| `torch` | `torch==2.5.1` | 2.5.1+cpu | ml | **sim** | Runtime de inferência de rede neural |
| `torchvision` | `torchvision==0.20.1` | 0.20.1+cpu | ml | não | Transformações de imagem para modelos torch |
| `torchxrayvision` | `torchxrayvision>=1.2` | 1.5.4 | ml | **sim** | Modelos de radiografia de tórax pré-treinados e publicados |

## Monitoramento de anomalias

O workflow `.github/workflows/security.yml` executa `pip-audit` contra a
base OSV a cada push e semanalmente. Uma vulnerabilidade em item marcado
como relevante para a segurança abre avaliação de impacto obrigatória
antes da próxima liberação (IEC 62304 §7.4).

## Pendências

- [ ] Registrar fabricante e URL do repositório de anomalias por item
- [ ] Anexar avaliação de impacto para cada item relevante à segurança
- [ ] Congelar o lockfile (`uv.lock`) como parte do registro de liberação
