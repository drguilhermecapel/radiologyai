---
doc_id: REG-08
title: Lista SOUP (Software of Unknown Provenance)
status: GERADO AUTOMATICAMENTE — não editar à mão
generated_by: scripts/soup.py
source_commit: 8554c64
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
| `fastapi` | `fastapi>=0.110,<1` | 0.165.0 | api | não | Camada de API REST |
| `python-multipart` | `python-multipart>=0.0.9` | 0.0.34 | api | não | Upload de arquivos na API |
| `uvicorn` | `uvicorn[standard]>=0.29` | 0.54.0 | api | não | Servidor ASGI |
| `numpy` | `numpy>=2.0,<3` | 2.4.6 | core | **sim** | Aritmética de arrays em todo o pipeline de imagem e métricas |
| `pydantic` | `pydantic>=2.7,<3` | 2.14.4 | core | **sim** | Validação de contratos de dados (model cards, metadados, resultados) |
| `pydantic-settings` | `pydantic-settings>=2.3,<3` | 2.16.3 | core | não | Carga de configuração a partir do ambiente |
| `pydicom` | `pydicom>=3.0.1,<4` | 3.0.2 | core | **sim** | Parsing de DICOM, acesso a tags e des-identificação |
| `pyyaml` | `PyYAML>=6.0,<7` | 6.0.4 | core | não | Leitura de model cards em YAML |
| `typer` | `typer>=0.12,<1` | 0.29.2 | core | não | Interface de linha de comando |
| `pandas` | `pandas>=2.2,<4` | 3.1.1 | eval | não | Tabelas de rótulos e manifestos de datasets de avaliação |
| `pyarrow` | `pyarrow>=17` | 26.0.1 | eval | não | Leitura de splits oficiais em Parquet (avaliação) |
| `scikit-learn` | `scikit-learn>=1.4,<2` | 1.9.1 | eval | não | Métricas de referência e utilidades de avaliação |
| `scipy` | `scipy>=1.11,<2` | 1.17.1 | eval | não | Estatística usada nos intervalos de confiança |
| `huggingface-hub` | `huggingface-hub>=0.26` | — | foundation | não | Download de pesos publicados no Hugging Face (verificados por sha256) |
| `open-clip-torch` | `open-clip-torch>=2.29` | — | foundation | **sim** | Modelos CLIP biomédicos (BiomedCLIP) para escores exploratórios |
| `safetensors` | `safetensors>=0.4.5` | 0.8.0 | foundation | **sim** | Carga segura (sem pickle) de pesos de modelos |
| `transformers` | `transformers>=4.46` | 5.24.0 | foundation | **sim** | Codificadores de fundação e modelos visão-linguagem pré-treinados |
| `nibabel` | `nibabel>=5.2,<6` | 5.5.0 | imaging | **sim** | Leitura e escrita de volumes NIfTI (TC/RM, máscaras de segmentação) |
| `pillow` | `Pillow>=10.0` | 13.0.0 | imaging | não | Decodificação de imagem não-DICOM |
| `pyjpegls` | `pyjpegls>=1.4,<2` | 1.5.1 | imaging | **sim** | Decodificação JPEG-LS de PixelData DICOM |
| `pylibjpeg` | `pylibjpeg>=2.0,<3` | 2.1.0 | imaging | **sim** | Interface de decodificação de PixelData comprimido para o pydicom |
| `pylibjpeg-openjpeg` | `pylibjpeg-openjpeg>=2.4,<3` | 2.6.0 | imaging | **sim** | Decodificação JPEG 2000 de PixelData DICOM |
| `pylibjpeg-rle` | `pylibjpeg-rle>=2.0,<3` | 2.2.0 | imaging | **sim** | Decodificação RLE Lossless de PixelData DICOM |
| `python-gdcm` | `python-gdcm>=3.0.24,<4` | 3.2.6 | imaging | **sim** | Decodificação JPEG Lossless (processo 14) de PixelData DICOM |
| `simpleitk` | `SimpleITK>=2.3,<3` | 2.5.8 | imaging | **sim** | Leitura de séries e reamostragem volumétrica (TC/RM) |
| `opencv-python-headless` | `opencv-python-headless>=4.10,<5` | 4.14.0.94 | mammography | **sim** | Pré-processamento bit-exato dos modelos de mamografia publicados |
| `monai` | `monai>=1.4,<2` | 1.6.2 | ml | **sim** | Transformações e arquiteturas de imagem médica (2D e 3D) |
| `onnxruntime` | `onnxruntime>=1.18` | 1.32.0 | ml | **sim** | Runtime de inferência para implantação em CPU |
| `timm` | `timm>=1.0` | 1.0.31 | ml | **sim** | Arquiteturas de backbone pré-treinadas |
| `torch` | `torch>=2.8,<3` | 2.15.0+cpu | ml | **sim** | Runtime de inferência de rede neural |
| `torchvision` | `torchvision>=0.23` | 0.29.2+cpu | ml | não | Transformações de imagem para modelos torch |
| `torchxrayvision` | `torchxrayvision>=1.5.7` | 1.5.7 | ml | **sim** | Modelos de radiografia de tórax pré-treinados e publicados |
| `alembic` | `alembic>=1.14,<2` | 1.20.1 | platform | não | Migrações versionadas do esquema do banco |
| `argon2-cffi` | `argon2-cffi>=23.1` | 25.1.0 | platform | não | Hash de senhas dos usuários da plataforma |
| `dicomweb-client` | `dicomweb-client>=0.60,<1` | 0.61.2 | platform | não | Consulta e recuperação de estudos via DICOMweb (QIDO/WADO/STOW) |
| `highdicom` | `highdicom>=0.25,<1` | 0.29.0 | platform | **sim** | Codificação de DICOM SR e DICOM SEG dos resultados |
| `httpx` | `httpx>=0.27,<1` | 0.28.1 | platform | não | Cliente HTTP para download de pesos e integração DICOMweb |
| `prometheus-client` | `prometheus-client>=0.21,<1` | — | platform | não | Métricas operacionais da API |
| `reportlab` | `reportlab>=4.1,<6` | 5.1.0 | platform | não | Geração do laudo de pesquisa em PDF |
| `sqlalchemy` | `sqlalchemy>=2.0.36,<3` | 2.1.9 | platform | não | Persistência de estudos, análises e trilha de auditoria |
| `totalsegmentator` | `TotalSegmentator>=2.10,<3` | 2.18.0 | volumetric | **sim** | Segmentação anatômica e de patologias em TC/RM (nnU-Net pré-treinado) |

## Monitoramento de anomalias

O workflow `.github/workflows/security.yml` executa `pip-audit` contra a
base OSV a cada push e semanalmente. Uma vulnerabilidade em item marcado
como relevante para a segurança abre avaliação de impacto obrigatória
antes da próxima liberação (IEC 62304 §7.4).

## Pendências

- [ ] Registrar fabricante e URL do repositório de anomalias por item
- [ ] Anexar avaliação de impacto para cada item relevante à segurança
- [ ] Congelar o lockfile (`uv.lock`) como parte do registro de liberação
