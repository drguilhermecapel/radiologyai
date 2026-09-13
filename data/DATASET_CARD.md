# Dados neste repositório

**Nenhum dado de paciente jamais esteve neste repositório, e nenhum jamais estará.**

## O que foi removido

`data/nih_chest_xray/` continha 120 imagens PNG geradas por `create_synthetic_dataset.py`,
que as desenha com primitivas do OpenCV:

```python
cv2.ellipse(img, (cx-80, cy), (60,120), 0, 0, 360, 80, -1)   # "Pulmão esquerdo"
for i in range(8):                                            # "Costelas"
    cv2.line(img, (cx-120, y), (cx+120, y), 100, 2)
```

Essas imagens foram gravadas com **nomes de arquivos do NIH ChestX-ray14**
(`00000001_000.png`, `00000002_013.png`, …), e o `Data_Entry_2017_v2020.csv`
que as acompanhava tinha 3 linhas apontando para `dummy_image_0001.png`.

Isso não é apenas dado sintético: é **risco de proveniência**. Quem copiasse esse
diretório passaria a possuir arquivos que se apresentam como NIH ChestX-ray14 e não são.

Foram removidos. `create_synthetic_dataset.py` continua no repositório e pode
regenerá-los, mas qualquer regeneração deve usar nomes explicitamente sintéticos
(`synthetic_phantom_0001.png`) e nunca imitar a convenção de um dataset real.

## Política de dados a partir de agora

1. **Nenhum pixel no git.** `.gitignore` cobre `data/`, `*.dcm`, `*.nii*`, `*.mhd`, `*.raw`.
   Exceção única: fixtures sintéticas explicitamente nomeadas em `tests/data/`.
2. **Manifests são commitados, dados não.** `datasets/manifests/*.csv` carrega
   `image_id`, `patient_id`, rótulos e **sha256 por arquivo**. É o registro de
   reprodutibilidade e custa poucos megabytes.
3. **Splits por paciente, sempre.** Um check de CI assere interseção zero de
   `patient_id` entre manifests de treino, validação e teste. Vazamento por paciente
   é a forma mais comum de um número publicado de IA radiológica estar errado.
4. **Todo dataset ganha um datasheet** em `docs/datasheets/<dataset>.md`: fonte,
   licença, termos de uso, direito de redistribuição, população, **proveniência do
   rótulo (minerado por NLP vs. adjudicado por radiologista)**, vieses conhecidos e citação.

Ver [`ROADMAP.md`](../ROADMAP.md) §7 para os datasets planejados por modalidade.
