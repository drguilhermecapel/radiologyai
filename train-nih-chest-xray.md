# Guia Completo: Treinamento de IA com Dataset NIH ChestX-ray14

## Análise Científica do Dataset

O **NIH ChestX-ray14** é um dos maiores datasets públicos de radiografias torácicas, contendo:
- **112.120 imagens** de raios-X frontais de 30.805 pacientes únicos
- **14 classes de patologias** + categoria "No Finding" (sem achados)
- Anotações extraídas automaticamente de laudos radiológicos usando NLP
- Resolução original de 1024×1024 pixels

### Patologias Identificáveis:
1. **Atelectasis** - Colapso pulmonar parcial ou completo
2. **Cardiomegaly** - Aumento do coração
3. **Effusion** - Derrame pleural
4. **Infiltration** - Infiltrado pulmonar
5. **Mass** - Massa pulmonar
6. **Nodule** - Nódulo pulmonar
7. **Pneumonia** - Infecção pulmonar
8. **Pneumothorax** - Ar no espaço pleural
9. **Consolidation** - Consolidação pulmonar
10. **Edema** - Edema pulmonar
11. **Emphysema** - Enfisema
12. **Fibrosis** - Fibrose pulmonar
13. **Pleural Thickening** - Espessamento pleural
14. **Hernia** - Hérnia diafragmática

## Preparação do Ambiente

### 1. Instalar Dependências

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (Linux/macOS)
source venv/bin/activate

# Ativar ambiente (Windows)
vvenv\Scripts\activate

# Instalar pacotes necessários
pip install tensorflow==2.15.0
pip install numpy pandas pillow
pip install scikit-learn matplotlib seaborn
pip install opencv-python-headless
pip install pydicom
```

### 2. Configurar Caminho do Dataset (HD Externo / Linux)

Para utilizar um HD externo ou um caminho específico no Linux, você deve definir a variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT` antes de executar os scripts. Isso garante que todos os scripts (configuração, treinamento e teste) encontrem o dataset corretamente.

**Exemplo de configuração no Linux (terminal):**

```bash
# Exemplo: seu HD externo está montado em /media/seu_usuario/MeuHD/NIH_CHEST_XRAY
export NIH_CHEST_XRAY_DATASET_ROOT="/media/seu_usuario/MeuHD/NIH_CHEST_XRAY"

# Ou se o dataset estiver em outro local, por exemplo, /data/radiology/NIH_CHEST_XRAY
export NIH_CHEST_XRAY_DATASET_ROOT="/data/radiology/NIH_CHEST_XRAY"

# Para verificar se a variável foi definida corretamente
echo $NIH_CHEST_XRAY_DATASET_ROOT
```

**Estrutura esperada do dataset no caminho configurado:**

```
/caminho/do/seu/dataset/NIH_CHEST_XRAY/
├── images/                 # Pasta contendo todas as imagens .png
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
├── Data_Entry_2017_v2020.csv # Arquivo CSV com os rótulos das imagens
└── models_trained/         # Pasta onde os modelos treinados serão salvos (será criada automaticamente)
```

### 3. Criar Script de Configuração (`config_training.py`)

O arquivo `config_training.py` foi modificado para ler o caminho do dataset da variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT`. Você não precisa editar este arquivo diretamente, a menos que queira ajustar outros parâmetros de treinamento (batch size, épocas, etc.).

```python
# config_training.py (Exemplo - não precisa editar diretamente)
import os
from pathlib import Path

DATASET_ROOT = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", "/home/ubuntu/NIH_CHEST_XRAY_SIMULATED") # Caminho padrão de fallback

CONFIG = {
    'data_dir': Path(DATASET_ROOT),
    'image_dir': Path(DATASET_ROOT) / 'images',
    'csv_file': Path(DATASET_ROOT) / 'Data_Entry_2017_v2020.csv',
    'output_dir': Path(DATASET_ROOT) / 'models_trained',
    # ... outros parâmetros de treinamento
}
```

## Script Principal de Treinamento

### 4. Executar o Script de Início Rápido (`quick-start-script.py`)

Este script irá verificar seu ambiente, instalar as dependências necessárias, criar o arquivo de configuração (`training_config.json`) e gerar um script de treinamento simplificado (`train_simple.py`).

**Passos:**

1. **Defina a variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT`** (conforme explicado no passo 2).
2. **Execute o script:**
   ```bash
   python quick-start-script.py
   ```
   O script irá guiá-lo e, ao final, perguntará se deseja iniciar o treinamento.

### 5. Iniciar o Treinamento

Após a execução do `quick-start-script.py`, um novo arquivo chamado `train_simple.py` será gerado. Este é o script que de fato executa o treinamento do modelo.

**Para iniciar o treinamento:**

```bash
python train_simple.py
```

**Observações:**
- O treinamento pode levar várias horas, dependendo do seu hardware (CPU vs. GPU).
- Os modelos treinados e os logs serão salvos na pasta `models_trained` dentro do seu `NIH_CHEST_XRAY_DATASET_ROOT`.

## Testando o Modelo Treinado

### 6. Utilizar `test-trained-model.py`

Após o treinamento, você pode usar o script `test-trained-model.py` para carregar o modelo salvo e fazer inferências em novas imagens.

**Para testar o modelo:**

1. **Defina a variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT`** (se ainda não estiver definida na sua sessão).
2. **Execute o script:**
   ```bash
   python test-trained-model.py
   ```
   O script entrará em um modo interativo, permitindo que você teste imagens únicas, pastas de imagens ou imagens de exemplo do dataset.

## Solução de Problemas Comuns

- **"ERRO: Pasta não encontrada" ou "Arquivo CSV de labels não encontrado"**: Verifique se a variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT` está apontando para o diretório correto e se a estrutura de pastas (`images/` e `Data_Entry_2017_v2020.csv`) está correta dentro desse diretório.
- **Erro de memória (OOM)**: Reduza o `batch_size` no arquivo `config_training.py` (para 8 ou 4, por exemplo) ou diminua o `image_size`.
- **Treinamento muito lento**: Certifique-se de que o TensorFlow está utilizando sua GPU (se disponível). Você pode verificar isso executando `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`. Se não houver GPU, o treinamento será executado na CPU, o que é significativamente mais lento.


