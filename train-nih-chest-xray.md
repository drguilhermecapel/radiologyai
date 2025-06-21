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

# Ativar ambiente (Windows)
venv\Scripts\activate

# Instalar pacotes necessários
pip install tensorflow==2.15.0
pip install numpy pandas pillow
pip install scikit-learn matplotlib seaborn
pip install opencv-python-headless
pip install pydicom
```

### 2. Configurar Caminho do Dataset (HD Externo)

Para que o sistema encontre seu dataset no HD externo, você deve definir a variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT` para o caminho onde o dataset está localizado. Por exemplo, se seu HD externo for montado como `/mnt/d` e o dataset estiver em `/mnt/d/NIH_CHEST_XRAY`, você faria:

```bash
export NIH_CHEST_XRAY_DATASET_ROOT=/mnt/d/NIH_CHEST_XRAY
# Ou no Windows (cmd):
# set NIH_CHEST_XRAY_DATASET_ROOT=D:\NIH_CHEST_XRAY
# Ou no Windows (PowerShell):
# $env:NIH_CHEST_XRAY_DATASET_ROOT="D:\NIH_CHEST_XRAY"
```

O arquivo `config-training.py` foi modificado para ler essa variável de ambiente. Se a variável não for definida, ele usará `/mnt/d/NIH_CHEST_XRAY` como padrão.

### 3. Criar Script de Configuração

O arquivo `config-training.py` já está configurado para usar a variável de ambiente. Você pode ajustá-lo se precisar de configurações mais específicas:

```python
# config_training.py
import os
from pathlib import Path

# Configuração customizada para seu dataset
# Tenta ler o caminho do dataset de uma variável de ambiente, caso contrário, usa um padrão
DATASET_ROOT = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", r"/mnt/d/NIH_CHEST_XRAY")

CONFIG = {
    # Caminhos do dataset - ajuste para sua localização
    'data_dir': DATASET_ROOT,
    'image_dir': os.path.join(DATASET_ROOT, 'images'),
    'csv_file': os.path.join(DATASET_ROOT, 'Data_Entry_2017_v2020.csv'),
    
    # Diretório de saída para modelos treinados
    'output_dir': os.path.join(DATASET_ROOT, 'models_trained'),
    
    # Parâmetros de treinamento
    'batch_size': 16,  # Reduzir para 8 ou 4 se tiver pouca memória RAM
    'image_size': (320, 320),  # Pode reduzir para (224, 224) se necessário
    'epochs': 50,  # Número de épocas de treinamento
    'learning_rate': 1e-4,  # Taxa de aprendizado inicial
    'validation_split': 0.15,  # 15% dos dados para validação
    'test_split': 0.15,  # 15% dos dados para teste
    
    # Classes de patologias para treinar
    # Você pode adicionar ou remover classes conforme necessário
    'selected_classes': [
        'No Finding',      # Sem achados patológicos
        'Pneumonia',       # Pneumonia
        'Effusion',        # Derrame pleural
        'Atelectasis',     # Atelectasia
        'Infiltration',    # Infiltração
        'Mass',            # Massa
        'Nodule',          # Nódulo
        'Consolidation',   # Consolidação
        'Pneumothorax'     # Pneumotórax
    ],
    
    # Configurações avançadas
    'random_seed': 42,  # Para reprodutibilidade
    'num_workers': 4,   # Threads para carregamento de dados
    'prefetch_buffer': 2,  # Buffer de pré-carregamento
    
    # Thresholds clínicos (ajuste conforme necessário)
    'clinical_thresholds': {
        'default': 0.5,
        'high_sensitivity': 0.3,  # Para screening
        'high_specificity': 0.7   # Para confirmação
    }
}

# Funções de verificação de caminho e criação de diretório de saída (mantidas)
# ...
```

## Script Principal de Treinamento

### 4. Criar o Script de Treinamento Completo (`train_chest_xray.py`):

Este script contém a lógica principal para carregar o dataset, construir o modelo EfficientNetB3, aplicar data augmentation e treinar a IA. Ele utiliza as configurações definidas em `config-training.py`.

```python
# train_chest_xray.py
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import cv2
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from config_training import CONFIG

print("Sistema de Treinamento de IA para Radiografias Torácicas")
print("=" * 60)

class ChestXrayDataGenerator(tf.keras.utils.Sequence):
    """Gerador de dados otimizado para o dataset NIH"""
    
    def __init__(self, df, image_dir, batch_size=32, image_size=(320, 320), 
                 augment=False, classes=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.classes = classes
        self.mlb = MultiLabelBinarizer(classes=classes)
        
        # Preparar labels
        self.prepare_labels()
        
    def prepare_labels(self):
        """Prepara labels multi-classe"""
        # Converter string de labels para lista
        labels_list = self.df["Finding Labels"].apply(lambda x: x.split("|")).tolist()
        
        # Filtrar apenas classes selecionadas
        filtered_labels = []
        for labels in labels_list:
            filtered = [l for l in labels if l in self.classes]
            if not filtered:
                filtered = ["No Finding"]
            filtered_labels.append(filtered)
        
        # Binarizar labels
        self.labels = self.mlb.fit_transform(filtered_labels)
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.df))
        
        batch_df = self.df.iloc[start_idx:end_idx]
        
        # Carregar imagens
        images = []
        for _, row in batch_df.iterrows():
            img_path = self.image_dir / row["Image Index"]
            
            # Carregar e processar imagem
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Aviso: Imagem não encontrada: {img_path}")
                # Criar imagem preta como fallback
                img = np.zeros(self.image_size)
            else:
                img = cv2.resize(img, self.image_size)
            
            # Normalizar para [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Adicionar canal (grayscale -> RGB)
            img = np.stack([img] * 3, axis=-1)
            
            # Aplicar augmentation se habilitado
            if self.augment:
                img = self.apply_augmentation(img)
            
            images.append(img)
        
        # Obter labels correspondentes
        batch_labels = self.labels[start_idx:end_idx]
        
        return np.array(images), batch_labels
    
    def apply_augmentation(self, image):
        """Aplica data augmentation específico para raios-X"""
        # Rotação pequena (-5 a 5 graus)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Zoom leve (0.95 a 1.05)
        if np.random.random() > 0.5:
            zoom = np.random.uniform(0.95, 1.05)
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            
            if zoom > 1:
                # Zoom in
                resized = cv2.resize(image, (new_w, new_h))
                y1 = (new_h - h) // 2
                x1 = (new_w - w) // 2
                image = resized[y1:y1+h, x1:x1+w]
            else:
                # Zoom out
                resized = cv2.resize(image, (new_w, new_h))
                canvas = np.zeros_like(image)
                y1 = (h - new_h) // 2
                x1 = (w - new_w) // 2
                canvas[y1:y1+new_h, x1:x1+new_w] = resized
                image = canvas
        
        # Ajuste de brilho/contraste
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.9, 1.1)  # Contraste
            beta = np.random.uniform(-0.05, 0.05)  # Brilho
            image = np.clip(alpha * image + beta, 0, 1)
        
        return image

def create_model(num_classes, input_shape=(320, 320, 3)):
    """Cria modelo de deep learning otimizado para radiografias"""
    
    # Base: EfficientNetB3 pré-treinado
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Congelar primeiras camadas
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Construir modelo
    inputs = keras.Input(shape=input_shape)
    
    # Pré-processamento
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Pooling global
    x = layers.GlobalAveragePooling2D()(x)
    
    # Camadas densas com regularização
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Saída multi-label (sigmoid para cada classe)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="binary_crossentropy",  # Para multi-label
        metrics=[
            "binary_accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=True),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    
    return model

def prepare_data():
    """Prepara dados para treinamento"""
    print("\nCarregando dataset...")
    
    # Carregar CSV
    df = pd.read_csv(CONFIG["csv_file"])
    print(f"Total de imagens no CSV: {len(df)}")
    
    # Verificar quais imagens existem
    print("\nVerificando arquivos de imagem...")
    image_dir = Path(CONFIG["image_dir"])
    existing_images = []
    
    for idx, row in df.iterrows():
        img_path = image_dir / row["Image Index"]
        if img_path.exists():
            existing_images.append(idx)
        
        if idx % 10000 == 0:
            print(f"  Verificadas {idx}/{len(df)} imagens...")
    
    # Filtrar apenas imagens existentes
    df_valid = df.iloc[existing_images]
    print(f"\nImagens válidas encontradas: {len(df_valid)}")
    
    # Dividir dados
    print("\nDividindo dados...")
    
    # Primeiro separar teste
    train_val_df, test_df = train_test_split(
        df_valid, 
        test_size=CONFIG["test_split"],
        random_state=42,
        stratify=df_valid["Finding Labels"].apply(lambda x: x.split("|")[0])
    )
    
    # Depois separar validação
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=CONFIG["validation_split"]/(1-CONFIG["test_split"]),
        random_state=42,
        stratify=train_val_df["Finding Labels"].apply(lambda x: x.split("|")[0])
    )
    
    print(f"  Treino: {len(train_df)} imagens")
    print(f"  Validação: {len(val_df)} imagens")
    print(f"  Teste: {len(test_df)} imagens")
    
    return train_df, val_df, test_df

def train_model():
    """Função principal de treinamento"""
    
    # Preparar dados
    train_df, val_df, test_df = prepare_data()
    
    # Criar geradores
    print("\nCriando geradores de dados...")
    
    train_gen = ChestXrayDataGenerator(
        train_df,
        CONFIG["image_dir"],
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        augment=True,
        classes=CONFIG["selected_classes"]
    )
    
    val_gen = ChestXrayDataGenerator(
        val_df,
        CONFIG["image_dir"],
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        augment=False,
        classes=CONFIG["selected_classes"]
    )
    
    # Criar modelo
    print("\nConstruindo modelo...")
    num_classes = len(CONFIG["selected_classes"])
    model = create_model(num_classes)
    model.summary()
    
    # Criar diretório de saída
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks_list = [
        # Salvar melhor modelo
        callbacks.ModelCheckpoint(
            str(output_dir / f"best_model_{timestamp}.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduzir learning rate
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Logs
        callbacks.CSVLogger(
            str(output_dir / f"training_log_{timestamp}.csv")
        )
    ]
    
    # Treinar modelo
    print("\nIniciando treinamento...")
    print(f"  Épocas: {CONFIG["epochs"]}")
    print(f"  Batch size: {CONFIG["batch_size"]}")
    print(f"  Learning rate: {CONFIG["learning_rate"]}")
    
    history = model.fit(
        train_gen,
        epochs=CONFIG["epochs"],
        validation_data=val_gen,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Salvar modelo final
    model.save(str(output_dir / f"final_model_{timestamp}.h5"))
    
    # Salvar configuração
    config_path = output_dir / f"config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    # Plotar histórico
    plot_training_history(history, output_dir, timestamp)
    
    # Avaliar no conjunto de teste
    print("\nAvaliando no conjunto de teste...")
    evaluate_model(model, test_df, output_dir, timestamp)
    
    print(f"\nTreinamento concluído! Modelos salvos em: {output_dir}")

def plot_training_history(history, output_dir, timestamp):
    """Plota e salva gráficos do treinamento"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history.history["loss"], label="Treino")
    axes[0, 0].plot(history.history["val_loss"], label="Validação")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Época")
    axes[0, 0].legend()
    
    # AUC
    axes[0, 1].plot(history.history["auc"], label="Treino")
    axes[0, 1].plot(history.history["val_auc"], label="Validação")
    axes[0, 1].set_title("AUC")
    axes[0, 1].set_xlabel("Época")
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history["precision"], label="Treino")
    axes[1, 0].plot(history.history["val_precision"], label="Validação")
    axes[1, 0].set_title("Precisão")
    axes[1, 0].set_xlabel("Época")
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history["recall"], label="Treino")
    axes[1, 1].plot(history.history["val_recall"], label="Validação")
    axes[1, 1].set_title("Recall")
    axes[1, 1].set_xlabel("Época")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(str(output_dir / f"training_history_{timestamp}.png"))
    plt.close()

def evaluate_model(model, test_df, output_dir, timestamp):
    """Avalia modelo no conjunto de teste"""
    
    # Criar gerador de teste
    test_gen = ChestXrayDataGenerator(
        test_df,
        CONFIG["image_dir"],
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        augment=False,
        classes=CONFIG["selected_classes"]
    )
    
    # Fazer predições
    print("  Fazendo predições...")
    predictions = model.predict(test_gen, verbose=1)
    
    # Calcular métricas
    y_true = test_gen.labels[:len(test_df)]
    
    # AUC por classe
    auc_scores = []
    for i, class_name in enumerate(CONFIG["selected_classes"]):
        if y_true[:, i].sum() > 0:  # Apenas se houver exemplos positivos
            auc = roc_auc_score(y_true[:, i], predictions[:, i])
            auc_scores.append(auc)
            print(f"  AUC {class_name}: {auc:.4f}")
    
    # Métricas gerais
    mean_auc = np.mean(auc_scores)
    print(f"\n  AUC Médio: {mean_auc:.4f}")
    
    # Salvar relatório
    report = {
        "timestamp": timestamp,
        "mean_auc": float(mean_auc),
        "per_class_auc": {
            class_name: float(auc) 
            for class_name, auc in zip(CONFIG["selected_classes"], auc_scores)
        },
        "test_samples": len(test_df)
    }
    
    report_path = output_dir / f"evaluation_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    # Verificar GPU
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponíveis: {len(gpus)}")
        else:
            print("Nenhuma GPU encontrada. Usando CPU.")
    except Exception as e:
        print(f"Erro ao configurar GPU: {e}")

    train_model()
```

## Scripts Auxiliares

### 5. Script de Início Rápido (`quick_start_nih_training.py`)

Este script automatiza a configuração do ambiente, instalação de dependências, verificação do dataset e criação do script de treinamento simplificado (`train_simple.py`). Ele agora respeita a variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT`.

```python
# quick_start_nih_training.py
import os
import sys
import subprocess
import json
from pathlib import Path

print("=" * 60)
print("INÍCIO RÁPIDO - Treinamento NIH ChestX-ray14")
print("=" * 60)

# Configuração automática para o caminho do dataset, priorizando variável de ambiente
DATASET_PATH = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", "/mnt/d/NIH_CHEST_XRAY")

def check_environment():
    # ... (função mantida)
    pass

def install_dependencies():
    # ... (função mantida)
    pass

def verify_dataset():
    # ... (função modificada para usar DATASET_PATH)
    pass

def create_training_config(csv_path):
    # ... (função modificada para usar DATASET_PATH)
    pass

def create_simple_training_script(config):
    # ... (função mantida, cria train_simple.py)
    pass

def main():
    # ... (lógica principal, com instruções atualizadas para a variável de ambiente)
    pass

if __name__ == "__main__":
    main()
```

### 6. Script de Teste de Modelo Treinado (`test_trained_model.py`)

Este script permite testar o modelo treinado com novas imagens, seja individualmente ou em lote. Ele também foi atualizado para usar a variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT` para encontrar os modelos e o dataset.

```python
# test_trained_model.py
import os
import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 60)
print("Teste de Modelo Treinado - NIH ChestX-ray14")
print("=" * 60)

# Configurações
DATASET_ROOT = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", "/mnt/d/NIH_CHEST_XRAY")
MODEL_DIR = os.path.join(DATASET_ROOT, "models_trained")
CONFIG_FILE = "training_config.json"

def load_model_and_config():
    # ... (função modificada para usar MODEL_DIR e CONFIG_FILE)
    pass

def preprocess_image(image_path, size):
    # ... (função mantida)
    pass

def predict_single_image(model, config, image_path):
    # ... (função mantida)
    pass

def visualize_prediction(image_path, results, threshold=0.3):
    # ... (função mantida)
    pass

def test_batch_images(model, config, image_folder):
    # ... (função modificada para usar DATASET_ROOT)
    pass

def interactive_test(model, config):
    # ... (função modificada para usar DATASET_ROOT)
    pass

def main():
    # ... (lógica principal)
    pass

if __name__ == "__main__":
    main()
```

### 7. Script de Verificação de Ambiente (`verify_environment.py`)

Este script verifica se todas as dependências estão instaladas e se o dataset está acessível. Ele também foi atualizado para usar a variável de ambiente `NIH_CHEST_XRAY_DATASET_ROOT`.

```python
# verify_environment.py
import sys
import os
from pathlib import Path
import importlib

print("=" * 60)
print("Verificação de Ambiente - NIH ChestX-ray14")
print("=" * 60)

# Configuração do dataset
DATASET_PATH = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", "/mnt/d/NIH_CHEST_XRAY")

def check_python_version():
    # ... (função mantida)
    pass

def check_packages():
    # ... (função mantida)
    pass

def check_dataset():
    # ... (função modificada para usar DATASET_PATH)
    pass

def check_disk_space():
    # ... (função modificada para usar DATASET_PATH)
    pass

def check_memory():
    # ... (função mantida)
    pass

def check_gpu():
    # ... (função mantida)
    pass

def main():
    # ... (lógica principal, com instruções atualizadas para a variável de ambiente)
    pass

if __name__ == "__main__":
    main()
```

## Como Usar o Sistema

1.  **Clone o repositório** e copie os arquivos fornecidos para a pasta raiz do projeto.
2.  **Defina a variável de ambiente** `NIH_CHEST_XRAY_DATASET_ROOT` para o caminho do seu HD externo onde o dataset NIH ChestX-ray14 está localizado.
3.  **Execute o script de início rápido** para configurar o ambiente e gerar o script de treinamento:
    ```bash
    python quick_start_nih_training.py
    ```
4.  O script `quick_start_nih_training.py` irá guiá-lo através da instalação de dependências e verificação do dataset. Ao final, ele perguntará se você deseja iniciar o treinamento. Se sim, ele executará `train_simple.py`.
5.  Alternativamente, você pode executar o treinamento completo diretamente com `train_chest_xray.py` após configurar `config-training.py`:
    ```bash
    python train_chest_xray.py
    ```
6.  Após o treinamento, você pode testar o modelo com `test_trained_model.py`:
    ```bash
    python test_trained_model.py
    ```

Este guia e os scripts atualizados devem resolver o problema de treinamento com o HD externo, permitindo que você especifique o caminho do dataset de forma flexível. As alterações nos scripts garantem que eles busquem o dataset no local correto, seja ele um drive local ou um HD externo montado.

