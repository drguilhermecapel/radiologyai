import os
import sys
import subprocess
import json
from pathlib import Path

print("=" * 60)
print("INÍCIO RÁPIDO - Treinamento NIH ChestX-ray14")
print("=" * 60)

# Configuração automática para o caminho do dataset
# Tenta ler o caminho do dataset de uma variável de ambiente, caso contrário, usa um padrão
DATASET_ROOT = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", "/home/ubuntu/NIH_CHEST_XRAY_SIMULATED")

def check_environment():
    """Verifica ambiente e instala dependências"""
    print("\n1. Verificando ambiente Python...")
    
    # Verificar versão Python
    python_version = sys.version_info
    print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or python_version.minor < 8:
        print("   Aviso: Recomendado Python 3.8 ou superior")
        return False
    
    print("   OK: Versão Python adequada")
    return True

def install_dependencies():
    """Instala pacotes necessários"""
    print("\n2. Instalando dependências...")
    
    packages = [
        "tensorflow==2.15.0",
        "numpy",
        "pandas",
        "pillow",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "opencv-python-headless"
    ]
    
    for package in packages:
        print(f"   Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("   OK: Dependências instaladas")

def verify_dataset():
    """Verifica se o dataset está presente"""
    print(f"\n3. Verificando dataset em {DATASET_ROOT}...")
    
    dataset_path = Path(DATASET_ROOT)
    
    # Verificar pasta principal
    if not dataset_path.exists():
        print(f"   ERRO: Pasta não encontrada: {DATASET_ROOT}")
        print("   Por favor, verifique o caminho do dataset")
        return False
    
    # Verificar CSV
    csv_file = dataset_path / "Data_Entry_2017_v2020.csv"
    if not csv_file.exists():
        # Tentar com nome alternativo
        csv_file = dataset_path / "Data_Entry_2017.csv"
    
    if not csv_file.exists():
        print("   ERRO: Arquivo CSV de labels não encontrado")
        return False
    
    # Verificar pasta de imagens
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        print("   ERRO: Pasta de imagens não encontrada")
        return False
    
    # Contar imagens
    image_count = len(list(images_dir.glob("*.png")))
    print(f"   OK: Dataset encontrado: {image_count} imagens")
    
    return True, str(csv_file)

def create_training_config(csv_path):
    """Cria arquivo de configuração"""
    print("\n4. Criando configuração de treinamento...")
    
    config = {
        "data_dir": DATASET_ROOT,
        "image_dir": os.path.join(DATASET_ROOT, "images"),
        "csv_file": csv_path,
        "output_dir": os.path.join(DATASET_ROOT, "models_trained"),
        
        # Configurações otimizadas
        "batch_size": 16,
        "image_size": [256, 256],  # Reduzido para treinar mais rápido
        "epochs": 30,  # Reduzido para demonstração
        "learning_rate": 0.0001,
        "validation_split": 0.15,
        "test_split": 0.15,
        
        # Classes principais (5 mais comuns + No Finding)
        "selected_classes": [
            "No Finding",
            "Infiltration",
            "Effusion", 
            "Atelectasis",
            "Nodule",
            "Pneumothorax"
        ]
    }
    
    # Salvar configuração
    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("   OK: Configuração criada: training_config.json")
    return config

def create_simple_training_script(config):
    """Cria script de treinamento simplificado"""
    print("\n5. Criando script de treinamento...")
    
    script_content = f'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduzir logs do TensorFlow

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

print("Iniciando Treinamento Simplificado NIH ChestX-ray14")
print("=" * 60)

# Carregar configuração
with open("training_config.json", "r") as f:
    CONFIG = json.load(f)

# Função para carregar e processar imagem
def load_and_preprocess_image(image_path, size):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, tuple(size))
    img = img.astype(np.float32) / 255.0
    img = np.stack([img] * 3, axis=-1)
    return img

# Preparar dados
print("Carregando dados...")
df = pd.read_csv(CONFIG["csv_file"])

# Filtrar apenas imagens que existem
image_dir = Path(CONFIG["image_dir"])
valid_indices = []
for idx, row in df.iterrows():
    if (image_dir / row["Image Index"]).exists():
        valid_indices.append(idx)
    if idx % 5000 == 0:
        print(f"   Verificadas {{idx}}/{{len(df)}} imagens...")

df_valid = df.iloc[valid_indices]
print(f"Imagens válidas: {{len(df_valid)}}")

# Preparar labels
print("Preparando labels...")
mlb = MultiLabelBinarizer(classes=CONFIG["selected_classes"])

labels_list = df_valid["Finding Labels"].apply(lambda x: [
    label for label in x.split("|") if label in CONFIG["selected_classes"]
] or ["No Finding"]).tolist()

labels = mlb.fit_transform(labels_list)

# Dividir dados
print("Dividindo dados...")
X_train, X_test, y_train, y_test = train_test_split(
    df_valid["Image Index"].values, labels, 
    test_size=CONFIG["test_split"], 
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=CONFIG["validation_split"]/(1-CONFIG["test_split"]),
    random_state=42
)

print(f"   Treino: {{len(X_train)}} | Validação: {{len(X_val)}} | Teste: {{len(X_test)}}")

# Criar geradores de dados
class SimpleDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_names, labels, batch_size=32):
        self.image_names = image_names
        self.labels = labels
        self.batch_size = batch_size
        self.image_dir = Path(CONFIG["image_dir"])
        self.image_size = CONFIG["image_size"]
        
    def __len__(self):
        return int(np.ceil(len(self.image_names) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.image_names[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        for img_name in batch_x:
            img = load_and_preprocess_image(self.image_dir / img_name, self.image_size)
            if img is not None:
                images.append(img)
            else:
                images.append(np.zeros((*self.image_size, 3)))
                
        return np.array(images), batch_y

# Criar modelo simplificado
print("Construindo modelo...")
def create_simple_model():
    model = keras.Sequential([
        # Base convolucional
        layers.Conv2D(32, 3, activation="relu", input_shape=(*CONFIG["image_size"], 3)),
        layers.MaxPooling2D(2),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(CONFIG["selected_classes"]), activation="sigmoid")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    return model

model = create_simple_model()
model.summary()

# Criar geradores
train_gen = SimpleDataGenerator(X_train, y_train, CONFIG["batch_size"])
val_gen = SimpleDataGenerator(X_val, y_val, CONFIG["batch_size"])

# Criar diretório de saída
output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

# Callbacks
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        str(output_dir / "best_model.h5"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Treinar
print("Iniciando treinamento...")
print(f"   Épocas: {{CONFIG['epochs']}}")
print(f"   Batch size: {{CONFIG['batch_size']}}")
print(f"   Classes: {{', '.join(CONFIG['selected_classes'])}}")

history = model.fit(
    train_gen,
    epochs=CONFIG["epochs"],
    validation_data=val_gen,
    callbacks=callbacks_list,
    verbose=1
)

# Salvar modelo final
model.save(str(output_dir / "final_model.h5"))

# Avaliar no teste
print("Avaliando modelo...")
test_gen = SimpleDataGenerator(X_test, y_test, CONFIG["batch_size"])
test_loss, test_acc, test_auc = model.evaluate(test_gen, verbose=1)

print(f"Treinamento concluído!")
print(f"   Loss teste: {{test_loss:.4f}}")
print(f"   Acurácia teste: {{test_acc:.4f}}")
print(f"   AUC teste: {{test_auc:.4f}}")
print(f"   Modelos salvos em: {{output_dir}}")

# Salvar resultados
results = {{
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "test_auc": float(test_auc),
    "classes": CONFIG["selected_classes"],
    "training_completed": datetime.now().isoformat()
}}

with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
'''
    
    with open("train_simple.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("   OK: Script criado: train_simple.py")

def main():
    """Função principal"""
    
    # 1. Verificar ambiente
    if not check_environment():
        print("\nERRO: Ambiente Python incompatível")
        input("Pressione ENTER para sair...")
        return
    
    # 2. Instalar dependências
    try:
        install_dependencies()
    except Exception as e:
        print(f"\nERRO: Erro ao instalar dependências: {e}")
        input("Pressione ENTER para sair...")
        return
    
    # 3. Verificar dataset
    dataset_check = verify_dataset()
    if not dataset_check:
        print("\nERRO: Dataset não encontrado ou incompleto")
        input("Pressione ENTER para sair...")
        return
    
    _, csv_path = dataset_check
    
    # 4. Criar configuração
    config = create_training_config(csv_path)
    
    # 5. Criar script de treinamento
    create_simple_training_script(config)
    
    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO CONCLUÍDA!")
    print("=" * 60)
    
    print("\nPróximos passos:")
    print(f"   1. Certifique-se de que seu dataset NIH ChestX-ray14 está em: {DATASET_ROOT}")
    print("   2. Execute: python train_simple.py")
    print("   3. O treinamento levará 1-3 horas (dependendo do hardware)")
    print(f"   4. Os modelos serão salvos em: {DATASET_ROOT}/models_trained")
    
    print("\nDicas:")
    print("   - Use GPU se disponível (instale tensorflow-gpu)")
    print("   - Monitore o uso de memória RAM")
    print("   - O modelo está configurado para 6 classes principais")
    
    resposta = input("\nDeseja iniciar o treinamento agora? (S/N): ")
    
    if resposta.upper() == 'S':
        print("\nIniciando treinamento...")
        os.system("python train_simple.py")
    else:
        print("\nExecute 'python train_simple.py' quando estiver pronto!")

if __name__ == "__main__":
    main()


