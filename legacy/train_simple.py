
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
        print(f"   Verificadas {idx}/{len(df)} imagens...")

df_valid = df.iloc[valid_indices]
print(f"Imagens válidas: {len(df_valid)}")

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

print(f"   Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

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
print(f"   Épocas: {CONFIG['epochs']}")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Classes: {', '.join(CONFIG['selected_classes'])}")

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
print(f"   Loss teste: {test_loss:.4f}")
print(f"   Acurácia teste: {test_acc:.4f}")
print(f"   AUC teste: {test_auc:.4f}")
print(f"   Modelos salvos em: {output_dir}")

# Salvar resultados
results = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "test_auc": float(test_auc),
    "classes": CONFIG["selected_classes"],
    "training_completed": datetime.now().isoformat()
}

with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
