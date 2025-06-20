# Script de treinamento rápido para demonstração
import sys
import os
sys.path.append('src')

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import cv2

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_model(input_shape=(224, 224, 3), num_classes=6):
    """Cria modelo simples para demonstração rápida"""
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

class SimpleDataGenerator(tf.keras.utils.Sequence):
    """Gerador de dados simples"""
    
    def __init__(self, df, image_dir, batch_size=8, image_size=(224, 224), 
                 num_classes=6, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        
        # Mapeamento simplificado
        self.pathology_map = {
            'No Finding': 0, 'Pneumonia': 1, 'Cardiomegaly': 2,
            'Effusion': 3, 'Mass': 4, 'Nodule': 5
        }
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]
        
        X, y = self.__data_generation(batch_df)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_df):
        X = np.empty((len(batch_df), *self.image_size, 3), dtype=np.float32)
        y = np.empty((len(batch_df), self.num_classes), dtype=np.float32)
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            # Carregar imagem
            img_path = self.image_dir / row['Image Index']
            img = self.load_and_preprocess_image(img_path)
            X[i,] = img
            
            # Criar label
            label = row['Finding Labels']
            label_vector = np.zeros(self.num_classes)
            if label in self.pathology_map:
                label_vector[self.pathology_map[label]] = 1
            y[i,] = label_vector
        
        return X, y
    
    def load_and_preprocess_image(self, img_path):
        """Carrega e preprocessa imagem"""
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Não foi possível carregar: {img_path}")
            
            # Redimensionar
            img = cv2.resize(img, self.image_size)
            
            # Converter para RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Normalizar
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"Erro ao processar {img_path}: {e}")
            return np.zeros((*self.image_size, 3), dtype=np.float32)

def quick_train():
    """Treinamento rápido para demonstração"""
    
    logger.info("Iniciando treinamento rápido de demonstração")
    
    # Preparar dados
    data_path = Path("data/nih_chest_xray/organized")
    if not data_path.exists():
        logger.error(f"Diretório não encontrado: {data_path}")
        return None
    
    # Criar DataFrame
    all_images = []
    for pathology_dir in data_path.iterdir():
        if pathology_dir.is_dir() and pathology_dir.name != 'dataset_stats.json':
            pathology = pathology_dir.name
            for img_file in pathology_dir.glob("*.png"):
                all_images.append({
                    'Image Index': img_file.name,
                    'Finding Labels': pathology,
                    'Path': str(img_file)
                })
    
    df = pd.DataFrame(all_images)
    logger.info(f"Total de imagens: {len(df)}")
    
    # Dividir dados
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, 
                                       stratify=df['Finding Labels'])
    
    logger.info(f"Treinamento: {len(train_df)}, Validação: {len(val_df)}")
    
    # Criar geradores
    train_gen = SimpleDataGenerator(train_df, data_path.parent / "images", batch_size=4)
    val_gen = SimpleDataGenerator(val_df, data_path.parent / "images", batch_size=4)
    
    # Criar modelo simples
    model = create_simple_model()
    
    # Compilar
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Criar diretório para modelos
    model_dir = Path("models/pre_trained/simple_demo")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.h5"),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            restore_best_weights=True,
            mode='max'
        )
    ]
    
    # Treinar (apenas 5 épocas para demonstração)
    logger.info("Iniciando treinamento (5 épocas)...")
    history = model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Salvar modelo
    model.save(model_dir / "model.h5")
    
    # Salvar configuração
    config = {
        "name": "simple_demo",
        "version": "1.0",
        "trained_on": "Synthetic ChestX-ray",
        "classes": ["No Finding", "Pneumonia", "Cardiomegaly", "Effusion", "Mass", "Nodule"],
        "input_shape": [224, 224, 3],
        "training_date": datetime.now().isoformat(),
        "epochs_trained": len(history.history['loss']),
        "best_val_auc": max(history.history['val_auc']) if 'val_auc' in history.history else 0.0
    }
    
    with open(model_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Treinamento concluído!")
    logger.info(f"Melhor AUC: {config['best_val_auc']:.4f}")
    
    return model, history

if __name__ == "__main__":
    # Verificar TensorFlow
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Treinar modelo
    try:
        model, history = quick_train()
        if model is not None:
            logger.info("✅ Modelo treinado com sucesso")
        else:
            logger.error("❌ Falha no treinamento")
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        import traceback
        traceback.print_exc()

