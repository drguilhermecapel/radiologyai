# Script de treinamento customizado para MedAI Radiologia
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
from sklearn.utils.class_weight import compute_class_weight
import cv2

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar mixed precision para acelerar treinamento
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

class MedicalDataGenerator(tf.keras.utils.Sequence):
    """Gerador de dados customizado para imagens médicas"""
    
    def __init__(self, df, image_dir, batch_size=32, image_size=(384, 384), 
                 num_classes=14, augment=False, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        
        # Mapeamento de patologias
        self.pathology_map = {
            'No Finding': 0, 'Pneumonia': 1, 'Effusion': 2, 'Atelectasis': 3,
            'Consolidation': 4, 'Pneumothorax': 5, 'Cardiomegaly': 6,
            'Mass': 7, 'Nodule': 8, 'Infiltration': 9, 'Emphysema': 10,
            'Fibrosis': 11, 'Pleural_Thickening': 12, 'Hernia': 13
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
            
            if self.augment:
                img = self.apply_medical_augmentation(img)
            
            X[i,] = img
            
            # Criar label multi-classe
            labels = row['Finding Labels'].split('|')
            label_vector = np.zeros(self.num_classes)
            for label in labels:
                if label in self.pathology_map:
                    label_vector[self.pathology_map[label]] = 1
            y[i,] = label_vector
        
        return X, y
    
    def load_and_preprocess_image(self, img_path):
        """Carrega e preprocessa imagem médica"""
        try:
            # Carregar imagem
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Não foi possível carregar a imagem: {img_path}")
            
            # Redimensionar
            img = cv2.resize(img, self.image_size)
            
            # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            
            # Converter para RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Normalizar
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"Erro ao processar imagem {img_path}: {e}")
            # Retornar imagem em branco em caso de erro
            return np.zeros((*self.image_size, 3), dtype=np.float32)
    
    def apply_medical_augmentation(self, img):
        """Aplica augmentação específica para imagens médicas"""
        # Rotação limitada (radiografias não devem ser muito rotacionadas)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Ajuste de brilho (simular diferentes exposições)
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
        
        # Zoom limitado
        if np.random.random() > 0.5:
            zoom_factor = np.random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            
            if zoom_factor > 1:
                # Crop central
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img_resized = cv2.resize(img, (new_w, new_h))
                img = img_resized[start_h:start_h+h, start_w:start_w+w]
            else:
                # Pad
                img_resized = cv2.resize(img, (new_w, new_h))
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                img = cv2.copyMakeBorder(img_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)
        
        return img

def create_efficientnet_model(input_shape=(384, 384, 3), num_classes=14):
    """Cria modelo EfficientNetV2 para radiologia"""
    base_model = tf.keras.applications.EfficientNetV2L(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Descongelar últimas camadas para fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Adicionar cabeças específicas para radiologia
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    return model

def create_vision_transformer_model(input_shape=(384, 384, 3), num_classes=14):
    """Cria modelo Vision Transformer simplificado"""
    inputs = layers.Input(shape=input_shape)
    
    # Patch embedding
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Dividir imagem em patches
    patches = layers.Conv2D(768, patch_size, strides=patch_size)(inputs)
    patches = layers.Reshape((num_patches, 768))(patches)
    
    # Position embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=768)(positions)
    patches = patches + position_embedding
    
    # Transformer blocks
    for _ in range(6):  # 6 camadas transformer
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=12, key_dim=64, dropout=0.1
        )(patches, patches)
        attention_output = layers.Dropout(0.1)(attention_output)
        patches = layers.LayerNormalization(epsilon=1e-6)(patches + attention_output)
        
        # MLP
        mlp_output = layers.Dense(3072, activation='gelu')(patches)
        mlp_output = layers.Dropout(0.1)(mlp_output)
        mlp_output = layers.Dense(768)(mlp_output)
        mlp_output = layers.Dropout(0.1)(mlp_output)
        patches = layers.LayerNormalization(epsilon=1e-6)(patches + mlp_output)
    
    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.3)(representation)
    outputs = layers.Dense(num_classes, activation='sigmoid', dtype='float32')(representation)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(model_name="efficientnetv2", data_dir="data/nih_chest_xray/organized", 
                epochs=100, batch_size=16):
    """Treina modelo específico"""
    
    logger.info(f"Iniciando treinamento do modelo {model_name}")
    
    # Preparar dados
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Diretório de dados não encontrado: {data_path}")
        return None
    
    # Criar DataFrame com todas as imagens
    all_images = []
    for pathology_dir in data_path.iterdir():
        if pathology_dir.is_dir():
            pathology = pathology_dir.name
            for img_file in pathology_dir.glob("*.png"):
                all_images.append({
                    'Image Index': img_file.name,
                    'Finding Labels': pathology,
                    'Path': str(img_file)
                })
    
    df = pd.DataFrame(all_images)
    logger.info(f"Total de imagens encontradas: {len(df)}")
    
    # Dividir dados
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, 
                                       stratify=df['Finding Labels'])
    
    logger.info(f"Treinamento: {len(train_df)} imagens")
    logger.info(f"Validação: {len(val_df)} imagens")
    
    # Criar geradores
    train_gen = MedicalDataGenerator(
        train_df, data_path.parent / "images", 
        batch_size=batch_size, augment=True
    )
    val_gen = MedicalDataGenerator(
        val_df, data_path.parent / "images", 
        batch_size=batch_size, augment=False
    )
    
    # Criar modelo
    if model_name == "efficientnetv2":
        model = create_efficientnet_model()
    elif model_name == "vision_transformer":
        model = create_vision_transformer_model()
    else:
        logger.error(f"Modelo não reconhecido: {model_name}")
        return None
    
    # Compilar modelo
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    model_dir = Path(f"models/pre_trained/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.h5"),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        callbacks.TensorBoard(
            log_dir=f"logs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            histogram_freq=1
        )
    ]
    
    # Treinar
    logger.info("Iniciando treinamento...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Salvar modelo final
    model.save(model_dir / "model.h5")
    
    # Salvar configuração
    config = {
        "name": model_name,
        "version": "2.0",
        "trained_on": "NIH ChestX-ray14",
        "classes": list(train_gen.pathology_map.keys()),
        "input_shape": [384, 384, 3],
        "training_date": datetime.now().isoformat(),
        "epochs_trained": len(history.history['loss']),
        "best_val_auc": max(history.history['val_auc'])
    }
    
    with open(model_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Treinamento do modelo {model_name} concluído!")
    logger.info(f"Melhor AUC de validação: {config['best_val_auc']:.4f}")
    
    return model, history

if __name__ == "__main__":
    # Verificar GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU disponível: {gpus}")
        # Configurar crescimento de memória
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning("Nenhuma GPU detectada. Treinamento será mais lento.")
    
    # Treinar modelos
    models_to_train = ["efficientnetv2", "vision_transformer"]
    
    for model_name in models_to_train:
        try:
            model, history = train_model(model_name, epochs=50, batch_size=8)
            if model is not None:
                logger.info(f"✅ Modelo {model_name} treinado com sucesso")
            else:
                logger.error(f"❌ Falha no treinamento do modelo {model_name}")
        except Exception as e:
            logger.error(f"❌ Erro no treinamento do modelo {model_name}: {e}")

