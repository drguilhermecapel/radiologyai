"""
setup_environment.py - Configura o ambiente e corrige problemas de compatibilidade
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configuração de logging sem emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_tensorflow_windows():
    """Configura TensorFlow para Windows com compatibilidade garantida"""
    logger.info("Configurando TensorFlow para Windows...")
    
    # Remove versões conflitantes
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "tensorflow", "tensorflow-intel", "-y"])
    
    # Instala versão compatível com Windows
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "tensorflow==2.13.0",
        "--no-cache-dir"
    ])
    
    # Instala dependências compatíveis
    deps = [
        "numpy==1.24.3",
        "opencv-python==4.8.1.78",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "pandas==2.0.3",
        "h5py==3.9.0",
        "Pillow==10.0.0"
    ]
    
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", dep])

def fix_logging_encoding():
    """Corrige problemas de encoding nos logs"""
    # Cria arquivo de configuração de logging
    config_content = '''import logging
import sys

# Remove emojis dos logs
class NoEmojiFormatter(logging.Formatter):
    def format(self, record):
        # Remove emojis comuns
        emoji_map = {
            '🧠': '[BRAIN]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '📊': '[CHART]',
            '📈': '[GROWTH]',
            '🏥': '[HOSPITAL]',
            '📁': '[FOLDER]',
            '🎯': '[TARGET]',
            '📋': '[CLIPBOARD]',
            '📐': '[METRICS]'
        }
        
        msg = super().format(record)
        for emoji, text in emoji_map.items():
            msg = msg.replace(emoji, text)
        
        return msg

# Configura logging global
def setup_logging(name='MedAI'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Handler para arquivo
    fh = logging.FileHandler(f'{name.lower()}_training.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # Handler para console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter sem emojis
    formatter = NoEmojiFormatter(
        '%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
'''
    
    with open('src/logging_config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)

def create_fixed_train_script():
    """Cria script de treinamento corrigido"""
    script_content = '''"""
train_fixed.py - Script de treinamento corrigido
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow

import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime

# Importa configuração de logging sem emojis
from src.logging_config import setup_logging

# Configuração
logger = setup_logging('MedAI.Training')

# Verifica GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"GPUs disponíveis: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    logger.warning("Nenhuma GPU detectada - usando CPU")

def create_simple_cnn(input_shape=(384, 384, 3), num_classes=2):
    """Cria CNN simples para teste"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        
        # Bloco 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Bloco 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Bloco 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classificador
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_and_preprocess_data(data_dir, batch_size=8):
    """Carrega dados usando tf.data para eficiência"""
    data_dir = Path(data_dir)
    
    # Define parâmetros
    img_height, img_width = 384, 384
    
    # Cria datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / 'train',
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / 'train',
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    # Normalização
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Augmentação para treino
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomFlip("horizontal"),
    ])
    
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Otimização de performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def train_model(data_dir, epochs=10, batch_size=8):
    """Função principal de treinamento"""
    logger.info("[START] Iniciando treinamento")
    
    # Carrega dados
    logger.info("Carregando dados...")
    train_ds, val_ds = load_and_preprocess_data(data_dir, batch_size)
    
    # Cria modelo
    logger.info("Criando modelo...")
    model = create_simple_cnn()
    
    # Compila modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Treina modelo
    logger.info("Iniciando treinamento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salva resultados
    logger.info("[OK] Treinamento concluído")
    
    # Avalia modelo final
    results = model.evaluate(val_ds, verbose=0)
    logger.info(f"Resultados finais - Loss: {results[0]:.4f}, Acc: {results[1]:.4f}, AUC: {results[2]:.4f}")
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/nih_chest_xray/organized')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    # Cria diretórios necessários
    os.makedirs('models', exist_ok=True)
    
    # Treina modelo
    model, history = train_model(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
'''
    
    with open('train_fixed.py', 'w', encoding='utf-8') as f:
        f.write(script_content)

if __name__ == "__main__":
    logger.info("Iniciando setup do ambiente...")
    
    # 1. Configura TensorFlow
    setup_tensorflow_windows()
    
    # 2. Corrige logging
    fix_logging_encoding()
    
    # 3. Cria script de treinamento corrigido
    create_fixed_train_script()
    
    logger.info("Setup concluído! Execute: python train_fixed.py")

