#!/usr/bin/env python3
"""
Sistema de Treinamento Completo para IA Radiológica - Uso Clínico
Dataset: NIH ChestX-ray14 (112,120 imagens)
Objetivo: Máxima acurácia clínica com alta sensibilidade e especificidade
Hardware: CPU com 32GB RAM
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, mixed_precision
import cv2
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import gc
import warnings
warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    """
    Encoder customizado para converter tipos numpy em tipos Python nativos.
    Essencial para salvar métricas de treinamento de modelos TensorFlow/Keras.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClinicalRadiologyAI")

# Configurações globais
CONFIG = {
    "data_dir": "data/nih_chest_xray",
    "image_dir": "data/nih_chest_xray/images",
    "csv_file": "data/nih_chest_xray/Data_Entry_2017_v2020.csv",
    "output_dir": "models/clinical_production",
    "batch_size": 16,  # Otimizado para CPU
    "image_size": (320, 320),  # Balanceamento entre qualidade e velocidade
    "epochs": 100,
    "learning_rate": 1e-4,
    "min_lr": 1e-7,
    "patience": 10,
    "num_classes": 14,
    "validation_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    "checkpoint_freq": 5,
    "mixed_precision": False,  # Desabilitado para CPU
    "num_workers": 4,
    "prefetch_buffer": 2
}

# Classes de patologias
PATHOLOGY_CLASSES = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", 
    "Infiltration", "Mass", "Nodule", "Pneumonia", 
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
    "Fibrosis", "Pleural_Thickening"
]

# Configurar TensorFlow para CPU otimizado
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(8)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

class ClinicalDataGenerator(tf.keras.utils.Sequence):
    """Gerador de dados otimizado para grandes datasets médicos"""
    
    def __init__(self, df, image_dir, batch_size=32, image_size=(320, 320), 
                 augment=False, shuffle=True, cache_size=1000):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.cache = {}
        self.cache_size = cache_size
        self.indexes = np.arange(len(self.df))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]
        
        X = np.zeros((len(batch_df), *self.image_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_df), CONFIG["num_classes"]), dtype=np.float32)
        
        for i, (idx, row) in enumerate(batch_df.iterrows()):
            # Carregar imagem com cache
            img = self._load_image(row["Image Index"])
            
            # Augmentação médica específica
            if self.augment:
                img = self._medical_augmentation(img)
            
            X[i] = img
            
            # Labels multi-label
            labels = row["Finding Labels"].split("|")
            for label in labels:
                if label in PATHOLOGY_CLASSES:
                    y[i, PATHOLOGY_CLASSES.index(label)] = 1.0
        
        return X, y
    
    def _load_image(self, filename):
        """Carrega imagem com cache eficiente"""
        if filename in self.cache:
            return self.cache[filename].copy()
        
        img_path = self.image_dir / filename
        img = cv2.imread(str(img_path))
        
        if img is None:
            logger.warning(f"Imagem não encontrada: {img_path}")
            return np.zeros((*self.image_size, 3), dtype=np.float32)
        
        # Converter para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar mantendo aspect ratio
        img = self._resize_with_padding(img)
        
        # Normalizar
        img = img.astype(np.float32) / 255.0
        
        # Adicionar ao cache se houver espaço
        if len(self.cache) < self.cache_size:
            self.cache[filename] = img.copy()
        
        return img
    
    def _resize_with_padding(self, img):
        """Redimensiona mantendo proporções e adiciona padding"""
        h, w = img.shape[:2]
        target_h, target_w = self.image_size
        
        # Calcular escala
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensionar
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Adicionar padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        
        img = cv2.copyMakeBorder(img, pad_top, pad_h - pad_top, 
                                pad_left, pad_w - pad_left,
                                cv2.BORDER_CONSTANT, value=0)
        
        return img
    
    def _medical_augmentation(self, img):
        """Augmentação específica para imagens médicas"""
        # Rotação pequena (±10 graus)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            center = (self.image_size[1] // 2, self.image_size[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, self.image_size)
        
        # Ajuste de brilho/contraste (simulando diferentes equipamentos)
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.9, 1.1)  # Contraste
            beta = np.random.uniform(-0.05, 0.05)  # Brilho
            img = np.clip(alpha * img + beta, 0, 1)
        
        # Flip horizontal (anatomicamente válido para raio-X de tórax)
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Ruído gaussiano leve (simulando variação de equipamento)
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        # Limpar cache parcialmente para liberar memória
        if len(self.cache) > self.cache_size * 0.8:
            keys_to_remove = list(self.cache.keys())[:int(self.cache_size * 0.2)]
            for key in keys_to_remove:
                del self.cache[key]

def create_clinical_model(input_shape=(320, 320, 3), num_classes=14):
    """Cria modelo otimizado para uso clínico"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Arquitetura customizada otimizada para radiologia
    # Bloco 1
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Bloco 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Bloco 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Bloco 4
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Bloco 5 - Atenção
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Mecanismo de atenção simples
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(512 // 16, activation="relu")(attention)
    attention = layers.Dense(512, activation="sigmoid")(attention)
    attention = layers.Reshape((1, 1, 512))(attention)
    x = layers.Multiply()([x, attention])
    
    # Pooling global
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classificador
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    # Saída multi-label com sigmoid
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ClinicalRadiologyNet")
    
    return model

class ClinicalMetricsCallback(callbacks.Callback):
    """Callback para métricas clínicas detalhadas"""
    
    def __init__(self, validation_data, output_dir):
        super().__init__()
        self.validation_data = validation_data
        self.output_dir = Path(output_dir)
        self.history = {
            "auc_per_class": [],
            "sensitivity_per_class": [],
            "specificity_per_class": []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # A cada 5 épocas
            y_true = []
            y_pred = []
            
            # Coletar predições
            for i in range(len(self.validation_data)):
                X_batch, y_batch = self.validation_data[i]
                pred_batch = self.model.predict(X_batch, verbose=0)
                y_true.append(y_batch)
                y_pred.append(pred_batch)
            
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            
            # Calcular métricas por classe
            auc_scores = []
            sensitivities = []
            specificities = []
            
            for i in range(CONFIG["num_classes"]):
                if y_true[:, i].sum() > 0:  # Classe presente
                    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    
                    # Calcular sensibilidade e especificidade no threshold ótimo
                    fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[optimal_idx]
                    
                    y_pred_binary = (y_pred[:, i] > optimal_threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_binary).ravel()
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    auc_scores.append(auc)
                    sensitivities.append(sensitivity)
                    specificities.append(specificity)
            
            # Registrar métricas
            mean_auc = np.mean(auc_scores)
            mean_sensitivity = np.mean(sensitivities)
            mean_specificity = np.mean(specificities)
            
            logger.info(f"Época {epoch + 1} - Métricas Clínicas:")
            logger.info(f"  AUC Médio: {mean_auc:.4f}")
            logger.info(f"  Sensibilidade Média: {mean_sensitivity:.4f}")
            logger.info(f"  Especificidade Média: {mean_specificity:.4f}")
            
            # Salvar relatório detalhado
            if epoch % 10 == 0:
                self._save_clinical_report(epoch, y_true, y_pred, auc_scores, 
                                         sensitivities, specificities)
    
    def _save_clinical_report(self, epoch, y_true, y_pred, auc_scores, 
                            sensitivities, specificities):
        """Salva relatório clínico detalhado"""
        report = {
            "epoch": epoch + 1,
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": {
                "mean_auc": float(np.mean(auc_scores)),
                "mean_sensitivity": float(np.mean(sensitivities)),
                "mean_specificity": float(np.mean(specificities))
            },
            "per_class_metrics": {}
        }
        
        for i, class_name in enumerate(PATHOLOGY_CLASSES):
            if i < len(auc_scores):
                report["per_class_metrics"][class_name] = {
                    "auc": float(auc_scores[i]),
                    "sensitivity": float(sensitivities[i]),
                    "specificity": float(specificities[i])
                }
        
        report_path = self.output_dir / f"clinical_report_epoch_{epoch + 1}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

def prepare_clinical_data():
    """Prepara dados para treinamento clínico"""
    logger.info("Carregando dataset NIH ChestX-ray14...")
    
    # Carregar CSV
    df = pd.read_csv(CONFIG["csv_file"])
    logger.info(f"Total de imagens: {len(df)}")
    
    # Filtrar imagens válidas
    image_dir = Path(CONFIG["image_dir"])
    valid_images = []
    
    for idx, row in df.iterrows():
        if (image_dir / row["Image Index"]).exists():
            valid_images.append(idx)
        
        if idx % 10000 == 0:
            logger.info(f"Verificadas {idx} imagens...")
    
    df_valid = df.iloc[valid_images]
    logger.info(f"Imagens válidas encontradas: {len(df_valid)}")
    
    # Dividir dados mantendo distribuição de patologias
    # Primeiro, separar conjunto de teste
    train_val_df, test_df = train_test_split(
        df_valid, 
        test_size=CONFIG["test_split"],
        random_state=CONFIG["random_seed"],
        stratify=df_valid["Finding Labels"].apply(lambda x: x.split("|")[0])
    )
    
    # Depois, separar treino e validação
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=CONFIG["validation_split"] / (1 - CONFIG["test_split"]),
        random_state=CONFIG["random_seed"],
        stratify=train_val_df["Finding Labels"].apply(lambda x: x.split("|")[0])
    )
    
    logger.info(f"Divisão dos dados:")
    logger.info(f"  Treino: {len(train_df)} imagens")
    logger.info(f"  Validação: {len(val_df)} imagens")
    logger.info(f"  Teste: {len(test_df)} imagens")
    
    return train_df, val_df, test_df

def calculate_class_weights(train_df):
    """Calcula pesos para classes desbalanceadas"""
    # Contar ocorrências de cada patologia
    class_counts = np.zeros(CONFIG["num_classes"])
    
    for labels in train_df["Finding Labels"]:
        for label in labels.split("|"):
            if label in PATHOLOGY_CLASSES:
                class_counts[PATHOLOGY_CLASSES.index(label)] += 1
    
    # Calcular pesos inversamente proporcionais
    total_samples = len(train_df)
    class_weights = {}
    
    for i, count in enumerate(class_counts):
        if count > 0:
            weight = total_samples / (CONFIG["num_classes"] * count)
            # Limitar pesos para evitar instabilidade
            class_weights[i] = np.clip(weight, 0.5, 5.0)
        else:
            class_weights[i] = 1.0
    
    logger.info("Pesos das classes calculados:")
    for i, class_name in enumerate(PATHOLOGY_CLASSES):
        logger.info(f"  {class_name}: {class_weights.get(i, 1.0):.3f}")
    
    return class_weights

def train_clinical_model():
    """Função principal para treinar o modelo clínico"""
    
    # 1. Preparar dados
    train_df, val_df, test_df = prepare_clinical_data()
    
    # 2. Calcular pesos das classes
    class_weights = calculate_class_weights(train_df)
    
    # 3. Criar geradores de dados
    logger.info("Criando geradores de dados...")
    train_generator = ClinicalDataGenerator(
        train_df, CONFIG["image_dir"], batch_size=CONFIG["batch_size"], 
        image_size=CONFIG["image_size"], augment=True, shuffle=True
    )
    val_generator = ClinicalDataGenerator(
        val_df, CONFIG["image_dir"], batch_size=CONFIG["batch_size"], 
        image_size=CONFIG["image_size"], augment=False, shuffle=False
    )
    test_generator = ClinicalDataGenerator(
        test_df, CONFIG["image_dir"], batch_size=CONFIG["batch_size"], 
        image_size=CONFIG["image_size"], augment=False, shuffle=False
    )
    
    # 4. Criar modelo
    logger.info("Criando modelo clínico...")
    model = create_clinical_model(input_shape=(*CONFIG["image_size"], 3), 
                                  num_classes=CONFIG["num_classes"])
    
    # 5. Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    
    # 6. Callbacks
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=os.path.join(CONFIG["output_dir"], "best_model.h5"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,
        patience=CONFIG["patience"] // 2,
        min_lr=CONFIG["min_lr"],
        mode="max",
        verbose=1
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor="val_auc",
        patience=CONFIG["patience"],
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
    
    # Ajuste para salvar a cada N épocas, não steps
    class CustomCheckpoint(callbacks.Callback):
        def __init__(self, filepath, save_freq):
            super().__init__()
            self.filepath = filepath
            self.save_freq = save_freq

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.save_freq == 0:
                self.model.save(self.filepath.format(epoch=epoch + 1))
                logger.info(f"Modelo salvo em checkpoint_epoch_{epoch + 1}.h5")

    custom_checkpoint = CustomCheckpoint(
        filepath=os.path.join(CONFIG["output_dir"], "checkpoint_epoch_{epoch:03d}.h5"),
        save_freq=CONFIG["checkpoint_freq"]
    )

    clinical_metrics_callback = ClinicalMetricsCallback(
        validation_data=val_generator, output_dir=CONFIG["output_dir"]
    )
    
    callbacks_list = [
        model_checkpoint_callback,
        reduce_lr_on_plateau,
        early_stopping,
        custom_checkpoint,
        clinical_metrics_callback
    ]
    
    # 7. Treinar modelo
    logger.info(f"Iniciando treinamento clínico...")
    logger.info(f"Configuração:")
    logger.info(f"  - Épocas: {CONFIG['epochs']}")
    logger.info(f"  - Batch size: {CONFIG['batch_size']}")
    logger.info(f"  - Learning rate inicial: {CONFIG['learning_rate']}")
    logger.info(f"  - Imagem size: {CONFIG['image_size']}")

    history = model.fit(
        train_generator,
        epochs=CONFIG["epochs"],
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks_list
    )
    
    # 8. Salvar histórico de treinamento
    history_path = Path(CONFIG["output_dir"]) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Histórico de treinamento salvo em {history_path}")

    # 9. Avaliar modelo no conjunto de teste
    logger.info("Avaliando modelo no conjunto de teste...")
    # Carregar o melhor modelo salvo
    best_model_path = os.path.join(CONFIG["output_dir"], "best_model.h5")
    if os.path.exists(best_model_path):
        model = keras.models.load_model(best_model_path)
        logger.info(f"Melhor modelo carregado de {best_model_path}")
    else:
        logger.warning("Melhor modelo não encontrado. Usando o modelo treinado final.")

    test_loss, test_binary_accuracy, test_auc, test_precision, test_recall = model.evaluate(
        test_generator,
        workers=CONFIG["num_workers"],
        use_multiprocessing=True,
        max_queue_size=CONFIG["prefetch_buffer"] * CONFIG["batch_size"]
    )
    logger.info(f"Métricas no conjunto de teste:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Binary Accuracy: {test_binary_accuracy:.4f}")
    logger.info(f"  AUC: {test_auc:.4f}")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall: {test_recall:.4f}")

    # Gerar relatório de classificação detalhado
    y_true_test = []
    y_pred_test = []
    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        pred_batch = model.predict(X_batch, verbose=0)
        y_true_test.append(y_batch)
        y_pred_test.append(pred_batch)
    
    y_true_test = np.vstack(y_true_test)
    y_pred_test = np.vstack(y_pred_test)

    # Calcular métricas por classe para o conjunto de teste
    test_auc_scores = []
    test_sensitivities = []
    test_specificities = []

    for i in range(CONFIG["num_classes"]):
        if y_true_test[:, i].sum() > 0:  # Classe presente
            auc = roc_auc_score(y_true_test[:, i], y_pred_test[:, i])
            fpr, tpr, thresholds = roc_curve(y_true_test[:, i], y_pred_test[:, i])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred_binary = (y_pred_test[:, i] > optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_test[:, i], y_pred_binary).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_auc_scores.append(auc)
            test_sensitivities.append(sensitivity)
            test_specificities.append(specificity)

    test_report = {
        "overall_metrics": {
            "mean_auc": float(np.mean(test_auc_scores)),
            "mean_sensitivity": float(np.mean(test_sensitivities)),
            "mean_specificity": float(np.mean(test_specificities))
        },
        "per_class_metrics": {}
    }

    for i, class_name in enumerate(PATHOLOGY_CLASSES):
        if i < len(test_auc_scores):
            test_report["per_class_metrics"][class_name] = {
                "auc": float(test_auc_scores[i]),
                "sensitivity": float(test_sensitivities[i]),
                "specificity": float(test_specificities[i])
            }

    test_report_path = Path(CONFIG["output_dir"]) / "test_evaluation_report.json"
    with open(test_report_path, "w") as f:
        json.dump(test_report, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Relatório de avaliação no conjunto de teste salvo em {test_report_path}")

    logger.info("Treinamento e avaliação concluídos.")

if __name__ == "__main__":
    train_clinical_model()


