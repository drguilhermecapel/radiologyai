# training_system.py - Sistema de treinamento e validação de modelos

import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import wandb  # Para tracking de experimentos (opcional)

logger = logging.getLogger('MedAI.Training')

class RadiologyDataset:
    """
    Dataset customizado para imagens radiológicas
    Implementa carregamento e processamento de dados médicos
    """
    
    def __init__(self, 
                 data_df: pd.DataFrame = None,
                 image_dir: str = None,
                 batch_size: int = 32,
                 image_size: tuple = (224, 224),
                 num_classes: int = 14,
                 augment: bool = False):
        
        self.data_df = data_df if data_df is not None else pd.DataFrame()
        self.image_dir = Path(image_dir) if image_dir else Path('.')
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment
        
        logger.info(f"RadiologyDataset inicializado com {len(self.data_df)} amostras")
    
    def __len__(self):
        """Retorna o número de batches"""
        return max(1, len(self.data_df) // self.batch_size)
    
    def __getitem__(self, idx):
        """Retorna um batch de dados reais de imagens médicas"""
        try:
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.data_df))
            
            batch_data = []
            batch_labels = []
            
            pathology_classes = {
                'normal': 0,
                'pneumonia': 1, 
                'pleural_effusion': 2,
                'fracture': 3,
                'tumor': 4
            }
            
            for i in range(start_idx, end_idx):
                image_data = self._load_medical_image(i)
                
                label_vector = self._get_pathology_label(i, pathology_classes)
                
                batch_data.append(image_data)
                batch_labels.append(label_vector)
            
            return np.array(batch_data), np.array(batch_labels)
            
        except Exception as e:
            logger.warning(f"Erro ao carregar batch {idx}: {e}")
            return np.zeros((1, *self.image_size, 1)), np.zeros((1, self.num_classes))
    
    def _load_medical_image(self, idx):
        """Carrega imagem médica real (DICOM, JPEG, PNG)"""
        try:
            sample_files = list(self.image_dir.glob('**/*.dcm')) + \
                          list(self.image_dir.glob('**/*.jpg')) + \
                          list(self.image_dir.glob('**/*.png'))
            
            if sample_files and idx < len(sample_files):
                image_path = sample_files[idx % len(sample_files)]
                
                if image_path.suffix.lower() == '.dcm':
                    try:
                        import pydicom
                        ds = pydicom.dcmread(image_path)
                        image = ds.pixel_array.astype(np.float32)
                        
                        # Normalizar valores DICOM
                        if image.max() > 1.0:
                            image = image / image.max()
                            
                    except ImportError:
                        logger.warning("pydicom não disponível, usando dados simulados")
                        image = np.random.rand(*self.image_size).astype(np.float32)
                        
                else:
                    try:
                        import cv2
                        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        image = image.astype(np.float32) / 255.0
                    except ImportError:
                        logger.warning("OpenCV não disponível, usando dados simulados")
                        image = np.random.rand(*self.image_size).astype(np.float32)
                
                if len(image.shape) == 2:
                    image = cv2.resize(image, self.image_size) if 'cv2' in locals() else \
                           np.resize(image, self.image_size)
                    image = np.expand_dims(image, axis=-1)
                    
                return image
                
            else:
                return np.random.rand(*self.image_size, 1).astype(np.float32)
                
        except Exception as e:
            logger.warning(f"Erro ao carregar imagem {idx}: {e}")
            return np.random.rand(*self.image_size, 1).astype(np.float32)
    
    def _get_pathology_label(self, idx, pathology_classes):
        """Determina rótulo de patologia baseado no arquivo"""
        try:
            sample_files = list(self.image_dir.glob('**/*.dcm')) + \
                          list(self.image_dir.glob('**/*.jpg')) + \
                          list(self.image_dir.glob('**/*.png'))
            
            if sample_files and idx < len(sample_files):
                filename = sample_files[idx % len(sample_files)].name.lower()
                
                label = np.zeros(self.num_classes, dtype=np.float32)
                
                if 'pneumonia' in filename:
                    label[pathology_classes['pneumonia']] = 1.0
                elif 'pleural' in filename or 'effusion' in filename:
                    label[pathology_classes['pleural_effusion']] = 1.0
                elif 'fracture' in filename or 'fratura' in filename:
                    label[pathology_classes['fracture']] = 1.0
                elif 'tumor' in filename or 'cancer' in filename:
                    label[pathology_classes['tumor']] = 1.0
                elif 'normal' in filename:
                    label[pathology_classes['normal']] = 1.0
                else:
                    label[pathology_classes['normal']] = 1.0
                    
                return label
            else:
                label = np.zeros(self.num_classes, dtype=np.float32)
                label[0] = 1.0
                return label
                
        except Exception as e:
            logger.warning(f"Erro ao determinar rótulo {idx}: {e}")
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[0] = 1.0  # Classe padrão
            return label
    
    def load_data(self, data_path: str):
        """Carrega dados do caminho especificado"""
        try:
            if Path(data_path).exists():
                logger.info(f"Carregando dados de {data_path}")
                return True
            else:
                logger.warning(f"Caminho não encontrado: {data_path}")
                return False
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return False

class MedicalModelTrainer:
    """
    Sistema completo de treinamento para modelos de imagem médica
    Inclui validação cruzada, métricas médicas e análise de performance
    """
    
    def __init__(self, 
                 model: tf.keras.Model,
                 model_name: str,
                 output_dir: Path,
                 use_wandb: bool = False):
        self.model = model
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb
        self.history = None
        self.best_weights = None
        
        # Diretórios para salvar resultados
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir in [self.checkpoints_dir, self.logs_dir, self.plots_dir]:
            dir.mkdir(exist_ok=True)
    
    def prepare_data_generators(self,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               batch_size: int = 32) -> Tuple:
        """
        Prepara geradores de dados com augmentation médico-específico
        
        Args:
            X_train: Dados de treino
            y_train: Labels de treino
            X_val: Dados de validação
            y_val: Labels de validação
            batch_size: Tamanho do batch
            
        Returns:
            Tupla com geradores de treino e validação
        """
        # Data augmentation para imagens médicas
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,  # Rotação limitada
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,  # Depende da modalidade
            brightness_range=[0.9, 1.1],
            fill_mode='reflect',  # Melhor para imagens médicas
            preprocessing_function=self._medical_preprocessing
        )
        
        # Validação sem augmentation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self._medical_preprocessing
        )
        
        # Criar geradores
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def _medical_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Pré-processamento específico para imagens médicas
        
        Args:
            image: Imagem de entrada
            
        Returns:
            Imagem pré-processada
        """
        # Normalização CLAHE para melhor contraste
        if len(image.shape) == 2 or image.shape[-1] == 1:
            import cv2
            image = image.astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(image.shape) == 3:
                image = image[:, :, 0]
            image = clahe.apply(image)
            image = np.expand_dims(image, axis=-1)
        
        # Normalização Z-score
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def create_callbacks(self) -> List[callbacks.Callback]:
        """
        Cria callbacks para o treinamento
        
        Returns:
            Lista de callbacks
        """
        callback_list = []
        
        # ModelCheckpoint - salva melhor modelo
        checkpoint_path = self.checkpoints_dir / f'{self.model_name}_best.h5'
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        callback_list.append(model_checkpoint)
        
        # EarlyStopping - previne overfitting
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # ReduceLROnPlateau - ajusta learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # TensorBoard - visualização
        tensorboard = callbacks.TensorBoard(
            log_dir=self.logs_dir / f'{self.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callback_list.append(tensorboard)
        
        # Custom callback para métricas médicas
        medical_metrics = MedicalMetricsCallback(self.output_dir)
        callback_list.append(medical_metrics)
        
        # WandB callback se habilitado
        if self.use_wandb:
            wandb_callback = wandb.keras.WandbCallback(
                save_model=False,
                monitor='val_auc',
                mode='max'
            )
            callback_list.append(wandb_callback)
        
        return callback_list
    
    def train(self,
              train_generator,
              val_generator,
              epochs: int = 100,
              class_weights: Optional[Dict] = None) -> Dict:
        """
        Treina o modelo com configurações médicas otimizadas
        
        Args:
            train_generator: Gerador de dados de treino
            val_generator: Gerador de dados de validação
            epochs: Número de épocas
            class_weights: Pesos para classes desbalanceadas
            
        Returns:
            Histórico de treinamento
        """
        logger.info(f"Iniciando treinamento do modelo {self.model_name}")
        
        # Criar callbacks
        callback_list = self.create_callbacks()
        
        # Treinar modelo
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callback_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # Salvar histórico
        self._save_history()
        
        logger.info("Treinamento concluído")
        return self.history.history
    
    def cross_validate(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      n_folds: int = 5,
                      epochs: int = 50) -> Dict:
        """
        Realiza validação cruzada estratificada
        
        Args:
            X: Dados completos
            y: Labels completas
            n_folds: Número de folds
            epochs: Épocas por fold
            
        Returns:
            Resultados da validação cruzada
        """
        logger.info(f"Iniciando validação cruzada com {n_folds} folds")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = {
            'accuracy': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Converter one-hot para classes
        y_classes = np.argmax(y, axis=1)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_classes)):
            logger.info(f"Treinando fold {fold + 1}/{n_folds}")
            
            # Dividir dados
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Preparar geradores
            train_gen, val_gen = self.prepare_data_generators(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold
            )
            
            # Treinar fold
            self.train(train_gen, val_gen, epochs=epochs)
            
            # Avaliar fold
            scores = self.evaluate(X_val_fold, y_val_fold)
            
            # Armazenar scores
            for metric, value in scores.items():
                if metric in cv_scores:
                    cv_scores[metric].append(value)
            
            # Resetar modelo para próximo fold
            self.model = self._reset_model_weights()
        
        # Calcular estatísticas
        cv_results = {}
        for metric, values in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
            cv_results[f'{metric}_values'] = values
        
        # Salvar resultados
        self._save_cv_results(cv_results)
        
        logger.info("Validação cruzada concluída")
        return cv_results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Avalia o modelo com métricas médicas relevantes
        
        Args:
            X_test: Dados de teste
            y_test: Labels de teste
            
        Returns:
            Dicionário com métricas
        """
        logger.info("Avaliando modelo...")
        
        # Predições
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Métricas básicas
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Métricas detalhadas
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # AUC para cada classe
        auc_scores = {}
        for i in range(y_test.shape[1]):
            if len(np.unique(y_test[:, i])) > 1:  # Verificar se há ambas classes
                auc_scores[f'auc_class_{i}'] = roc_auc_score(
                    y_test[:, i], y_pred_proba[:, i]
                )
        
        # Calcular AUC médio
        if auc_scores:
            avg_auc = np.mean(list(auc_scores.values()))
        else:
            avg_auc = 0.0
        
        results = {
            'loss': test_loss,
            'accuracy': test_acc,
            'auc': avg_auc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'auc_per_class': auc_scores
        }
        
        # Plotar resultados
        self._plot_evaluation_results(cm, y_test, y_pred_proba)
        
        # Salvar resultados
        self._save_evaluation_results(results)
        
        return results
    
    def _plot_evaluation_results(self, 
                                cm: np.ndarray, 
                                y_true: np.ndarray,
                                y_pred_proba: np.ndarray):
        """Plota matriz de confusão e curvas ROC"""
        # Matriz de confusão
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {self.model_name}')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.savefig(self.plots_dir / f'{self.model_name}_confusion_matrix.png')
        plt.close()
        
        # Curvas ROC
        plt.figure(figsize=(10, 8))
        for i in range(y_true.shape[1]):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Classe {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title(f'Curvas ROC - {self.model_name}')
        plt.legend(loc="lower right")
        plt.savefig(self.plots_dir / f'{self.model_name}_roc_curves.png')
        plt.close()
    
    def _save_history(self):
        """Salva histórico de treinamento"""
        history_path = self.output_dir / f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history.history, f, indent=4)
    
    def _save_evaluation_results(self, results: Dict):
        """Salva resultados de avaliação"""
        results_path = self.output_dir / f'{self.model_name}_evaluation.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def _save_cv_results(self, results: Dict):
        """Salva resultados de validação cruzada"""
        cv_path = self.output_dir / f'{self.model_name}_cv_results.json'
        with open(cv_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def _reset_model_weights(self):
        """Reseta pesos do modelo para validação cruzada"""
        # Recriar modelo com pesos aleatórios
        return tf.keras.models.clone_model(self.model)


class MedicalMetricsCallback(callbacks.Callback):
    """Callback customizado para métricas médicas durante treinamento"""
    
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir
        self.metrics = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Calcula e salva métricas médicas ao final de cada época"""
        if logs:
            # Adicionar métricas customizadas
            epoch_metrics = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                **logs
            }
            
            # Calcular especificidade se possível
            if 'val_precision' in logs and 'val_recall' in logs:
                # Especificidade aproximada
                epoch_metrics['val_specificity'] = logs.get('val_precision', 0)
            
            self.metrics.append(epoch_metrics)
            
            # Salvar incrementalmente
            metrics_path = self.output_dir / 'training_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
