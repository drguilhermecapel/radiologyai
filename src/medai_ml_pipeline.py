# ml_pipeline.py - Sistema de pipeline de machine learning e otimização

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.mixed_precision import Policy
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import optuna
from optuna.integration import TFKerasPruningCallback
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import mlflow
import mlflow.tensorflow
from datetime import datetime
import psutil
import GPUtil
import wandb
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback

logger = logging.getLogger('MedAI.MLPipeline')

@dataclass
class DatasetConfig:
    """Configuração do dataset"""
    name: str
    data_dir: Path
    image_size: Tuple[int, int]
    num_classes: int
    augmentation_config: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    validation_split: float = 0.2
    test_split: float = 0.1
    class_weights: Optional[Dict[int, float]] = None

@dataclass
class ModelConfig:
    """Configuração do modelo"""
    architecture: str
    input_shape: Tuple[int, int, int]
    num_classes: int
    pretrained: bool = True
    freeze_base: bool = True
    dropout_rate: float = 0.5
    regularization: float = 0.01
    activation: str = 'relu'
    optimizer_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Configuração de treinamento"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    
class MLPipeline:
    """
    Pipeline completo de Machine Learning para imagens médicas
    Inclui preparação de dados, treinamento, otimização e deployment
    """
    
    def __init__(self, 
                 project_name: str,
                 experiment_name: str,
                 config_path: Optional[str] = None):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = self._load_config(config_path)
        
        # Configurar MLflow
        mlflow.set_experiment(experiment_name)
        
        # Configurar mixed precision se disponível
        if self.config.get('training', {}).get('mixed_precision', False):
            policy = Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training habilitado")
        
        # Verificar recursos
        self._check_resources()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Carrega configuração do pipeline"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            # Configuração padrão
            return {
                'dataset': {
                    'name': 'chest_xray',
                    'data_dir': './data',
                    'image_size': [224, 224],
                    'num_classes': 5,
                    'augmentation': {
                        'rotation_range': 15,
                        'width_shift_range': 0.1,
                        'height_shift_range': 0.1,
                        'zoom_range': 0.1,
                        'horizontal_flip': True
                    }
                },
                'model': {
                    'architecture': 'efficientnet',
                    'pretrained': True,
                    'dropout_rate': 0.5
                },
                'training': {
                    'batch_size': 32,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'mixed_precision': True
                }
            }
    
    def _check_resources(self):
        """Verifica recursos disponíveis"""
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memória
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # GPU
        gpus = GPUtil.getGPUs()
        
        logger.info(f"Recursos disponíveis:")
        logger.info(f"  CPUs: {cpu_count} (uso: {cpu_percent}%)")
        logger.info(f"  Memória: {memory_gb:.1f} GB (disponível: {memory.available/(1024**3):.1f} GB)")
        
        if gpus:
            for gpu in gpus:
                logger.info(f"  GPU {gpu.id}: {gpu.name} ({gpu.memoryTotal} MB)")
        else:
            logger.warning("  Nenhuma GPU detectada")
    
    def prepare_data(self, 
                    data_dir: str,
                    config: DatasetConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepara datasets para treinamento
        
        Args:
            data_dir: Diretório com dados
            config: Configuração do dataset
            
        Returns:
            Tupla com datasets de treino, validação e teste
        """
        logger.info(f"Preparando dados de: {data_dir}")
        
        # Criar data pipeline eficiente com tf.data
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Função de parsing
        def parse_image(filepath, label):
            # Ler arquivo
            image = tf.io.read_file(filepath)
            
            # Decodificar baseado na extensão
            if tf.strings.regex_full_match(filepath, r".*\.dcm"):
                # DICOM precisa de processamento especial
                image = self._parse_dicom_tf(filepath)
            else:
                image = tf.image.decode_image(image, channels=1)
            
            # Redimensionar
            image = tf.image.resize(image, config.image_size)
            
            # Normalizar
            image = tf.cast(image, tf.float32) / 255.0
            
            # Aplicar pré-processamento médico
            image = self._medical_preprocessing_tf(image)
            
            return image, label
        
        # Função de augmentation
        def augment(image, label):
            # Rotação
            if config.augmentation_config.get('rotation_range', 0) > 0:
                angle = tf.random.uniform([], 
                    -config.augmentation_config['rotation_range'], 
                    config.augmentation_config['rotation_range']
                ) * np.pi / 180
                image = tfa.image.rotate(image, angle)
            
            # Translação
            if config.augmentation_config.get('width_shift_range', 0) > 0:
                dx = tf.random.uniform([], 
                    -config.augmentation_config['width_shift_range'], 
                    config.augmentation_config['width_shift_range']
                ) * tf.cast(tf.shape(image)[1], tf.float32)
                dy = tf.random.uniform([], 
                    -config.augmentation_config.get('height_shift_range', 0), 
                    config.augmentation_config.get('height_shift_range', 0)
                ) * tf.cast(tf.shape(image)[0], tf.float32)
                image = tfa.image.translate(image, [dx, dy])
            
            # Zoom
            if config.augmentation_config.get('zoom_range', 0) > 0:
                zoom_factor = tf.random.uniform([], 
                    1 - config.augmentation_config['zoom_range'],
                    1 + config.augmentation_config['zoom_range']
                )
                new_size = tf.cast(
                    tf.cast(tf.shape(image)[:2], tf.float32) * zoom_factor, 
                    tf.int32
                )
                image = tf.image.resize(image, new_size)
                image = tf.image.resize_with_crop_or_pad(
                    image, config.image_size[0], config.image_size[1]
                )
            
            # Flip horizontal
            if config.augmentation_config.get('horizontal_flip', False):
                image = tf.image.random_flip_left_right(image)
            
            # Ajustes de brilho/contraste
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            
            # Ruído gaussiano
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
            image = image + noise
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
        
        # Carregar lista de arquivos
        data_dir = Path(data_dir)
        all_files = []
        all_labels = []
        
        for class_idx, class_name in enumerate(config.class_names):
            class_dir = data_dir / class_name
            if class_dir.exists():
                files = list(class_dir.glob('*'))
                all_files.extend([str(f) for f in files])
                all_labels.extend([class_idx] * len(files))
        
        # Dividir dados
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_files, all_labels, 
            test_size=config.test_split, 
            stratify=all_labels,
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.validation_split / (1 - config.test_split),
            stratify=y_temp,
            random_state=42
        )
        
        # Criar datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.batch(config.batch_size)
        train_ds = train_ds.prefetch(AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.batch(config.batch_size)
        val_ds = val_ds.prefetch(AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.batch(config.batch_size)
        test_ds = test_ds.prefetch(AUTOTUNE)
        
        logger.info(f"Datasets criados - Treino: {len(X_train)}, Val: {len(X_val)}, Teste: {len(X_test)}")
        
        return train_ds, val_ds, test_ds
    
    def _medical_preprocessing_tf(self, image: tf.Tensor) -> tf.Tensor:
        """Pré-processamento específico para imagens médicas"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Implementação simplificada em TensorFlow
        
        # Equalização de histograma global
        image_uint8 = tf.cast(image * 255, tf.uint8)
        hist = tf.histogram_fixed_width(image_uint8, [0, 255], nbins=256)
        cdf = tf.cumsum(hist)
        cdf_normalized = cdf / tf.reduce_max(cdf)
        
        # Aplicar equalização
        image_equalized = tf.gather(cdf_normalized, tf.cast(image_uint8, tf.int32))
        
        # Normalização Z-score
        mean = tf.reduce_mean(image_equalized)
        std = tf.math.reduce_std(image_equalized)
        image_normalized = (image_equalized - mean) / (std + 1e-8)
        
        # Clip valores extremos
        image_clipped = tf.clip_by_value(image_normalized, -3, 3)
        
        # Re-normalizar para [0, 1]
        image_final = (image_clipped + 3) / 6
        
        return image_final
    
    def build_model(self, config: ModelConfig) -> tf.keras.Model:
        """
        Constrói modelo baseado na configuração
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Modelo compilado
        """
        logger.info(f"Construindo modelo: {config.architecture}")
        
        # Entrada
        inputs = layers.Input(shape=config.input_shape)
        
        # Pré-processamento
        x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        
        # Arquitetura base - Estado da arte
        if config.architecture == 'efficientnetv2':
            base_model = tf.keras.applications.EfficientNetV2L(
                input_shape=config.input_shape,
                include_top=False,
                weights='imagenet' if config.pretrained else None,
                pooling='avg'
            )
        elif config.architecture == 'convnext':
            base_model = tf.keras.applications.ConvNeXtXLarge(
                input_shape=config.input_shape,
                include_top=False,
                weights='imagenet' if config.pretrained else None,
                pooling='avg'
            )
        elif config.architecture == 'regnet':
            base_model = tf.keras.applications.RegNetY128GF(
                input_shape=config.input_shape,
                include_top=False,
                weights='imagenet' if config.pretrained else None,
                pooling='avg'
            )
        elif config.architecture == 'efficientnet':
            base_model = tf.keras.applications.EfficientNetB7(
                input_shape=config.input_shape,
                include_top=False,
                weights='imagenet' if config.pretrained else None,
                pooling='avg'
            )
        elif config.architecture == 'densenet':
            base_model = tf.keras.applications.DenseNet201(
                input_shape=config.input_shape,
                include_top=False,
                weights='imagenet' if config.pretrained else None,
                pooling='avg'
            )
        elif config.architecture == 'resnet':
            base_model = tf.keras.applications.ResNet152V2(
                input_shape=config.input_shape,
                include_top=False,
                weights='imagenet' if config.pretrained else None,
                pooling='avg'
            )
        elif config.architecture == 'custom':
            base_model = self._build_custom_architecture(config)
        else:
            raise ValueError(f"Arquitetura não suportada: {config.architecture}")
        
        # Congelar base se necessário
        if config.freeze_base and config.pretrained:
            base_model.trainable = False
        
        # Aplicar base model
        x = base_model(x, training=True)
        
        # Camadas adicionais
        x = layers.Dropout(config.dropout_rate)(x)
        x = layers.Dense(512, activation=config.activation,
                        kernel_regularizer=tf.keras.regularizers.l2(config.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate * 0.7)(x)
        x = layers.Dense(256, activation=config.activation,
                        kernel_regularizer=tf.keras.regularizers.l2(config.regularization))(x)
        x = layers.BatchNormalization()(x)
        
        # Camada de saída
        outputs = layers.Dense(config.num_classes, activation='softmax')(x)
        
        # Criar modelo
        model = models.Model(inputs, outputs, name=f'{config.architecture}_medical')
        
        # Compilar
        optimizer = self._get_optimizer(config.optimizer_config)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=config.optimizer_config.get('label_smoothing', 0.1)
            ),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tfa.metrics.F1Score(num_classes=config.num_classes, name='f1')
            ]
        )
        
        logger.info(f"Modelo construído com {model.count_params():,} parâmetros")
        
        return model
    
    def _build_custom_architecture(self, config: ModelConfig) -> tf.keras.Model:
        """Constrói arquitetura customizada"""
        inputs = layers.Input(shape=config.input_shape)
        
        # Stem
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Blocos residuais com atenção
        for i, filters in enumerate([64, 128, 256, 512]):
            # Downsample
            x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Bloco residual
            shortcut = x
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Atenção
            x = self._attention_block(x, filters)
            
            # Conexão residual
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            
            # Dropout progressivo
            x = layers.Dropout(config.dropout_rate * (i + 1) / 4)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        return models.Model(inputs, x)
    
    def _attention_block(self, x: tf.Tensor, filters: int) -> tf.Tensor:
        """Bloco de atenção"""
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        avg_pool = layers.Reshape((1, 1, filters))(avg_pool)
        max_pool = layers.Reshape((1, 1, filters))(max_pool)
        
        shared_dense1 = layers.Dense(filters // 8, activation='relu')
        shared_dense2 = layers.Dense(filters)
        
        avg_out = shared_dense2(shared_dense1(avg_pool))
        max_out = shared_dense2(shared_dense1(max_pool))
        
        channel_attention = layers.Activation('sigmoid')(avg_out + max_out)
        x = layers.Multiply()([x, channel_attention])
        
        # Spatial attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        
        concat = layers.Concatenate()([avg_pool, max_pool])
        spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
        
        x = layers.Multiply()([x, spatial_attention])
        
        return x
    
    def _get_optimizer(self, optimizer_config: Dict) -> tf.keras.optimizers.Optimizer:
        """Cria otimizador baseado na configuração"""
        optimizer_type = optimizer_config.get('type', 'adam')
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        
        # Learning rate schedule
        if optimizer_config.get('use_schedule', True):
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=learning_rate,
                first_decay_steps=optimizer_config.get('decay_steps', 1000),
                t_mul=2.0,
                m_mul=0.9
            )
        else:
            lr_schedule = learning_rate
        
        if optimizer_type == 'adam':
            optimizer = optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=optimizer_config.get('beta_1', 0.9),
                beta_2=optimizer_config.get('beta_2', 0.999),
                clipnorm=optimizer_config.get('gradient_clipping', 1.0)
            )
        elif optimizer_type == 'sgd':
            optimizer = optimizers.SGD(
                learning_rate=lr_schedule,
                momentum=optimizer_config.get('momentum', 0.9),
                nesterov=True,
                clipnorm=optimizer_config.get('gradient_clipping', 1.0)
            )
        elif optimizer_type == 'rmsprop':
            optimizer = optimizers.RMSprop(
                learning_rate=lr_schedule,
                rho=optimizer_config.get('rho', 0.9),
                clipnorm=optimizer_config.get('gradient_clipping', 1.0)
            )
        elif optimizer_type == 'adamw':
            optimizer = tfa.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                clipnorm=optimizer_config.get('gradient_clipping', 1.0)
            )
        elif optimizer_type == 'ranger':
            # RAdam + Lookahead
            radam = tfa.optimizers.RectifiedAdam(
                learning_rate=lr_schedule,
                beta_1=optimizer_config.get('beta_1', 0.9),
                beta_2=optimizer_config.get('beta_2', 0.999)
            )
            optimizer = tfa.optimizers.Lookahead(
                radam,
                sync_period=optimizer_config.get('sync_period', 6),
                slow_step_size=optimizer_config.get('slow_step_size', 0.5)
            )
        else:
            raise ValueError(f"Otimizador não suportado: {optimizer_type}")
        
        return optimizer
    
    def train(self,
             model: tf.keras.Model,
             train_ds: tf.data.Dataset,
             val_ds: tf.data.Dataset,
             config: TrainingConfig,
             experiment_name: Optional[str] = None) -> Dict:
        """
        Treina modelo com tracking completo
        
        Args:
            model: Modelo a treinar
            train_ds: Dataset de treino
            val_ds: Dataset de validação
            config: Configuração de treinamento
            experiment_name: Nome do experimento
            
        Returns:
            Histórico de treinamento
        """
        # Iniciar MLflow run
        with mlflow.start_run(run_name=experiment_name):
            # Log parâmetros
            mlflow.log_params({
                'batch_size': config.batch_size,
                'epochs': config.epochs,
                'learning_rate': config.learning_rate,
                'architecture': model.name,
                'parameters': model.count_params()
            })
            
            # Callbacks
            callbacks_list = self._create_callbacks(config)
            
            # Adicionar MLflow callback
            callbacks_list.append(
                MLflowCallback(log_every_n_steps=10)
            )
            
            # Treinar
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config.epochs,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Log modelo
            mlflow.tensorflow.log_model(
                model,
                "model",
                registered_model_name=f"{self.project_name}_model"
            )
            
            # Log métricas finais
            final_metrics = {
                'final_train_loss': history.history['loss'][-1],
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_val_acc': history.history['val_accuracy'][-1],
                'best_val_acc': max(history.history['val_accuracy']),
                'best_val_auc': max(history.history['val_auc'])
            }
            mlflow.log_metrics(final_metrics)
            
            return history.history
    
    def _create_callbacks(self, config: TrainingConfig) -> List[callbacks.Callback]:
        """Cria callbacks para treinamento"""
        callbacks_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Reduce LR on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            filepath='checkpoints/model_{epoch:02d}_{val_auc:.4f}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # TensorBoard
        tensorboard = callbacks.TensorBoard(
            log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1,
            profile_batch='10,20'
        )
        callbacks_list.append(tensorboard)
        
        # Custom metrics callback
        metrics_callback = CustomMetricsCallback()
        callbacks_list.append(metrics_callback)
        
        # Learning rate logger
        lr_logger = LearningRateLogger()
        callbacks_list.append(lr_logger)
        
        return callbacks_list
    
    def hyperparameter_optimization(self,
                                  dataset_config: DatasetConfig,
                                  n_trials: int = 100,
                                  optimization_metric: str = 'val_auc') -> Dict:
        """
        Otimização de hiperparâmetros com Optuna
        
        Args:
            dataset_config: Configuração do dataset
            n_trials: Número de trials
            optimization_metric: Métrica para otimizar
            
        Returns:
            Melhores hiperparâmetros
        """
        def objective(trial):
            # Sugerir hiperparâmetros
            model_config = ModelConfig(
                architecture=trial.suggest_categorical('architecture', 
                    ['efficientnet', 'densenet', 'resnet']),
                input_shape=(*dataset_config.image_size, 1),
                num_classes=dataset_config.num_classes,
                dropout_rate=trial.suggest_float('dropout_rate', 0.2, 0.7),
                regularization=trial.suggest_float('regularization', 1e-4, 1e-2, log=True),
                optimizer_config={
                    'type': trial.suggest_categorical('optimizer', 
                        ['adam', 'adamw', 'ranger']),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                }
            )
            
            training_config = TrainingConfig(
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
                learning_rate=model_config.optimizer_config['learning_rate'],
                label_smoothing=trial.suggest_float('label_smoothing', 0.0, 0.3)
            )
            
            # Preparar dados
            train_ds, val_ds, _ = self.prepare_data(
                dataset_config.data_dir,
                dataset_config
            )
            
            # Construir e treinar modelo
            model = self.build_model(model_config)
            
            # Callback para pruning
            pruning_callback = TFKerasPruningCallback(trial, optimization_metric)
            
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=20,  # Menos épocas para otimização
                callbacks=[pruning_callback],
                verbose=0
            )
            
            # Retornar métrica para otimizar
            return max(history.history[optimization_metric])
        
        # Criar estudo Optuna
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Otimizar
        study.optimize(objective, n_trials=n_trials)
        
        # Log melhores parâmetros
        logger.info(f"Melhores hiperparâmetros: {study.best_params}")
        logger.info(f"Melhor valor: {study.best_value}")
        
        # Salvar estudo
        study_path = f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        import joblib
        joblib.dump(study, study_path)
        
        return study.best_params
    
    def distributed_training(self,
                           model_fn: Callable,
                           dataset_fn: Callable,
                           config: Dict) -> None:
        """
        Treinamento distribuído com Ray Tune
        
        Args:
            model_fn: Função que retorna o modelo
            dataset_fn: Função que retorna os datasets
            config: Configuração para Ray Tune
        """
        def train_func(config):
            # Configurar estratégia distribuída
            strategy = tf.distribute.MirroredStrategy()
            
            with strategy.scope():
                # Criar modelo
                model = model_fn(config)
                
                # Preparar dados
                train_ds, val_ds = dataset_fn(config)
                
                # Treinar
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=config['epochs'],
                    callbacks=[
                        TuneReportCallback({
                            'loss': 'loss',
                            'accuracy': 'accuracy',
                            'val_loss': 'val_loss',
                            'val_accuracy': 'val_accuracy'
                        })
                    ]
                )
        
        # Configurar Ray Tune
        analysis = tune.run(
            train_func,
            config=config,
            num_samples=10,
            scheduler=ASHAScheduler(
                metric='val_accuracy',
                mode='max',
                max_t=100,
                grace_period=10
            ),
            resources_per_trial={
                'cpu': 2,
                'gpu': 1
            }
        )
        
        # Obter melhores resultados
        best_config = analysis.get_best_config(metric='val_accuracy', mode='max')
        logger.info(f"Melhor configuração: {best_config}")
    
    def deploy_model(self,
                    model_path: str,
                    deployment_type: str = 'tfserving',
                    optimization: str = 'none') -> Dict:
        """
        Prepara modelo para deployment
        
        Args:
            model_path: Caminho do modelo
            deployment_type: Tipo de deployment (tfserving, tflite, onnx)
            optimization: Tipo de otimização (none, quantization, pruning)
            
        Returns:
            Informações do deployment
        """
        # Carregar modelo
        model = tf.keras.models.load_model(model_path)
        
        deployment_info = {
            'original_size': self._get_model_size(model_path),
            'deployment_type': deployment_type,
            'optimization': optimization
        }
        
        if optimization == 'quantization':
            # Quantização para INT8
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            # Salvar modelo quantizado
            tflite_path = model_path.replace('.h5', '_quantized.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            deployment_info['quantized_size'] = len(tflite_model)
            deployment_info['size_reduction'] = (
                1 - len(tflite_model) / deployment_info['original_size']
            ) * 100
            
        elif optimization == 'pruning':
            # Pruning
            import tensorflow_model_optimization as tfmot
            
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.10,
                    final_sparsity=0.80,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            model_pruned = prune_low_magnitude(model, **pruning_params)
            model_pruned.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Salvar modelo podado
            pruned_path = model_path.replace('.h5', '_pruned.h5')
            model_pruned.save(pruned_path)
            
            deployment_info['pruned_size'] = self._get_model_size(pruned_path)
            
        if deployment_type == 'tfserving':
            # Preparar para TensorFlow Serving
            export_path = f"./models/{int(datetime.now().timestamp())}"
            tf.saved_model.save(model, export_path)
            deployment_info['export_path'] = export_path
            
        elif deployment_type == 'onnx':
            # Converter para ONNX
            import tf2onnx
            
            spec = (tf.TensorSpec(model.input_shape, tf.float32, name='input'),)
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
            
            onnx_path = model_path.replace('.h5', '.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(model_proto.SerializeToString())
            
            deployment_info['onnx_path'] = onnx_path
        
        return deployment_info
    
    def _get_model_size(self, model_path: str) -> int:
        """Obtém tamanho do modelo em bytes"""
        return Path(model_path).stat().st_size
    
    def _representative_dataset_gen(self):
        """Gerador de dataset representativo para quantização"""
        # Implementar com subset dos dados de treino
        pass


class CustomMetricsCallback(callbacks.Callback):
    """Callback para métricas customizadas"""
    
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Calcular métricas adicionais
            if 'val_precision' in logs and 'val_recall' in logs:
                # F1 Score manual
                precision = logs['val_precision']
                recall = logs['val_recall']
                f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
                logs['val_f1'] = f1
                
                # Specificity (aproximado)
                logs['val_specificity'] = logs.get('val_precision', 0)
                
            # Log para MLflow
            if mlflow.active_run():
                mlflow.log_metrics(logs, step=epoch)


class LearningRateLogger(callbacks.Callback):
    """Callback para logar learning rate"""
    
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)
        else:
            lr = tf.keras.backend.get_value(lr)
        
        if logs:
            logs['learning_rate'] = lr
            
        if mlflow.active_run():
            mlflow.log_metric('learning_rate', lr, step=epoch)


class MLflowCallback(callbacks.Callback):
    """Callback para integração com MLflow"""
    
    def __init__(self, log_every_n_steps: int = 1):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._step = 0
    
    def on_batch_end(self, batch, logs=None):
        if logs and self._step % self.log_every_n_steps == 0:
            mlflow.log_metrics(
                {f"batch_{k}": v for k, v in logs.items()},
                step=self._step
            )
        self._step += 1
    
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Log métricas de época
            mlflow.log_metrics(logs, step=epoch)
            
            # Log histogramas de pesos
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.kernel.numpy()
                    mlflow.log_metric(
                        f"{layer.name}_weight_mean",
                        np.mean(weights),
                        step=epoch
                    )
                    mlflow.log_metric(
                        f"{layer.name}_weight_std",
                        np.std(weights),
                        step=epoch
                    )
