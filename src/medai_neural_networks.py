# neural_networks.py - Arquiteturas de redes neurais para análise médica

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import (
    ResNet50, DenseNet121, EfficientNetB4, InceptionV3
)
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger('MedAI.Networks')

class MedicalImageNetwork:
    """
    Classe base para redes neurais de análise de imagens médicas
    Implementa arquiteturas otimizadas para diferentes modalidades
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 num_classes: int,
                 model_name: str = 'densenet'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        
    def build_model(self) -> tf.keras.Model:
        """Constrói o modelo baseado na arquitetura escolhida"""
        if self.model_name == 'densenet':
            self.model = self._build_densenet()
        elif self.model_name == 'resnet':
            self.model = self._build_resnet()
        elif self.model_name == 'efficientnet':
            self.model = self._build_efficientnet()
        elif self.model_name == 'custom_cnn':
            self.model = self._build_custom_cnn()
        elif self.model_name == 'attention_unet':
            self.model = self._build_attention_unet()
        else:
            raise ValueError(f"Modelo {self.model_name} não reconhecido")
        
        logger.info(f"Modelo {self.model_name} construído com sucesso")
        return self.model
    
    def _build_densenet(self) -> tf.keras.Model:
        """
        DenseNet121 adaptada para imagens médicas
        Excelente para classificação de patologias
        """
        # Base model pré-treinada
        base_model = DenseNet121(
            input_shape=self.input_shape,
            weights='imagenet' if self.input_shape[-1] == 3 else None,
            include_top=False,
            pooling='avg'
        )
        
        # Congelar primeiras camadas
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Construir modelo completo
        inputs = layers.Input(shape=self.input_shape)
        
        # Pré-processamento
        x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        x = layers.GaussianNoise(0.1)(x)
        
        # Base model
        x = base_model(x, training=True)
        
        # Camadas adicionais com regularização
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        
        # Camada de saída
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _build_resnet(self) -> tf.keras.Model:
        """
        ResNet50 adaptada para imagens médicas
        Boa para features complexas
        """
        base_model = ResNet50(
            input_shape=self.input_shape,
            weights='imagenet' if self.input_shape[-1] == 3 else None,
            include_top=False,
            pooling='avg'
        )
        
        # Fine-tuning
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        inputs = layers.Input(shape=self.input_shape)
        x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        
        # Augmentation layers
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _build_efficientnet(self) -> tf.keras.Model:
        """
        EfficientNetB4 - Eficiente e precisa
        """
        base_model = EfficientNetB4(
            input_shape=self.input_shape,
            weights='imagenet' if self.input_shape[-1] == 3 else None,
            include_top=False,
            pooling='avg'
        )
        
        # Descongelar últimas camadas
        for layer in base_model.layers[-30:]:
            layer.trainable = True
        
        inputs = layers.Input(shape=self.input_shape)
        x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        x = base_model(x, training=True)
        
        # Multi-scale feature extraction
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _build_custom_cnn(self) -> tf.keras.Model:
        """
        CNN customizada para casos específicos
        Arquitetura otimizada para imagens médicas
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Bloco 1
        x = layers.Conv2D(32, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Bloco 2
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Bloco 3 com conexão residual
        shortcut = x
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        # Adaptar shortcut
        shortcut = layers.Conv2D(128, 1, padding='same')(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Bloco 4
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Global pooling e classificação
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _build_attention_unet(self) -> tf.keras.Model:
        """
        U-Net com mecanismo de atenção
        Para segmentação de estruturas anatômicas
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        # Bloco 1
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(2)(conv1)
        
        # Bloco 2
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(2)(conv2)
        
        # Bloco 3
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(2)(conv3)
        
        # Bloco 4
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(2)(conv4)
        
        # Bridge
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        
        # Decoder com Attention Gates
        # Bloco 6
        up6 = layers.UpSampling2D(2)(conv5)
        up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(up6)
        
        # Attention Gate 1
        att6 = self._attention_gate(conv4, up6, 512)
        merge6 = layers.concatenate([att6, up6], axis=3)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
        
        # Bloco 7
        up7 = layers.UpSampling2D(2)(conv6)
        up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(up7)
        
        # Attention Gate 2
        att7 = self._attention_gate(conv3, up7, 256)
        merge7 = layers.concatenate([att7, up7], axis=3)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
        
        # Bloco 8
        up8 = layers.UpSampling2D(2)(conv7)
        up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(up8)
        
        # Attention Gate 3
        att8 = self._attention_gate(conv2, up8, 128)
        merge8 = layers.concatenate([att8, up8], axis=3)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
        
        # Bloco 9
        up9 = layers.UpSampling2D(2)(conv8)
        up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(up9)
        
        # Attention Gate 4
        att9 = self._attention_gate(conv1, up9, 64)
        merge9 = layers.concatenate([att9, up9], axis=3)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
        
        # Output
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(conv9)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _attention_gate(self, x, g, inter_channels):
        """
        Implementa attention gate para U-Net
        
        Args:
            x: Feature map do encoder
            g: Feature map do decoder
            inter_channels: Número de canais intermediários
        """
        # Transformar x
        theta_x = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(x)
        
        # Transformar g
        phi_g = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(g)
        
        # Combinar
        f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
        
        # Atenção
        psi_f = layers.Conv2D(1, 1, strides=1, padding='same')(f)
        rate = layers.Activation('sigmoid')(psi_f)
        
        # Aplicar atenção
        att_x = layers.multiply([x, rate])
        
        return att_x
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compila o modelo com otimizador e métricas apropriadas"""
        if not self.model:
            raise ValueError("Modelo não construído. Execute build_model() primeiro.")
        
        # Otimizador com learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Métricas médicas relevantes
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        logger.info("Modelo compilado com sucesso")
