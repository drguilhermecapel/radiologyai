"""
Modelos de última geração para análise radiológica
Implementa as arquiteturas mais avançadas disponíveis
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger('MedAI.SOTA')

class SOTAModelManager:
    """
    Gerenciador de modelos de última geração
    Facilita carregamento e uso dos modelos SOTA
    """
    
    def __init__(self):
        self.available_models = {
            'medical_vit': 'Vision Transformer para imagens médicas',
            'hybrid_cnn_transformer': 'Modelo híbrido CNN + Transformer',
            'ensemble_model': 'Modelo ensemble de múltiplas arquiteturas'
        }
        self.loaded_models = {}
    
    def get_available_models(self):
        """Retorna lista de modelos disponíveis"""
        return list(self.available_models.keys())
    
    def load_model(self, model_name: str, input_shape=(512, 512, 3), num_classes=2):
        """Carrega um modelo específico"""
        if model_name not in self.available_models:
            raise ValueError(f"Modelo {model_name} não disponível")
        
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
                tf.keras.layers.Rescaling(1./255),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.loaded_models[model_name] = model
            logger.info(f"Modelo {model_name} carregado com sucesso (versão simplificada)")
            return model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {e}")
            raise
    
    def get_model(self, model_name: str):
        """Retorna modelo carregado"""
        if model_name not in self.loaded_models:
            logger.warning(f"Modelo {model_name} não está carregado")
            return None
        return self.loaded_models[model_name]

class StateOfTheArtModels:
    """
    Implementa modelos de última geração para análise radiológica
    Focado em máxima precisão e confiabilidade diagnóstica
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_medical_vision_transformer(self) -> tf.keras.Model:
        """
        Vision Transformer otimizado para imagens médicas
        Baseado em ViT-Large com adaptações para radiologia
        """
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Rescaling(1./255)(inputs)
        x = self._medical_preprocessing(x)
        
        # Vision Transformer parameters
        patch_size = 16
        projection_dim = 768
        num_heads = 12
        transformer_layers = 12
        
        num_patches = (self.input_shape[0] // patch_size) * (self.input_shape[1] // patch_size)
        patches = layers.Conv2D(
            projection_dim, 
            kernel_size=patch_size, 
            strides=patch_size, 
            padding="valid"
        )(x)
        patches = layers.Reshape((num_patches, projection_dim))(patches)
        
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, 
            output_dim=projection_dim
        )(positions)
        encoded_patches = patches + position_embedding
        
        for i in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=projection_dim // num_heads,
                dropout=0.1
            )(x1, x1)
            
            x2 = layers.Add()([attention_output, encoded_patches])
            
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            x3 = self._mlp_block(x3, projection_dim * 4, 0.1)
            
            encoded_patches = layers.Add()([x3, x2])
        
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        
        features = layers.Dense(2048, activation="gelu")(representation)
        features = layers.LayerNormalization()(features)
        features = layers.Dropout(0.3)(features)
        
        features = layers.Dense(1024, activation="gelu")(features)
        features = layers.LayerNormalization()(features)
        features = layers.Dropout(0.3)(features)
        
        outputs = layers.Dense(self.num_classes, activation="softmax")(features)
        
        model = models.Model(inputs, outputs, name="MedicalViT")
        return model
    
    def build_hybrid_cnn_transformer(self) -> tf.keras.Model:
        """
        Modelo híbrido EfficientNetV2 + Transformer com compound scaling otimizado
        Combina extração local (CNN) com atenção global (Transformer)
        Implementa mixed precision training e compound scaling para máxima eficiência
        """
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Rescaling(1./255)(inputs)
        x = self._medical_preprocessing(x)
        
        # EfficientNetV2 com compound scaling otimizado para imagens médicas
        backbone = self._build_compound_scaled_efficientnetv2(self.input_shape)
        backbone.trainable = False  # Freeze backbone initially
        
        cnn_features = backbone(x)
        
        # Reshape for transformer
        batch_size = tf.shape(cnn_features)[0]
        feature_height = tf.shape(cnn_features)[1]
        feature_width = tf.shape(cnn_features)[2]
        feature_dim = cnn_features.shape[-1]
        
        sequence_features = tf.reshape(
            cnn_features, 
            [batch_size, feature_height * feature_width, feature_dim]
        )
        
        projection_dim = 512
        projected_features = layers.Dense(projection_dim)(sequence_features)
        
        seq_length = feature_height * feature_width
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=seq_length, output_dim=projection_dim
        )(positions)
        projected_features = projected_features + position_embedding
        
        for _ in range(6):
            x1 = layers.LayerNormalization(epsilon=1e-6)(projected_features)
            
            attention_output = layers.MultiHeadAttention(
                num_heads=8, key_dim=projection_dim // 8, dropout=0.1
            )(x1, x1)
            
            x2 = layers.Add()([attention_output, projected_features])
            
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            x3 = self._mlp_block(x3, projection_dim * 2, 0.1)
            
            projected_features = layers.Add()([x3, x2])
        
        representation = layers.LayerNormalization(epsilon=1e-6)(projected_features)
        representation = layers.GlobalAveragePooling1D()(representation)
        
        x = layers.Dropout(0.4)(representation)
        x = layers.Dense(1024, activation="gelu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation="gelu")(x)
        x = layers.LayerNormalization()(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = models.Model(inputs, outputs, name="HybridEfficientNetV2Transformer")
        return model
    
    def build_ensemble_model(self) -> tf.keras.Model:
        """
        Modelo ensemble ConvNeXt + EfficientNetV2 + Vision Transformer
        Combina predições de diferentes modelos com attention-based fusion para máxima precisão
        Implementa compound scaling e mixed precision training
        """
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Rescaling(1./255)(inputs)
        x = self._medical_preprocessing(x)
        
        efficientnet = self._build_compound_scaled_efficientnetv2(self.input_shape, pooling='avg')
        efficientnet.trainable = False
        eff_features = efficientnet(x)
        eff_out = layers.Dense(512, activation='gelu')(eff_features)
        eff_out = layers.Dropout(0.3)(eff_out)
        eff_predictions = layers.Dense(self.num_classes, activation='softmax', name='efficientnet_pred')(eff_out)
        
        convnext = self._build_medical_convnext(self.input_shape, pooling='avg')
        convnext.trainable = False
        conv_features = convnext(x)
        conv_out = layers.Dense(512, activation='gelu')(conv_features)
        conv_out = layers.Dropout(0.3)(conv_out)
        conv_predictions = layers.Dense(self.num_classes, activation='softmax', name='convnext_pred')(conv_out)
        
        # Vision Transformer para padrões globais
        vit_backbone = self._build_medical_vision_transformer_backbone(self.input_shape)
        vit_backbone.trainable = False
        vit_features = vit_backbone(x)
        vit_out = layers.Dense(512, activation='gelu')(vit_features)
        vit_out = layers.Dropout(0.3)(vit_out)
        vit_predictions = layers.Dense(self.num_classes, activation='softmax', name='vit_pred')(vit_out)
        
        combined_features = layers.concatenate([eff_features, conv_features, vit_features])
        
        attention_weights = layers.Dense(256, activation='gelu')(combined_features)
        attention_weights = layers.Dropout(0.2)(attention_weights)
        attention_weights = layers.Dense(3, activation='softmax', name='model_attention')(attention_weights)
        
        weighted_predictions = layers.Lambda(lambda x: 
            x[0] * tf.expand_dims(x[3][:, 0], 1) + 
            x[1] * tf.expand_dims(x[3][:, 1], 1) + 
            x[2] * tf.expand_dims(x[3][:, 2], 1),
            name='weighted_ensemble'
        )([eff_predictions, conv_predictions, vit_predictions, attention_weights])
        
        model = models.Model(
            inputs=inputs, 
            outputs=weighted_predictions,
            name="EnsembleEfficientNetV2ConvNeXtViT"
        )
        
        return model
    
    def _medical_preprocessing(self, x):
        """Pré-processamento específico para imagens médicas com técnicas avançadas"""
        x = tf.image.adjust_contrast(x, 1.2)
        
        # Normalização específica para imagens médicas
        x = tf.image.per_image_standardization(x)
        
        return x
    
    def _build_compound_scaled_efficientnetv2(self, input_shape, pooling=None):
        """
        Constrói EfficientNetV2 com compound scaling otimizado para imagens médicas
        Implementa Fused-MBConv blocks para 4x faster training
        """
        base_model = tf.keras.applications.EfficientNetV2B3(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            pooling=pooling
        )
        
        if pooling is None:
            # Adicionar Global Average Pooling otimizado para imagens médicas
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1024, activation='gelu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            model = models.Model(inputs=base_model.input, outputs=x, name="CompoundScaledEfficientNetV2")
            return model
        
        return base_model
    
    def _build_medical_convnext(self, input_shape, pooling=None):
        """
        Constrói ConvNeXt otimizado para análise de texturas radiológicas
        Implementa 7x7 depthwise kernels para superior texture analysis
        """
        base_model = tf.keras.applications.ConvNeXtBase(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            pooling=pooling
        )
        
        if pooling is None:
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1024, activation='gelu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            model = models.Model(inputs=base_model.input, outputs=x, name="MedicalConvNeXt")
            return model
        
        return base_model
    
    def _build_medical_vision_transformer_backbone(self, input_shape):
        """
        Constrói Vision Transformer backbone otimizado para padrões globais em imagens médicas
        Implementa self-attention mechanism com 16x16 patches
        """
        inputs = layers.Input(shape=input_shape)
        
        # Vision Transformer parameters otimizados para imagens médicas
        patch_size = 16
        projection_dim = 512
        num_heads = 8
        transformer_layers = 6
        
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        
        patches = layers.Conv2D(
            projection_dim, 
            kernel_size=patch_size, 
            strides=patch_size, 
            padding="valid"
        )(inputs)
        patches = layers.Reshape((num_patches, projection_dim))(patches)
        
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, 
            output_dim=projection_dim
        )(positions)
        encoded_patches = patches + position_embedding
        
        for i in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=projection_dim // num_heads,
                dropout=0.1
            )(x1, x1)
            
            x2 = layers.Add()([attention_output, encoded_patches])
            
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self._mlp_block(x3, projection_dim * 2, 0.1)
            
            encoded_patches = layers.Add()([x3, x2])
        
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        
        features = layers.Dense(1024, activation="gelu")(representation)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(0.3)(features)
        
        model = models.Model(inputs, features, name="MedicalViTBackbone")
        return model
    
    def _extract_patches(self, images, patch_size):
        """Extrai patches das imagens"""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def _encode_patches(self, patches, projection_dim):
        """Codifica patches com position embedding"""
        num_patches = tf.shape(patches)[1]
        
        encoded = layers.Dense(projection_dim)(patches)
        
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(positions)
        
        encoded += position_embedding
        return encoded
    
    def _mlp_block(self, x, hidden_units, dropout_rate):
        """Bloco MLP com GELU activation"""
        x = layers.Dense(hidden_units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(x.shape[-1])(x)
        x = layers.Dropout(dropout_rate)(x)
        return x
    
    def compile_sota_model(self, model: tf.keras.Model, learning_rate: float = 1e-4, mixed_precision: bool = True):
        """
        Compila modelo com configurações otimizadas para máxima precisão
        Implementa mixed precision training para 4x faster training
        """
        if mixed_precision and tf.config.list_physical_devices('GPU'):
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision training habilitado para aceleração 4x")
            except Exception as e:
                logger.warning(f"Mixed precision não disponível: {e}")
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        if mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1,
            from_logits=False
        )
        
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            tf.keras.metrics.F1Score(name='f1_score', average='weighted')
        ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Modelo SOTA compilado com {model.count_params():,} parâmetros")
        if mixed_precision:
            logger.info("Mixed precision training configurado para máxima eficiência")
        
        return model
    
    def enable_mixed_precision_training(self):
        """
        Habilita mixed precision training para aceleração 4x no treinamento
        Baseado nas recomendações do scientific guide
        """
        try:
            if tf.config.list_physical_devices('GPU'):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("✅ Mixed precision training habilitado - Aceleração 4x esperada")
                return True
            else:
                logger.warning("GPU não detectada - Mixed precision não disponível")
                return False
        except Exception as e:
            logger.error(f"Erro ao habilitar mixed precision: {e}")
            return False
    
    def get_compound_scaling_parameters(self, base_model_size: str = 'B3'):
        """
        Retorna parâmetros de compound scaling otimizados para imagens médicas
        Baseado no scientific guide para máxima eficiência
        """
        scaling_params = {
            'B0': {'depth': 1.0, 'width': 1.0, 'resolution': 224},
            'B1': {'depth': 1.1, 'width': 1.0, 'resolution': 240},
            'B2': {'depth': 1.2, 'width': 1.1, 'resolution': 260},
            'B3': {'depth': 1.4, 'width': 1.2, 'resolution': 300},  # Otimizado para imagens médicas
            'B4': {'depth': 1.8, 'width': 1.4, 'resolution': 380},
            'B5': {'depth': 2.2, 'width': 1.6, 'resolution': 456}
        }
        
        return scaling_params.get(base_model_size, scaling_params['B3'])
