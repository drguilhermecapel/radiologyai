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
        
        sota_models = StateOfTheArtModels(input_shape, num_classes)
        
        if model_name == 'medical_vit':
            model = sota_models.build_medical_vision_transformer()
        elif model_name == 'hybrid_cnn_transformer':
            model = sota_models.build_hybrid_cnn_transformer()
        elif model_name == 'ensemble_model':
            model = sota_models.build_ensemble_model()
        
        model = sota_models.compile_sota_model(model)
        self.loaded_models[model_name] = model
        
        logger.info(f"Modelo {model_name} carregado com sucesso")
        return model
    
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
        
        x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        x = self._medical_preprocessing(x)
        
        patch_size = 16
        projection_dim = 1024
        num_heads = 16
        transformer_layers = 24
        mlp_head_units = [2048, 1024]
        
        patches = self._extract_patches(x, patch_size)
        encoded_patches = self._encode_patches(patches, projection_dim)
        
        for i in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=projection_dim // num_heads,
                dropout=0.1 if i < 12 else 0.05  # Menos dropout nas camadas finais
            )(x1, x1)
            
            x2 = layers.Add()([attention_output, encoded_patches])
            
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self._mlp_block(x3, projection_dim * 4, 0.1)
            
            encoded_patches = layers.Add()([x3, x2])
        
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        
        features = representation
        for units in mlp_head_units:
            features = layers.Dense(units, activation="gelu")(features)
            features = layers.LayerNormalization()(features)
            features = layers.Dropout(0.3)(features)
        
        outputs = layers.Dense(self.num_classes, activation="softmax")(features)
        
        model = models.Model(inputs, outputs, name="MedicalViT")
        return model
    
    def build_hybrid_cnn_transformer(self) -> tf.keras.Model:
        """
        Modelo híbrido CNN + Transformer
        Combina extração local (CNN) com atenção global (Transformer)
        """
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        x = self._medical_preprocessing(x)
        
        backbone = tf.keras.applications.EfficientNetV2L(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        cnn_features = backbone(x)
        
        batch_size = tf.shape(cnn_features)[0]
        feature_height = tf.shape(cnn_features)[1]
        feature_width = tf.shape(cnn_features)[2]
        feature_dim = cnn_features.shape[-1]
        
        sequence_features = tf.reshape(
            cnn_features, 
            [batch_size, feature_height * feature_width, feature_dim]
        )
        
        projection_dim = 768
        projected_features = layers.Dense(projection_dim)(sequence_features)
        
        seq_length = feature_height * feature_width
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=seq_length, output_dim=projection_dim
        )(positions)
        projected_features += position_embedding
        
        for _ in range(8):  # Menos camadas que ViT puro
            x1 = layers.LayerNormalization(epsilon=1e-6)(projected_features)
            attention_output = layers.MultiHeadAttention(
                num_heads=12, key_dim=projection_dim // 12, dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, projected_features])
            
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self._mlp_block(x3, projection_dim * 4, 0.1)
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
        
        model = models.Model(inputs, outputs, name="HybridCNNTransformer")
        return model
    
    def build_ensemble_model(self) -> tf.keras.Model:
        """
        Modelo ensemble de múltiplas arquiteturas
        Combina predições de diferentes modelos para máxima precisão
        """
        inputs = layers.Input(shape=self.input_shape)
        
        efficientnet = tf.keras.applications.EfficientNetV2L(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        eff_features = efficientnet(inputs)
        eff_out = layers.Dense(512, activation='gelu')(eff_features)
        eff_out = layers.Dropout(0.3)(eff_out)
        eff_predictions = layers.Dense(self.num_classes, activation='softmax', name='efficientnet_pred')(eff_out)
        
        convnext = tf.keras.applications.ConvNeXtXLarge(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        conv_features = convnext(inputs)
        conv_out = layers.Dense(512, activation='gelu')(conv_features)
        conv_out = layers.Dropout(0.3)(conv_out)
        conv_predictions = layers.Dense(self.num_classes, activation='softmax', name='convnext_pred')(conv_out)
        
        regnet = tf.keras.applications.RegNetY128GF(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        reg_features = regnet(inputs)
        reg_out = layers.Dense(512, activation='gelu')(reg_features)
        reg_out = layers.Dropout(0.3)(reg_out)
        reg_predictions = layers.Dense(self.num_classes, activation='softmax', name='regnet_pred')(reg_out)
        
        combined_features = layers.concatenate([eff_features, conv_features, reg_features])
        
        attention_weights = layers.Dense(3, activation='softmax', name='model_attention')(
            layers.Dense(256, activation='gelu')(combined_features)
        )
        
        weighted_predictions = layers.Lambda(lambda x: 
            x[0] * tf.expand_dims(x[3][:, 0], 1) + 
            x[1] * tf.expand_dims(x[3][:, 1], 1) + 
            x[2] * tf.expand_dims(x[3][:, 2], 1)
        )([eff_predictions, conv_predictions, reg_predictions, attention_weights])
        
        model = models.Model(
            inputs=inputs, 
            outputs=weighted_predictions,
            name="EnsembleModel"
        )
        
        return model
    
    def _medical_preprocessing(self, x):
        """Pré-processamento específico para imagens médicas"""
        x = tf.image.adjust_contrast(x, 1.2)
        
        x = tf.nn.depthwise_conv2d(
            x, 
            tf.constant([[[[0.1]], [[0.1]], [[0.1]]], 
                        [[[0.1]], [[0.8]], [[0.1]]], 
                        [[[0.1]], [[0.1]], [[0.1]]]], dtype=tf.float32),
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        
        return x
    
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
    
    def compile_sota_model(self, model: tf.keras.Model, learning_rate: float = 1e-4):
        """
        Compila modelo com configurações otimizadas para máxima precisão
        """
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
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
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Modelo SOTA compilado com {model.count_params():,} parâmetros")
        return model
