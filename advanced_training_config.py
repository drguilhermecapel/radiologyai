# advanced_training_config.py
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Configuração para treinamento eficiente
class TrainingConfig:
    # Hiperparâmetros otimizados para dataset médico
    BATCH_SIZE = 16  # Ajustar conforme GPU disponível
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    # Tamanhos de entrada por modelo
    IMAGE_SIZES = {
        'EfficientNetV2': (384, 384),
        'VisionTransformer': (384, 384),
        'ConvNeXt': (384, 384)
    }
    
    # Configurações de augmentação médica
    AUGMENTATION_CONFIG = {
        'rotation_range': 15,  # Rotação limitada para raios-X
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': False,  # Não inverter raios-X horizontalmente
        'zoom_range': 0.15,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'constant',
        'cval': 0  # Preto para preenchimento
    }
    
    # Mixed precision para acelerar treinamento
    @staticmethod
    def setup_mixed_precision():
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f' Mixed precision configurado: {policy.name}')