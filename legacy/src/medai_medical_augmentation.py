"""
MedAI Medical-Specific Augmentation System
Implementa técnicas de augmentação realistas específicas para imagens médicas
Baseado nas melhorias do usuário para precisão diagnóstica
"""

import numpy as np
import cv2
import tensorflow as tf
from typing import Dict, Tuple, List, Optional
import logging
from scipy.ndimage import gaussian_filter, map_coordinates
import random

logger = logging.getLogger(__name__)

class MedicalAugmentation:
    """
    Augmentação específica para imagens médicas
    PROBLEMA ATUAL: Augmentação genérica não considera características radiológicas
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_medical_augmentations(self, modality: str = 'CR') -> List:
        """
        Retorna augmentações específicas por modalidade
        """
        base_augmentations = [
            self.realistic_rotation,
            self.add_medical_noise,
            self.simulate_breathing_motion
        ]
        
        if modality in ['CR', 'DX']:
            base_augmentations.append(self.elastic_deformation)
            
        return base_augmentations
    
    def realistic_rotation(self, image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
        """Rotação realista para imagens médicas (máximo 5 graus)"""
        angle = np.random.uniform(-max_angle, max_angle)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    
    def add_medical_noise(self, image: np.ndarray, noise_type: str = 'gaussian') -> np.ndarray:
        """
        Adiciona ruído médico realista
        """
        if noise_type == 'gaussian':
            noise_std = np.random.uniform(0.01, 0.05) * np.std(image)
            noise = np.random.normal(0, noise_std, image.shape)
            return np.clip(image + noise, 0, 1)
        
        elif noise_type == 'poisson':
            scaled = image * 255
            noisy = np.random.poisson(scaled) / 255.0
            return np.clip(noisy, 0, 1)
        
        return image
    
    def simulate_breathing_motion(self, image: np.ndarray, max_displacement: float = 2.0) -> np.ndarray:
        """
        Simula movimento respiratório em radiografias de tórax
        """
        dy = np.random.uniform(-max_displacement, max_displacement)
        dx = np.random.uniform(-max_displacement/2, max_displacement/2)
        
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        return cv2.warpAffine(image, M, (cols, rows))
    
    def elastic_deformation(self, img: np.ndarray, alpha: float = 1, sigma: float = 50, 
                          random_state=None) -> np.ndarray:
        """
        Deformação elástica realista para simular variações anatômicas
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
        
        shape = img.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        return map_coordinates(img, indices, order=1).reshape(shape)

class MedicalAugmentationTF:
    """
    Versão TensorFlow das augmentações médicas para integração com pipeline de treinamento
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @tf.function
    def realistic_rotation_tf(self, image: tf.Tensor, max_angle: float = 5.0) -> tf.Tensor:
        """Rotação realista usando TensorFlow"""
        angle = tf.random.uniform([], -max_angle, max_angle) * np.pi / 180
        return tf.keras.utils.image_utils.apply_affine_transform(
            image, theta=angle, fill_mode='nearest'
        )
    
    @tf.function
    def add_gaussian_noise_tf(self, image: tf.Tensor, noise_factor: float = 0.05) -> tf.Tensor:
        """Adiciona ruído gaussiano usando TensorFlow"""
        noise_std = tf.random.uniform([], 0.01, noise_factor) * tf.math.reduce_std(image)
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=noise_std)
        return tf.clip_by_value(image + noise, 0.0, 1.0)
    
    @tf.function
    def simulate_breathing_motion_tf(self, image: tf.Tensor, max_displacement: float = 2.0) -> tf.Tensor:
        """Simula movimento respiratório usando TensorFlow"""
        dy = tf.random.uniform([], -max_displacement, max_displacement)
        dx = tf.random.uniform([], -max_displacement/2, max_displacement/2)
        
        return tf.keras.utils.image_utils.apply_affine_transform(
            image, tx=dx, ty=dy, fill_mode='nearest'
        )
    
    def get_tf_augmentation_pipeline(self, modality: str = 'CR') -> tf.keras.Sequential:
        """
        Retorna pipeline de augmentação TensorFlow para treinamento
        """
        layers = []
        
        layers.append(tf.keras.layers.Lambda(
            lambda x: self.realistic_rotation_tf(x, max_angle=5.0)
        ))
        
        layers.append(tf.keras.layers.Lambda(
            lambda x: self.add_gaussian_noise_tf(x, noise_factor=0.05)
        ))
        
        if modality in ['CR', 'DX']:
            layers.append(tf.keras.layers.Lambda(
                lambda x: self.simulate_breathing_motion_tf(x, max_displacement=2.0)
            ))
        
        if modality == 'CT':
            layers.append(tf.keras.layers.RandomContrast(0.1))
        elif modality in ['CR', 'DX']:
            layers.append(tf.keras.layers.RandomContrast(0.2))
            layers.append(tf.keras.layers.RandomBrightness(0.1))
        
        return tf.keras.Sequential(layers)

class ModalitySpecificAugmentation:
    """
    Augmentações específicas por modalidade médica
    """
    
    def __init__(self):
        self.augmentation_configs = {
            'CR': {
                'rotation_range': 5.0,
                'noise_factor': 0.05,
                'breathing_motion': True,
                'contrast_range': 0.2,
                'brightness_range': 0.1
            },
            'DX': {
                'rotation_range': 3.0,
                'noise_factor': 0.03,
                'breathing_motion': True,
                'contrast_range': 0.15,
                'brightness_range': 0.08
            },
            'CT': {
                'rotation_range': 2.0,
                'noise_factor': 0.02,
                'breathing_motion': False,
                'contrast_range': 0.1,
                'brightness_range': 0.05
            },
            'MR': {
                'rotation_range': 3.0,
                'noise_factor': 0.04,
                'breathing_motion': False,
                'contrast_range': 0.15,
                'brightness_range': 0.1
            }
        }
    
    def get_augmentation_config(self, modality: str) -> Dict:
        """Retorna configuração de augmentação para modalidade específica"""
        return self.augmentation_configs.get(modality, self.augmentation_configs['CR'])
    
    def create_modality_pipeline(self, modality: str) -> MedicalAugmentationTF:
        """Cria pipeline de augmentação específico para modalidade"""
        config = self.get_augmentation_config(modality)
        augmenter = MedicalAugmentationTF()
        
        logger.info(f"Pipeline de augmentação criado para modalidade {modality}")
        logger.info(f"Configuração: {config}")
        
        return augmenter

def create_medical_augmentation_pipeline(modality: str = 'CR', 
                                       training: bool = True) -> tf.keras.Sequential:
    """
    Função utilitária para criar pipeline de augmentação médica
    
    Args:
        modality: Modalidade médica (CR, DX, CT, MR, etc.)
        training: Se True, aplica augmentações de treinamento
        
    Returns:
        Pipeline de augmentação TensorFlow
    """
    if not training:
        return tf.keras.Sequential([])
    
    modality_augmenter = ModalitySpecificAugmentation()
    tf_augmenter = modality_augmenter.create_modality_pipeline(modality)
    
    return tf_augmenter.get_tf_augmentation_pipeline(modality)

def integrate_with_existing_pipeline():
    """
    Integra augmentações médicas com pipeline existente
    """
    logger.info("Integrando augmentações médicas específicas com pipeline existente")
    logger.info("Substituindo augmentações genéricas por técnicas médicas realistas")
    
    return {
        'medical_augmentation': MedicalAugmentation(),
        'tf_augmentation': MedicalAugmentationTF(),
        'modality_specific': ModalitySpecificAugmentation()
    }
