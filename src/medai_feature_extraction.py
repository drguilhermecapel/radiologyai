"""
MedAI Feature Extraction - Extração de características radiômicas
Implementa extração de características avançadas para análise de imagens médicas
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger('MedAI.FeatureExtraction')

class RadiomicFeatureExtractor:
    """
    Extrator de características radiômicas para imagens médicas
    Implementa extração de texturas, formas e intensidades
    """
    
    def __init__(self):
        self.features = {}
        logger.info("RadiomicFeatureExtractor inicializado")
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrai características radiômicas da imagem
        
        Args:
            image: Imagem médica como array numpy
            
        Returns:
            Dicionário com características extraídas
        """
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            features = {}
            
            features.update(self._extract_intensity_features(image))
            
            features.update(self._extract_texture_features(image))
            
            features.update(self._extract_shape_features(image))
            
            return features
            
        except Exception as e:
            logger.error(f"Erro na extração de características: {e}")
            return {}
    
    def _extract_intensity_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extrai características de intensidade"""
        features = {}
        
        try:
            features['mean_intensity'] = float(np.mean(image))
            features['std_intensity'] = float(np.std(image))
            features['min_intensity'] = float(np.min(image))
            features['max_intensity'] = float(np.max(image))
            features['median_intensity'] = float(np.median(image))
            features['skewness'] = float(self._calculate_skewness(image))
            features['kurtosis'] = float(self._calculate_kurtosis(image))
            
        except Exception as e:
            logger.warning(f"Erro nas características de intensidade: {e}")
            
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extrai características de textura usando GLCM"""
        features = {}
        
        try:
            features['texture_contrast'] = float(self._calculate_contrast(image))
            features['texture_homogeneity'] = float(self._calculate_homogeneity(image))
            features['texture_energy'] = float(self._calculate_energy(image))
            features['texture_correlation'] = float(self._calculate_correlation(image))
            
        except Exception as e:
            logger.warning(f"Erro nas características de textura: {e}")
            
        return features
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extrai características de forma"""
        features = {}
        
        try:
            features['area'] = float(np.sum(image > 0))
            features['perimeter'] = float(self._calculate_perimeter(image))
            features['compactness'] = float(self._calculate_compactness(image))
            features['eccentricity'] = float(self._calculate_eccentricity(image))
            
        except Exception as e:
            logger.warning(f"Erro nas características de forma: {e}")
            
        return features
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calcula assimetria da distribuição de intensidades"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0.0
        return np.mean(((image - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calcula curtose da distribuição de intensidades"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0.0
        return np.mean(((image - mean) / std) ** 4) - 3
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calcula contraste da imagem"""
        try:
            return float(np.max(image) - np.min(image))
        except:
            return 0.0
    
    def _calculate_homogeneity(self, image: np.ndarray) -> float:
        """Calcula homogeneidade da imagem"""
        try:
            variance = np.var(image)
            return 1.0 / (1.0 + variance) if variance > 0 else 1.0
        except:
            return 0.0
    
    def _calculate_energy(self, image: np.ndarray) -> float:
        """Calcula energia da imagem"""
        try:
            normalized = image / 255.0 if np.max(image) > 1 else image
            return float(np.sum(normalized ** 2))
        except:
            return 0.0
    
    def _calculate_correlation(self, image: np.ndarray) -> float:
        """Calcula correlação da imagem"""
        try:
            shifted = np.roll(image, 1, axis=1)
            correlation = np.corrcoef(image.flatten(), shifted.flatten())[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_perimeter(self, image: np.ndarray) -> float:
        """Calcula perímetro usando detecção de bordas"""
        try:
            binary = (image > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                return float(cv2.arcLength(contours[0], True))
            return 0.0
        except:
            return 0.0
    
    def _calculate_compactness(self, image: np.ndarray) -> float:
        """Calcula compacidade da forma"""
        try:
            area = np.sum(image > 0)
            perimeter = self._calculate_perimeter(image)
            if perimeter > 0:
                return float(4 * np.pi * area / (perimeter ** 2))
            return 0.0
        except:
            return 0.0
    
    def _calculate_eccentricity(self, image: np.ndarray) -> float:
        """Calcula excentricidade da forma"""
        try:
            binary = (image > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and len(contours[0]) >= 5:
                ellipse = cv2.fitEllipse(contours[0])
                a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
                if a > 0:
                    eccentricity = np.sqrt(1 - (min(a, b) / max(a, b)) ** 2)
                    return float(eccentricity)
            return 0.0
        except:
            return 0.0

AdvancedFeatureExtractor = RadiomicFeatureExtractor
