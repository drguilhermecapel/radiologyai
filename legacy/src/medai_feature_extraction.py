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

class MedicalFeatureExtractor:
    """
    Extração de características médicas específicas
    Baseado nas melhorias do usuário para precisão diagnóstica
    PROBLEMA ATUAL: Features genéricas não capturam conhecimento médico
    """
    
    def __init__(self):
        self.feature_extractors = {
            'texture': self.extract_texture_features,
            'shape': self.extract_shape_features,
            'intensity': self.extract_intensity_features,
            'clinical': self.extract_clinical_features
        }
        self.logger = logging.getLogger(__name__)
    
    def extract_all_features(self, img: np.ndarray, metadata: dict) -> Dict[str, np.ndarray]:
        """
        Extrai todas as características médicas da imagem
        """
        features = {}
        
        try:
            features['texture'] = self.extract_texture_features(img)
            
            features['shape'] = self.extract_shape_features(img)
            
            features['intensity'] = self.extract_intensity_features(img)
            
            features['clinical'] = self.extract_clinical_features(img, metadata)
            
            self.logger.info(f"Características extraídas: {list(features.keys())}")
            
        except Exception as e:
            self.logger.error(f"Erro na extração de características: {e}")
            features = {key: np.array([]) for key in self.feature_extractors.keys()}
        
        return features
    
    def extract_texture_features(self, img: np.ndarray) -> np.ndarray:
        """
        Features de textura (Haralick, LBP, etc.)
        """
        features = []
        
        try:
            if img.dtype != np.uint8:
                img_8bit = (img * 255).astype(np.uint8)
            else:
                img_8bit = img
            
            try:
                from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
                
                distances = [1, 3, 5]
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                
                glcm = graycomatrix(img_8bit, distances=distances, angles=angles, 
                                   levels=256, symmetric=True, normed=True)
                
                for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                    features.extend(graycoprops(glcm, prop).flatten())
                
                lbp = local_binary_pattern(img_8bit, P=8, R=1, method='uniform')
                hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
                features.extend(hist / np.sum(hist))  # Normalizar histograma
                
            except ImportError:
                self.logger.warning("scikit-image não disponível, usando características básicas")
                features.extend([
                    float(np.std(img)),  # Contraste básico
                    float(np.var(img)),  # Variância
                    float(np.mean(np.abs(np.gradient(img)))),  # Gradiente médio
                ])
            
        except Exception as e:
            self.logger.warning(f"Erro na extração de características de textura: {e}")
            features = [0.0] * 50  # Características padrão em caso de erro
        
        return np.array(features)
    
    def extract_shape_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extrai características de forma e morfologia
        """
        features = []
        
        try:
            try:
                from skimage import filters, morphology
                threshold = filters.threshold_otsu(img)
                binary = img > threshold
                
                area = np.sum(binary)
                perimeter = np.sum(morphology.binary_erosion(binary) != binary)
                
            except ImportError:
                threshold = np.percentile(img, 50)
                binary = img > threshold
                area = np.sum(binary)
                perimeter = np.sum(np.abs(np.gradient(binary.astype(float))))
            
            if perimeter > 0:
                compactness = (perimeter ** 2) / (4 * np.pi * area)
            else:
                compactness = 0
            
            features.extend([area / img.size, perimeter / img.size, compactness])
            
            moments = cv2.moments(binary.astype(np.uint8))
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments)
            
            contours, _ = cv2.findContours(binary.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                contour_area = cv2.contourArea(largest_contour)
                contour_perimeter = cv2.arcLength(largest_contour, True)
                
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = contour_area / hull_area if hull_area > 0 else 0
                
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0
                
                features.extend([contour_area / img.size, contour_perimeter / img.size, 
                               solidity, aspect_ratio])
            else:
                features.extend([0, 0, 0, 0])
                
        except Exception as e:
            self.logger.warning(f"Erro na extração de características de forma: {e}")
            features = [0.0] * 15  # Características padrão em caso de erro
        
        return np.array(features)
    
    def extract_intensity_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extrai características de intensidade e distribuição
        """
        features = []
        
        try:
            features.extend([
                np.mean(img),
                np.std(img),
                np.min(img),
                np.max(img),
                np.median(img),
                np.percentile(img, 25),
                np.percentile(img, 75)
            ])
            
            try:
                from scipy.stats import skew, kurtosis
                features.extend([skew(img.flatten()), kurtosis(img.flatten())])
            except ImportError:
                # Implementação básica sem scipy
                mean = np.mean(img)
                std = np.std(img)
                if std > 0:
                    normalized = (img - mean) / std
                    skewness = np.mean(normalized ** 3)
                    kurt = np.mean(normalized ** 4) - 3
                else:
                    skewness = 0
                    kurt = 0
                features.extend([skewness, kurt])
            
            hist, _ = np.histogram(img, bins=32, range=(0, 1))
            hist_normalized = hist / np.sum(hist)
            features.extend(hist_normalized)
            
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
            features.append(entropy)
            
            grad_x = np.gradient(img, axis=1)
            grad_y = np.gradient(img, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude)
            ])
            
        except Exception as e:
            self.logger.warning(f"Erro na extração de características de intensidade: {e}")
            features = [0.0] * 45  # Características padrão em caso de erro
        
        return np.array(features)
    
    def extract_clinical_features(self, img: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Features clinicamente relevantes
        """
        features = []
        modality = metadata.get('modality', 'CR')
        
        try:
            if modality in ['CR', 'DX']:
                ctr = self.calculate_cardiothoracic_ratio(img)
                features.append(ctr)
            else:
                features.append(0.0)
            
            lung_density = self.calculate_lung_density(img)
            features.extend(lung_density)
            
            symmetry = self.calculate_symmetry(img)
            features.append(symmetry)
            
            if modality == 'CT':
                ct_features = self._extract_ct_specific_features(img, metadata)
                features.extend(ct_features)
            elif modality in ['CR', 'DX']:
                xray_features = self._extract_xray_specific_features(img)
                features.extend(xray_features)
            else:
                features.extend([0.0] * 5)
                
        except Exception as e:
            self.logger.warning(f"Erro na extração de características clínicas: {e}")
            features = [0.0] * 10  # Características padrão em caso de erro
        
        return np.array(features)
    
    def calculate_cardiothoracic_ratio(self, img: np.ndarray) -> float:
        """
        Calcula razão cardiotorácica aproximada
        """
        try:
            h, w = img.shape[:2]
            
            heart_region = img[h//3:2*h//3, w//3:2*w//3]
            
            threshold = np.percentile(img, 70)  # Estruturas mais densas
            cardiac_mask = heart_region > threshold
            
            cardiac_width = np.sum(np.any(cardiac_mask, axis=0))
            
            thoracic_width = w * 0.8  # Assumir que 80% da largura é tórax
            
            ctr = cardiac_width / thoracic_width if thoracic_width > 0 else 0
            
            return min(ctr, 1.0)  # Limitar a 1.0
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo da razão cardiotorácica: {e}")
            return 0.5
    
    def calculate_lung_density(self, img: np.ndarray) -> List[float]:
        """
        Calcula densidade pulmonar em diferentes regiões
        """
        try:
            h, w = img.shape[:2]
            
            left_lung = img[:, :w//2]
            right_lung = img[:, w//2:]
            
            upper_region = img[:h//2, :]
            lower_region = img[h//2:, :]
            
            densities = [
                np.mean(left_lung),
                np.mean(right_lung),
                np.mean(upper_region),
                np.mean(lower_region)
            ]
            
            return densities
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo da densidade pulmonar: {e}")
            return [0.5, 0.5, 0.5, 0.5]
    
    def calculate_symmetry(self, img: np.ndarray) -> float:
        """
        Calcula simetria da imagem
        """
        try:
            h, w = img.shape[:2]
            left_half = img[:, :w//2]
            right_half = img[:, w//2:]
            
            right_flipped = np.flip(right_half, axis=1)
            
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_resized = left_half[:, :min_width]
            right_resized = right_flipped[:, :min_width]
            
            correlation = np.corrcoef(left_resized.flatten(), right_resized.flatten())[0, 1]
            
            if np.isnan(correlation):
                return 0.5
            
            return np.clip(correlation, 0, 1)
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo da simetria: {e}")
            return 0.5
    
    def _extract_ct_specific_features(self, img: np.ndarray, metadata: dict) -> List[float]:
        """
        Características específicas para CT
        """
        features = []
        
        try:
            hu_stats = [
                np.percentile(img, 10),  # Ar/pulmão
                np.percentile(img, 50),  # Tecidos moles
                np.percentile(img, 90),  # Osso/contraste
            ]
            features.extend(hu_stats)
            
            slice_thickness = metadata.get('SliceThickness', 5.0)
            features.append(slice_thickness / 10.0)  # Normalizar
            
            features.append(np.std(img))  # Variação de densidade
            
        except Exception as e:
            self.logger.warning(f"Erro nas características específicas de CT: {e}")
            features = [0.0] * 5
        
        return features
    
    def _extract_xray_specific_features(self, img: np.ndarray) -> List[float]:
        """
        Características específicas para radiografias
        """
        features = []
        
        try:
            h, w = img.shape[:2]
            
            mediastinum = img[h//4:3*h//4, 2*w//5:3*w//5]
            features.append(np.mean(mediastinum))
            
            left_lung_region = img[h//4:3*h//4, w//10:2*w//5]
            right_lung_region = img[h//4:3*h//4, 3*w//5:9*w//10]
            
            features.extend([
                np.mean(left_lung_region),
                np.mean(right_lung_region),
                np.std(left_lung_region),
                np.std(right_lung_region)
            ])
            
        except Exception as e:
            self.logger.warning(f"Erro nas características específicas de raios-X: {e}")
            features = [0.0] * 5
        
        return features
    
    def combine_features(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combina todas as características em um vetor único
        """
        combined = []
        
        for feature_type in ['texture', 'shape', 'intensity', 'clinical']:
            if feature_type in feature_dict:
                combined.extend(feature_dict[feature_type])
            else:
                self.logger.warning(f"Características {feature_type} não encontradas")
        
        return np.array(combined)

AdvancedFeatureExtractor = RadiomicFeatureExtractor
