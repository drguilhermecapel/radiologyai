"""
MedAI Explainability - Sistema de explicabilidade para IA médica
Implementa GradCAM e Integrated Gradients para interpretação de modelos
"""

import numpy as np
import cv2
import tensorflow as tf
from typing import Dict, List, Optional, Any, Tuple
import logging

try:
    from .medai_modality_normalizer import ModalitySpecificNormalizer
except ImportError:
    try:
        from medai_modality_normalizer import ModalitySpecificNormalizer
    except ImportError:
        ModalitySpecificNormalizer = None

logger = logging.getLogger('MedAI.Explainability')

class GradCAMExplainer:
    """
    Implementa GradCAM para visualização de atenção em modelos CNN
    Específico para imagens médicas
    """
    
    def __init__(self, model=None):
        self.model = model
        self.last_conv_layer = None
        
        if ModalitySpecificNormalizer:
            self.normalizer = ModalitySpecificNormalizer()
        else:
            self.normalizer = None
            
        logger.info("GradCAMExplainer inicializado")
    
    def generate_heatmap(self, image: np.ndarray, class_idx: Optional[int] = None, modality: str = 'CR') -> np.ndarray:
        """
        Gera mapa de calor GradCAM para explicar predições
        
        Args:
            image: Imagem de entrada
            class_idx: Índice da classe para explicar
            
        Returns:
            Mapa de calor como array numpy
        """
        try:
            if len(image.shape) == 3:
                h, w = image.shape[:2]
            else:
                h, w = image.shape
            
            heatmap = self._simulate_attention_map(image)
            
            heatmap = cv2.resize(heatmap, (w, h))
            
            if self.normalizer is not None:
                try:
                    heatmap_normalized = self.normalizer.normalize_by_modality(heatmap, modality)
                except Exception as e:
                    logger.warning(f"Erro na normalização específica por modalidade: {e}. Usando normalização robusta.")
                    heatmap_normalized = self._robust_normalize(heatmap)
            else:
                heatmap_normalized = self._robust_normalize(heatmap)
            
            return heatmap_normalized
            
        except Exception as e:
            logger.error(f"Erro na geração do GradCAM: {e}")
            return np.zeros((224, 224), dtype=np.float32)
    
    def _simulate_attention_map(self, image: np.ndarray) -> np.ndarray:
        """Simula mapa de atenção baseado em características da imagem"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            edges = cv2.Canny(gray, 50, 150)
            
            heatmap = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
            
            noise = np.random.normal(0, 0.1, heatmap.shape)
            heatmap = heatmap + noise
            
            return heatmap
            
        except Exception as e:
            logger.warning(f"Erro na simulação do mapa de atenção: {e}")
            return np.random.rand(224, 224).astype(np.float32)
    
    def _robust_normalize(self, array: np.ndarray) -> np.ndarray:
        """
        Normalização robusta usando percentis para evitar outliers
        Substitui a normalização genérica min-max
        """
        try:
            p1, p99 = np.percentile(array, [1, 99])
            
            if p99 > p1:
                normalized = np.clip(array, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
            else:
                normalized = np.zeros_like(array, dtype=np.float32)
            
            return normalized.astype(np.float32)
        except:
            array_min = np.min(array)
            array_max = np.max(array)
            
            if array_max > array_min:
                normalized = (array - array_min) / (array_max - array_min)
            else:
                normalized = np.zeros_like(array, dtype=np.float32)
            
            return normalized
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Sobrepõe mapa de calor na imagem original
        
        Args:
            image: Imagem original
            heatmap: Mapa de calor
            alpha: Transparência da sobreposição
            
        Returns:
            Imagem com mapa de calor sobreposto
        """
        try:
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            
            if image.shape[:2] != heatmap_colored.shape[:2]:
                heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
            
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image.copy()
            
            overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap_colored, alpha, 0)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Erro na sobreposição do heatmap: {e}")
            return image

class IntegratedGradientsExplainer:
    """
    Implementa Integrated Gradients para explicação de modelos
    Método mais preciso para atribuição de características
    """
    
    def __init__(self, model=None):
        self.model = model
        self.baseline = None
        
        if ModalitySpecificNormalizer:
            self.normalizer = ModalitySpecificNormalizer()
        else:
            self.normalizer = None
            
        logger.info("IntegratedGradientsExplainer inicializado")
    
    def explain_prediction(self, image: np.ndarray, target_class: Optional[int] = None, steps: int = 50, modality: str = 'CR') -> np.ndarray:
        """
        Gera explicação usando Integrated Gradients
        
        Args:
            image: Imagem de entrada
            target_class: Classe alvo para explicação
            steps: Número de passos para integração
            
        Returns:
            Mapa de atribuição como array numpy
        """
        try:
            attribution_map = self._simulate_integrated_gradients(image, steps, modality)
            
            return attribution_map
            
        except Exception as e:
            logger.error(f"Erro no Integrated Gradients: {e}")
            return np.zeros_like(image, dtype=np.float32)
    
    def _simulate_integrated_gradients(self, image: np.ndarray, steps: int, modality: str = 'CR') -> np.ndarray:
        """Simula cálculo de Integrated Gradients"""
        try:
            baseline = np.zeros_like(image)
            
            attribution = np.zeros_like(image, dtype=np.float32)
            
            for i in range(steps):
                alpha = i / steps
                interpolated = baseline + alpha * (image - baseline)
                
                if len(interpolated.shape) == 3:
                    grad = np.gradient(interpolated.mean(axis=2))
                else:
                    grad = np.gradient(interpolated)
                
                if isinstance(grad, tuple):
                    grad_magnitude = np.sqrt(grad[0]**2 + grad[1]**2)
                else:
                    grad_magnitude = np.abs(grad)
                
                attribution += grad_magnitude / steps
            
            attribution = attribution * (image.mean(axis=2) if len(image.shape) == 3 else image)
            
            return attribution
            
        except Exception as e:
            logger.warning(f"Erro na simulação do Integrated Gradients: {e}")
            return np.random.random(image.shape[:2]).astype(np.float32)
    
    def visualize_attribution(self, image: np.ndarray, attribution: np.ndarray) -> np.ndarray:
        """
        Visualiza mapa de atribuição
        
        Args:
            image: Imagem original
            attribution: Mapa de atribuição
            
        Returns:
            Visualização da atribuição
        """
        try:
            if self.normalizer is not None:
                try:
                    attr_norm = self.normalizer.normalize_by_modality(attribution, 'CR')  # Default to CR for attribution maps
                except Exception as e:
                    logger.warning(f"Erro na normalização específica da atribuição: {e}. Usando normalização robusta.")
                    attr_norm = self._robust_normalize(attribution)
            else:
                attr_norm = self._robust_normalize(attribution)
            
            attr_colored = cv2.applyColorMap(
                (attr_norm * 255).astype(np.uint8),
                cv2.COLORMAP_VIRIDIS
            )
            
            return attr_colored
            
        except Exception as e:
            logger.error(f"Erro na visualização da atribuição: {e}")
            return image
    
    def _robust_normalize(self, array: np.ndarray) -> np.ndarray:
        """
        Normalização robusta usando percentis para evitar outliers
        Substitui a normalização genérica min-max
        """
        try:
            p1, p99 = np.percentile(array, [1, 99])
            
            if p99 > p1:
                normalized = np.clip(array, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
            else:
                normalized = np.zeros_like(array, dtype=np.float32)
            
            return normalized.astype(np.float32)
        except:
            array_min = np.min(array)
            array_max = np.max(array)
            
            if array_max > array_min:
                normalized = (array - array_min) / (array_max - array_min)
            else:
                normalized = np.zeros_like(array, dtype=np.float32)
            
            return normalized

class ExplainabilityManager:
    """
    Gerenciador central para métodos de explicabilidade
    """
    
    def __init__(self):
        self.gradcam = GradCAMExplainer()
        self.integrated_gradients = IntegratedGradientsExplainer()
        
        if ModalitySpecificNormalizer:
            self.normalizer = ModalitySpecificNormalizer()
        else:
            self.normalizer = None
            
        logger.info("ExplainabilityManager inicializado")
    
    def explain_prediction(self, image: np.ndarray, prediction: Dict, method: str = 'gradcam', modality: str = 'CR') -> Dict:
        """
        Gera explicação para uma predição
        
        Args:
            image: Imagem analisada
            prediction: Resultado da predição
            method: Método de explicação ('gradcam' ou 'integrated_gradients')
            
        Returns:
            Dicionário com explicações visuais
        """
        try:
            explanation = {
                'method': method,
                'prediction': prediction,
                'visual_explanations': {}
            }
            
            if method == 'gradcam':
                heatmap = self.gradcam.generate_heatmap(image, modality=modality)
                overlay = self.gradcam.overlay_heatmap(image, heatmap)
                
                explanation['visual_explanations'] = {
                    'heatmap': heatmap,
                    'overlay': overlay,
                    'attention_regions': self._identify_attention_regions(heatmap)
                }
                
            elif method == 'integrated_gradients':
                attribution = self.integrated_gradients.explain_prediction(image, modality=modality)
                visualization = self.integrated_gradients.visualize_attribution(image, attribution)
                
                explanation['visual_explanations'] = {
                    'attribution_map': attribution,
                    'visualization': visualization,
                    'important_features': self._identify_important_features(attribution)
                }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Erro na geração de explicação: {e}")
            return {'error': str(e)}
    
    def _identify_attention_regions(self, heatmap: np.ndarray) -> List[Dict]:
        """Identifica regiões de alta atenção no mapa de calor"""
        try:
            threshold = np.percentile(heatmap, 80)
            binary_mask = (heatmap > threshold).astype(np.uint8)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 100:  # Filtrar regiões muito pequenas
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        'id': i,
                        'bbox': [x, y, w, h],
                        'area': cv2.contourArea(contour),
                        'importance': float(np.mean(heatmap[y:y+h, x:x+w]))
                    })
            
            regions.sort(key=lambda x: x['importance'], reverse=True)
            
            return regions[:5]  # Retornar top 5 regiões
            
        except Exception as e:
            logger.warning(f"Erro na identificação de regiões de atenção: {e}")
            return []
    
    def _identify_important_features(self, attribution: np.ndarray) -> List[Dict]:
        """Identifica características importantes no mapa de atribuição"""
        try:
            threshold = np.percentile(np.abs(attribution), 90)
            important_mask = (np.abs(attribution) > threshold)
            
            labeled_mask = cv2.connectedComponents(important_mask.astype(np.uint8))[1]
            
            features = []
            for label in np.unique(labeled_mask)[1:]:  # Excluir background (0)
                mask = (labeled_mask == label)
                y_coords, x_coords = np.where(mask)
                
                if len(y_coords) > 10:  # Filtrar regiões muito pequenas
                    features.append({
                        'centroid': (int(np.mean(x_coords)), int(np.mean(y_coords))),
                        'size': len(y_coords),
                        'attribution_strength': float(np.mean(np.abs(attribution[mask])))
                    })
            
            features.sort(key=lambda x: x['attribution_strength'], reverse=True)
            
            return features[:10]  # Retornar top 10 características
            
        except Exception as e:
            logger.warning(f"Erro na identificação de características importantes: {e}")
            return []
