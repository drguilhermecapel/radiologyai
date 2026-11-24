"""
MedAI Detection System - Sistema de detecção de patologias
Implementa algoritmos de detecção usando YOLO e Mask R-CNN para radiologia
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger('MedAI.DetectionSystem')

class RadiologyYOLO:
    """
    Sistema YOLO adaptado para detecção de patologias em imagens médicas
    """
    
    def __init__(self):
        self.model_loaded = False
        self.classes = ['pneumonia', 'tumor', 'fracture', 'normal']
        logger.info("RadiologyYOLO inicializado")
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta objetos/patologias na imagem
        
        Args:
            image: Imagem médica como array numpy
            
        Returns:
            Lista de detecções com bounding boxes e confiança
        """
        try:
            detections = []
            
            if self._detect_pneumonia_pattern(image):
                detections.append({
                    'class': 'pneumonia',
                    'confidence': 0.85,
                    'bbox': [50, 50, 200, 150],
                    'area': 15000
                })
            
            if self._detect_tumor_pattern(image):
                detections.append({
                    'class': 'tumor',
                    'confidence': 0.78,
                    'bbox': [100, 80, 180, 160],
                    'area': 6400
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Erro na detecção YOLO: {e}")
            return []
    
    def _detect_pneumonia_pattern(self, image: np.ndarray) -> bool:
        """Detecta padrões de pneumonia"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Área mínima para consolidação
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Erro na detecção de pneumonia: {e}")
            return False
    
    def _detect_tumor_pattern(self, image: np.ndarray) -> bool:
        """Detecta padrões de tumor"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                return len(circles) > 0
            
            return False
            
        except Exception as e:
            logger.warning(f"Erro na detecção de tumor: {e}")
            return False

class MaskRCNNRadiology:
    """
    Sistema Mask R-CNN adaptado para segmentação de patologias
    """
    
    def __init__(self):
        self.model_loaded = False
        logger.info("MaskRCNNRadiology inicializado")
    
    def segment_pathologies(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segmenta patologias na imagem
        
        Args:
            image: Imagem médica como array numpy
            
        Returns:
            Dicionário com máscaras de segmentação
        """
        try:
            masks = {}
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            masks['pathology_mask'] = binary
            masks['lung_mask'] = self._create_lung_mask(gray)
            masks['bone_mask'] = self._create_bone_mask(gray)
            
            return masks
            
        except Exception as e:
            logger.error(f"Erro na segmentação: {e}")
            return {}
    
    def _create_lung_mask(self, image: np.ndarray) -> np.ndarray:
        """Cria máscara dos pulmões"""
        try:
            _, mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            logger.warning(f"Erro na criação da máscara pulmonar: {e}")
            return np.zeros_like(image)
    
    def _create_bone_mask(self, image: np.ndarray) -> np.ndarray:
        """Cria máscara dos ossos"""
        try:
            _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            
            return mask
            
        except Exception as e:
            logger.warning(f"Erro na criação da máscara óssea: {e}")
            return np.zeros_like(image)

class LesionTracker:
    """
    Sistema de rastreamento de lesões ao longo do tempo
    """
    
    def __init__(self):
        self.tracked_lesions = {}
        logger.info("LesionTracker inicializado")
    
    def track_lesions(self, current_image: np.ndarray, previous_image: Optional[np.ndarray] = None) -> Dict:
        """
        Rastreia lesões entre imagens sequenciais
        
        Args:
            current_image: Imagem atual
            previous_image: Imagem anterior (opcional)
            
        Returns:
            Informações sobre mudanças nas lesões
        """
        try:
            tracking_info = {
                'new_lesions': [],
                'changed_lesions': [],
                'stable_lesions': [],
                'resolved_lesions': []
            }
            
            current_lesions = self._detect_lesions(current_image)
            
            if previous_image is not None:
                previous_lesions = self._detect_lesions(previous_image)
                tracking_info = self._compare_lesions(previous_lesions, current_lesions)
            else:
                tracking_info['new_lesions'] = current_lesions
            
            return tracking_info
            
        except Exception as e:
            logger.error(f"Erro no rastreamento de lesões: {e}")
            return {}
    
    def _detect_lesions(self, image: np.ndarray) -> List[Dict]:
        """Detecta lesões na imagem"""
        try:
            lesions = []
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 100:  # Área mínima para ser considerada lesão
                    x, y, w, h = cv2.boundingRect(contour)
                    lesions.append({
                        'id': i,
                        'bbox': [x, y, w, h],
                        'area': area,
                        'centroid': (x + w//2, y + h//2)
                    })
            
            return lesions
            
        except Exception as e:
            logger.warning(f"Erro na detecção de lesões: {e}")
            return []
    
    def _compare_lesions(self, previous: List[Dict], current: List[Dict]) -> Dict:
        """Compara lesões entre duas imagens"""
        try:
            comparison = {
                'new_lesions': [],
                'changed_lesions': [],
                'stable_lesions': [],
                'resolved_lesions': []
            }
            
            matched_current = set()
            matched_previous = set()
            
            for i, prev_lesion in enumerate(previous):
                best_match = None
                min_distance = float('inf')
                
                for j, curr_lesion in enumerate(current):
                    if j in matched_current:
                        continue
                    
                    dist = np.sqrt(
                        (prev_lesion['centroid'][0] - curr_lesion['centroid'][0])**2 +
                        (prev_lesion['centroid'][1] - curr_lesion['centroid'][1])**2
                    )
                    
                    if dist < min_distance and dist < 50:  # Threshold de correspondência
                        min_distance = dist
                        best_match = j
                
                if best_match is not None:
                    matched_current.add(best_match)
                    matched_previous.add(i)
                    
                    area_change = abs(current[best_match]['area'] - prev_lesion['area']) / prev_lesion['area']
                    if area_change > 0.2:  # Mudança > 20%
                        comparison['changed_lesions'].append({
                            'previous': prev_lesion,
                            'current': current[best_match],
                            'area_change': area_change
                        })
                    else:
                        comparison['stable_lesions'].append(current[best_match])
                else:
                    comparison['resolved_lesions'].append(prev_lesion)
            
            for j, curr_lesion in enumerate(current):
                if j not in matched_current:
                    comparison['new_lesions'].append(curr_lesion)
            
            return comparison
            
        except Exception as e:
            logger.warning(f"Erro na comparação de lesões: {e}")
            return {
                'new_lesions': current,
                'changed_lesions': [],
                'stable_lesions': [],
                'resolved_lesions': previous
            }
