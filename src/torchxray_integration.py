#!/usr/bin/env python3
"""
TorchXRayVision Integration for MedAI Radiologia
Replaces dummy models with real pre-trained medical AI models
"""

import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
import time

logger = logging.getLogger(__name__)

class TorchXRayInference:
    """
    Real medical AI inference using TorchXRayVision pre-trained models
    Replaces the dummy model system with actual diagnostic capabilities
    """
    
    def __init__(self, model_name: str = 'densenet121-res224-all'):
        """
        Initialize TorchXRayVision model
        
        Args:
            model_name: Pre-trained model to use ('densenet121-res224-all', 'resnet50-res512-all', etc.)
        """
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pathologies = []
        self._load_model()
        
        self.pathology_mapping = {
            'Pneumonia': 'pneumonia',
            'Effusion': 'pleural_effusion', 
            'Fracture': 'fracture',
            'Mass': 'tumor',
            'Nodule': 'tumor',
            'Atelectasis': 'normal',
            'Consolidation': 'pneumonia',
            'Infiltration': 'pneumonia',
            'Pneumothorax': 'pneumonia',
            'Edema': 'normal',
            'Emphysema': 'normal',
            'Fibrosis': 'normal',
            'Pleural_Thickening': 'pleural_effusion',
            'Cardiomegaly': 'normal',
            'Hernia': 'normal',
            'Lung Lesion': 'tumor',
            'Lung Opacity': 'pneumonia',
            'Enlarged Cardiomediastinum': 'normal'
        }
        
        self.clinical_thresholds = {
            'pneumonia': 0.3,
            'pleural_effusion': 0.25,
            'fracture': 0.2,
            'tumor': 0.15,
            'normal': 0.5
        }
        
    def _load_model(self):
        """Load the TorchXRayVision pre-trained model"""
        try:
            logger.info(f"Loading TorchXRayVision model: {self.model_name}")
            
            if 'densenet121' in self.model_name:
                self.model = xrv.models.DenseNet(weights=self.model_name)
            elif 'resnet50' in self.model_name:
                self.model = xrv.models.ResNet(weights=self.model_name)
            else:
                self.model = xrv.models.DenseNet(weights='densenet121-res224-all')
                
            self.model.to(self.device)
            self.model.eval()
            self.pathologies = self.model.pathologies
            
            logger.info(f"Model loaded successfully. Can predict {len(self.pathologies)} pathologies")
            logger.info(f"Pathologies: {self.pathologies}")
            
        except Exception as e:
            logger.error(f"Error loading TorchXRayVision model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for TorchXRayVision model
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif len(image.shape) == 3 and image.shape[2] == 1:
                    image = image.squeeze()
                    
            if len(image.shape) != 2:
                raise ValueError(f"Expected 2D image, got shape {image.shape}")
                
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
            img = xrv.datasets.normalize(image, 255)
            
            img = img[None, ...]
            
            import torchvision.transforms
            transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])
            
            img = transform(img)
            
            image_tensor = torch.from_numpy(img).float()
            
            # Add batch dimension
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Perform medical diagnosis on chest X-ray image
        
        Args:
            image: Input chest X-ray image as numpy array
            
        Returns:
            Dictionary with diagnosis results
        """
        start_time = time.time()
        
        try:
            image_tensor = self.preprocess_image(image)
            
            with torch.no_grad():
                predictions = self.model(image_tensor)
                
            if isinstance(predictions, torch.Tensor):
                probs = torch.sigmoid(predictions).cpu().numpy()[0]
            else:
                probs = predictions[0]
                
            pathology_scores = {}
            for i, pathology in enumerate(self.pathologies):
                pathology_scores[pathology] = float(probs[i])
            
            category_scores = {}
            for pathology, score in pathology_scores.items():
                if pathology in self.pathology_mapping:
                    category = self.pathology_mapping[pathology]
                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append(score)
            
            final_scores = {}
            for category, scores in category_scores.items():
                final_scores[category] = max(scores)
            
            primary_diagnosis = self._determine_primary_diagnosis(final_scores)
            confidence = final_scores.get(primary_diagnosis, 0.0)
            
            findings = self._generate_clinical_findings(pathology_scores, final_scores)
            recommendations = self._generate_recommendations(primary_diagnosis, confidence)
            
            processing_time = time.time() - start_time
            
            result = {
                'primary_diagnosis': primary_diagnosis,
                'confidence': confidence,
                'pathology_scores': pathology_scores,
                'category_scores': final_scores,
                'clinical_findings': findings,
                'recommendations': recommendations,
                'processing_time': processing_time,
                'model_info': {
                    'model_name': self.model_name,
                    'pathologies_detected': len([p for p, s in pathology_scores.items() if s > 0.1]),
                    'total_pathologies': len(self.pathologies)
                }
            }
            
            logger.info(f"Diagnosis completed: {primary_diagnosis} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'primary_diagnosis': 'error',
                'confidence': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _determine_primary_diagnosis(self, category_scores: Dict[str, float]) -> str:
        """
        Determine primary diagnosis based on category scores and clinical thresholds
        """
        priority_order = ['fracture', 'tumor', 'pneumonia', 'pleural_effusion', 'normal']
        
        for category in priority_order:
            if category in category_scores:
                score = category_scores[category]
                threshold = self.clinical_thresholds.get(category, 0.2)
                
                if score >= threshold:
                    return category
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'normal'
    
    def _generate_clinical_findings(self, pathology_scores: Dict[str, float], category_scores: Dict[str, float]) -> List[str]:
        """Generate clinical findings based on pathology scores"""
        findings = []
        
        for pathology, score in pathology_scores.items():
            if score > 0.2:  # Significant finding threshold
                if pathology == 'Pneumonia':
                    findings.append(f"Possível pneumonia detectada (confiança: {score:.1%})")
                elif pathology == 'Effusion':
                    findings.append(f"Derrame pleural suspeito (confiança: {score:.1%})")
                elif pathology == 'Fracture':
                    findings.append(f"Fratura óssea identificada (confiança: {score:.1%})")
                elif pathology in ['Mass', 'Nodule']:
                    findings.append(f"Massa ou nódulo detectado (confiança: {score:.1%})")
                elif pathology == 'Pneumothorax':
                    findings.append(f"Pneumotórax identificado (confiança: {score:.1%})")
                elif pathology == 'Cardiomegaly':
                    findings.append(f"Cardiomegalia observada (confiança: {score:.1%})")
        
        if not findings:
            normal_score = category_scores.get('normal', 0.0)
            if normal_score > 0.4:
                findings.append("Radiografia dentro dos padrões de normalidade")
            else:
                findings.append("Achados inespecíficos - requer avaliação clínica")
        
        return findings
    
    def _generate_recommendations(self, diagnosis: str, confidence: float) -> List[str]:
        """Generate clinical recommendations based on diagnosis"""
        recommendations = []
        
        if diagnosis == 'pneumonia':
            if confidence > 0.6:
                recommendations.extend([
                    "Considerar tratamento antibiótico",
                    "Acompanhamento radiológico em 48-72h",
                    "Avaliação clínica urgente"
                ])
            else:
                recommendations.extend([
                    "Correlação clínica necessária",
                    "Considerar exames complementares",
                    "Acompanhamento médico"
                ])
                
        elif diagnosis == 'pleural_effusion':
            recommendations.extend([
                "Avaliação da causa do derrame pleural",
                "Considerar toracocentese se indicado",
                "Acompanhamento radiológico"
            ])
            
        elif diagnosis == 'fracture':
            recommendations.extend([
                "Avaliação ortopédica urgente",
                "Imobilização se necessário",
                "Controle da dor"
            ])
            
        elif diagnosis == 'tumor':
            recommendations.extend([
                "Investigação oncológica",
                "TC de tórax para melhor caracterização",
                "Encaminhamento para especialista"
            ])
            
        elif diagnosis == 'normal':
            if confidence > 0.7:
                recommendations.append("Acompanhamento de rotina")
            else:
                recommendations.extend([
                    "Correlação clínica recomendada",
                    "Considerar repetir exame se sintomas persistirem"
                ])
        
        else:
            recommendations.extend([
                "Avaliação médica especializada",
                "Correlação com quadro clínico",
                "Considerar exames complementares"
            ])
        
        return recommendations
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'pathologies': self.pathologies,
            'num_pathologies': len(self.pathologies),
            'clinical_categories': list(self.pathology_mapping.values()),
            'thresholds': self.clinical_thresholds
        }
