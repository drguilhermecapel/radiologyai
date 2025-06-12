"""
MedAI Dynamic Threshold Calibration System
Implements clinical performance-based threshold optimization
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from sklearn.calibration import CalibratedClassifierCV
import json

logger = logging.getLogger('MedAI.DynamicThresholds')

class DynamicThresholdCalibrator:
    """
    Sistema de calibração dinâmica de thresholds baseado em performance clínica
    Otimiza thresholds para maximizar sensibilidade e especificidade por patologia
    """
    
    def __init__(self):
        self.clinical_requirements = {
            'critical_conditions': {
                'pneumothorax': {'min_sensitivity': 0.95, 'min_specificity': 0.90},
                'massive_hemorrhage': {'min_sensitivity': 0.98, 'min_specificity': 0.85},
                'acute_stroke': {'min_sensitivity': 0.95, 'min_specificity': 0.88}
            },
            'moderate_conditions': {
                'pneumonia': {'min_sensitivity': 0.90, 'min_specificity': 0.85},
                'pleural_effusion': {'min_sensitivity': 0.88, 'min_specificity': 0.87},
                'fracture': {'min_sensitivity': 0.92, 'min_specificity': 0.90}
            },
            'standard_conditions': {
                'tumor': {'min_sensitivity': 0.85, 'min_specificity': 0.92},
                'normal': {'min_sensitivity': 0.80, 'min_specificity': 0.95}
            }
        }
        
        self.calibrated_thresholds = {}
    
    def calibrate_thresholds(self, 
                           y_true: np.ndarray, 
                           y_pred_proba: np.ndarray,
                           class_names: List[str],
                           validation_type: str = 'clinical') -> Dict[str, float]:
        """
        Calibra thresholds dinamicamente baseado em performance clínica
        
        Args:
            y_true: Labels verdadeiros
            y_pred_proba: Probabilidades preditas
            class_names: Nomes das classes
            validation_type: Tipo de validação ('clinical', 'balanced', 'sensitivity_focused')
            
        Returns:
            Dicionário com thresholds otimizados por classe
        """
        try:
            calibrated_thresholds = {}
            
            for i, class_name in enumerate(class_names):
                if len(y_pred_proba.shape) > 1:
                    class_proba = y_pred_proba[:, i]
                    class_true = (y_true == i).astype(int)
                else:
                    class_proba = y_pred_proba
                    class_true = y_true
                
                optimal_threshold = self._find_optimal_threshold(
                    class_true, class_proba, class_name, validation_type
                )
                
                calibrated_thresholds[class_name] = optimal_threshold
                
                logger.info(f"Threshold calibrado para {class_name}: {optimal_threshold:.3f}")
            
            self.calibrated_thresholds = calibrated_thresholds
            return calibrated_thresholds
            
        except Exception as e:
            logger.error(f"Erro na calibração de thresholds: {e}")
            return self._get_default_thresholds(class_names)
    
    def _find_optimal_threshold(self, 
                              y_true: np.ndarray, 
                              y_proba: np.ndarray,
                              class_name: str,
                              validation_type: str) -> float:
        """
        Encontra threshold ótimo para uma classe específica
        """
        try:
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
            
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
            
            clinical_req = self._get_clinical_requirements(class_name)
            min_sensitivity = clinical_req['min_sensitivity']
            min_specificity = clinical_req['min_specificity']
            
            if validation_type == 'clinical':
                optimal_threshold = self._optimize_clinical_threshold(
                    fpr, tpr, thresholds_roc, min_sensitivity, min_specificity
                )
            elif validation_type == 'balanced':
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = thresholds_roc[optimal_idx]
            elif validation_type == 'sensitivity_focused':
                valid_indices = tpr >= min_sensitivity
                if np.any(valid_indices):
                    specificity = 1 - fpr
                    valid_specificity = specificity[valid_indices]
                    best_idx = np.argmax(valid_specificity)
                    optimal_threshold = thresholds_roc[valid_indices][best_idx]
                else:
                    optimal_threshold = 0.5
            else:
                optimal_threshold = 0.5
            
            return float(np.clip(optimal_threshold, 0.1, 0.9))
            
        except Exception as e:
            logger.warning(f"Erro ao encontrar threshold ótimo para {class_name}: {e}")
            return 0.5
    
    def _optimize_clinical_threshold(self, 
                                   fpr: np.ndarray, 
                                   tpr: np.ndarray,
                                   thresholds: np.ndarray,
                                   min_sensitivity: float,
                                   min_specificity: float) -> float:
        """
        Otimiza threshold baseado em requisitos clínicos específicos
        """
        specificity = 1 - fpr
        
        valid_sensitivity = tpr >= min_sensitivity
        valid_specificity = specificity >= min_specificity
        valid_both = valid_sensitivity & valid_specificity
        
        if np.any(valid_both):
            valid_indices = np.where(valid_both)[0]
            best_idx = valid_indices[np.argmax(tpr[valid_indices] + specificity[valid_indices])]
            return thresholds[best_idx]
        elif np.any(valid_sensitivity):
            valid_indices = np.where(valid_sensitivity)[0]
            best_idx = valid_indices[np.argmax(specificity[valid_indices])]
            return thresholds[best_idx]
        else:
            best_idx = np.argmax(tpr)
            return thresholds[best_idx]
    
    def _get_clinical_requirements(self, class_name: str) -> Dict[str, float]:
        """
        Obtém requisitos clínicos para uma classe específica
        """
        for category in self.clinical_requirements.values():
            if class_name in category:
                return category[class_name]
        
        return {'min_sensitivity': 0.85, 'min_specificity': 0.85}
    
    def _get_default_thresholds(self, class_names: List[str]) -> Dict[str, float]:
        """
        Retorna thresholds padrão em caso de erro
        """
        return {class_name: 0.5 for class_name in class_names}
    
    def save_calibrated_thresholds(self, filepath: str):
        """
        Salva thresholds calibrados em arquivo JSON
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.calibrated_thresholds, f, indent=2)
            logger.info(f"Thresholds calibrados salvos em {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar thresholds: {e}")
    
    def load_calibrated_thresholds(self, filepath: str) -> Dict[str, float]:
        """
        Carrega thresholds calibrados de arquivo JSON
        """
        try:
            with open(filepath, 'r') as f:
                self.calibrated_thresholds = json.load(f)
            logger.info(f"Thresholds calibrados carregados de {filepath}")
            return self.calibrated_thresholds
        except Exception as e:
            logger.error(f"Erro ao carregar thresholds: {e}")
            return {}

    def calibrate_ensemble_thresholds(self, 
                                    ensemble_predictions: Dict[str, np.ndarray],
                                    y_true: np.ndarray,
                                    class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calibra thresholds para ensemble de modelos
        
        Args:
            ensemble_predictions: Dicionário com predições de cada modelo
            y_true: Labels verdadeiros
            class_names: Nomes das classes
            
        Returns:
            Dicionário com thresholds calibrados para cada modelo
        """
        try:
            ensemble_thresholds = {}
            
            for model_name, predictions in ensemble_predictions.items():
                logger.info(f"Calibrando thresholds para modelo: {model_name}")
                model_thresholds = self.calibrate_thresholds(
                    y_true, predictions, class_names, 'clinical'
                )
                ensemble_thresholds[model_name] = model_thresholds
            
            return ensemble_thresholds
            
        except Exception as e:
            logger.error(f"Erro na calibração de ensemble: {e}")
            return {}
    
    def optimize_ensemble_weights(self, 
                                ensemble_predictions: Dict[str, np.ndarray],
                                y_true: np.ndarray,
                                class_names: List[str]) -> Dict[str, float]:
        """
        Otimiza pesos do ensemble baseado em performance clínica
        
        Args:
            ensemble_predictions: Predições de cada modelo
            y_true: Labels verdadeiros
            class_names: Nomes das classes
            
        Returns:
            Pesos otimizados para cada modelo
        """
        try:
            from scipy.optimize import minimize
            
            model_names = list(ensemble_predictions.keys())
            n_models = len(model_names)
            
            def objective(weights):
                weights = weights / np.sum(weights)
                
                ensemble_pred = np.zeros_like(list(ensemble_predictions.values())[0])
                for i, model_name in enumerate(model_names):
                    ensemble_pred += weights[i] * ensemble_predictions[model_name]
                
                total_score = 0
                for i, class_name in enumerate(class_names):
                    if len(ensemble_pred.shape) > 1:
                        class_proba = ensemble_pred[:, i]
                        class_true = (y_true == i).astype(int)
                    else:
                        class_proba = ensemble_pred
                        class_true = y_true
                    
                    try:
                        fpr, tpr, _ = roc_curve(class_true, class_proba)
                        specificity = 1 - fpr
                        
                        clinical_req = self._get_clinical_requirements(class_name)
                        min_sens = clinical_req['min_sensitivity']
                        min_spec = clinical_req['min_specificity']
                        
                        max_sens = np.max(tpr)
                        max_spec = np.max(specificity)
                        
                        sens_penalty = max(0, min_sens - max_sens) * 10
                        spec_penalty = max(0, min_spec - max_spec) * 10
                        
                        score = max_sens + max_spec - sens_penalty - spec_penalty
                        total_score += score
                        
                    except:
                        total_score -= 1  # Penalidade por erro
                
                return -total_score  # Minimizar (negativo para maximizar)
            
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = [(0.1, 0.9) for _ in range(n_models)]
            initial_weights = np.ones(n_models) / n_models
            
            result = minimize(objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = result.x / np.sum(result.x)
                weight_dict = {model_names[i]: float(optimized_weights[i]) 
                             for i in range(n_models)}
                
                logger.info(f"Pesos otimizados: {weight_dict}")
                return weight_dict
            else:
                logger.warning("Otimização de pesos falhou, usando pesos uniformes")
                return {name: 1.0/n_models for name in model_names}
                
        except Exception as e:
            logger.error(f"Erro na otimização de pesos: {e}")
            n_models = len(ensemble_predictions)
            return {name: 1.0/n_models for name in ensemble_predictions.keys()}
    
    def validate_clinical_performance(self, 
                                    y_true: np.ndarray,
                                    y_pred_proba: np.ndarray,
                                    class_names: List[str],
                                    thresholds: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Valida performance clínica com thresholds calibrados
        
        Args:
            y_true: Labels verdadeiros
            y_pred_proba: Probabilidades preditas
            class_names: Nomes das classes
            thresholds: Thresholds calibrados
            
        Returns:
            Métricas de performance por classe
        """
        try:
            performance_metrics = {}
            
            for i, class_name in enumerate(class_names):
                if len(y_pred_proba.shape) > 1:
                    class_proba = y_pred_proba[:, i]
                    class_true = (y_true == i).astype(int)
                else:
                    class_proba = y_pred_proba
                    class_true = y_true
                
                threshold = thresholds.get(class_name, 0.5)
                class_pred = (class_proba >= threshold).astype(int)
                
                try:
                    from sklearn.metrics import confusion_matrix
                    tn, fp, fn, tp = confusion_matrix(class_true, class_pred).ravel()
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                    
                    clinical_req = self._get_clinical_requirements(class_name)
                    meets_sensitivity = sensitivity >= clinical_req['min_sensitivity']
                    meets_specificity = specificity >= clinical_req['min_specificity']
                    
                    performance_metrics[class_name] = {
                        'sensitivity': float(sensitivity),
                        'specificity': float(specificity),
                        'precision': float(precision),
                        'f1_score': float(f1),
                        'threshold': float(threshold),
                        'meets_clinical_requirements': meets_sensitivity and meets_specificity,
                        'meets_sensitivity_req': meets_sensitivity,
                        'meets_specificity_req': meets_specificity,
                        'required_sensitivity': clinical_req['min_sensitivity'],
                        'required_specificity': clinical_req['min_specificity']
                    }
                    
                except Exception as e:
                    logger.warning(f"Erro ao calcular métricas para {class_name}: {e}")
                    performance_metrics[class_name] = {
                        'error': str(e),
                        'threshold': float(threshold)
                    }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Erro na validação de performance: {e}")
            return {}
