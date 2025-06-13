"""
MedAI Confidence Calibration System
Implementa calibração de confiança para uso clínico seguro
Baseado nas melhorias do usuário para precisão diagnóstica
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, List, Optional, Callable
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfidenceCalibration:
    """
    Calibração de confiança para uso clínico
    PROBLEMA ATUAL: Modelos overconfident podem ser perigosos em medicina
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.calibration_data = None
        self.calibrators = {}
        self.calibration_methods = ['temperature', 'platt', 'isotonic']
        self.logger = logging.getLogger(__name__)
        
    def temperature_scaling(self, logits: np.ndarray, temperature: float = None) -> np.ndarray:
        """
        Temperature scaling para calibrar probabilidades
        Método simples mas efetivo para calibração
        """
        if temperature is None:
            temperature = self.temperature
            
        scaled_logits = logits / temperature
        calibrated_probs = tf.nn.softmax(scaled_logits).numpy()
        
        return calibrated_probs
    
    def fit_temperature_scaling(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Encontra temperatura ótima usando validação
        """
        from scipy.optimize import minimize_scalar
        
        def temperature_loss(temp):
            scaled_probs = self.temperature_scaling(logits, temp)
            return -np.mean(np.log(scaled_probs[np.arange(len(y_true)), y_true] + 1e-8))
        
        result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
        optimal_temp = result.x
        
        self.temperature = optimal_temp
        self.logger.info(f"Temperatura ótima encontrada: {optimal_temp:.3f}")
        
        return optimal_temp
    
    def platt_scaling(self, predictions: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
        """
        Platt scaling para calibração usando regressão logística
        """
        if len(np.unique(y_true)) == 2:
            lr = LogisticRegression()
            lr.fit(predictions.reshape(-1, 1), y_true)
            
            self.calibrators['platt'] = lr
            return lr
        
        from sklearn.multiclass import OneVsRestClassifier
        ovr = OneVsRestClassifier(LogisticRegression())
        ovr.fit(predictions, y_true)
        
        self.calibrators['platt_ovr'] = ovr
        return ovr
    
    def isotonic_regression(self, predictions: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
        """
        Isotonic regression para calibração não-paramétrica
        """
        if len(np.unique(y_true)) == 2:
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(predictions[:, 1] if predictions.ndim > 1 else predictions, y_true)
            
            self.calibrators['isotonic'] = iso_reg
            return iso_reg
        
        calibrators = {}
        for class_idx in range(predictions.shape[1]):
            y_binary = (y_true == class_idx).astype(int)
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(predictions[:, class_idx], y_binary)
            calibrators[f'class_{class_idx}'] = iso_reg
        
        self.calibrators['isotonic_multiclass'] = calibrators
        return calibrators
    
    def apply_calibration(self, predictions: np.ndarray, method: str = 'temperature') -> np.ndarray:
        """
        Aplica calibração usando método especificado
        """
        if method == 'temperature':
            if predictions.ndim == 1:
                logits = np.column_stack([-predictions, predictions])
            else:
                logits = predictions
            return self.temperature_scaling(logits)
        
        elif method == 'platt':
            if 'platt' in self.calibrators:
                calibrator = self.calibrators['platt']
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                return calibrator.predict_proba(predictions)
            elif 'platt_ovr' in self.calibrators:
                return self.calibrators['platt_ovr'].predict_proba(predictions)
        
        elif method == 'isotonic':
            if 'isotonic' in self.calibrators:
                calibrator = self.calibrators['isotonic']
                calibrated = calibrator.predict(predictions[:, 1] if predictions.ndim > 1 else predictions)
                return np.column_stack([1 - calibrated, calibrated])
            elif 'isotonic_multiclass' in self.calibrators:
                calibrators = self.calibrators['isotonic_multiclass']
                calibrated_probs = np.zeros_like(predictions)
                for class_idx, calibrator in calibrators.items():
                    if class_idx.startswith('class_'):
                        idx = int(class_idx.split('_')[1])
                        calibrated_probs[:, idx] = calibrator.predict(predictions[:, idx])
                calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
                return calibrated_probs
        
        self.logger.warning(f"Método de calibração '{method}' não disponível. Retornando predições originais.")
        return predictions
    
    def expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Calcula ECE (Expected Calibration Error) para avaliar calibração
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Calcula MCE (Maximum Calibration Error)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def reliability_diagram(self, y_true: np.ndarray, y_prob: np.ndarray, 
                          n_bins: int = 10, save_path: Optional[str] = None) -> Dict:
        """
        Gera diagrama de confiabilidade para visualizar calibração
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                count_in_bin = in_bin.sum()
                
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(count_in_bin)
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfeitamente calibrado')
        plt.bar(bin_centers, bin_accuracies, width=0.1, alpha=0.7, 
                label='Acurácia por bin', edgecolor='black')
        plt.plot(bin_centers, bin_confidences, 'ro-', label='Confiança média')
        
        plt.xlabel('Confiança Média')
        plt.ylabel('Acurácia')
        plt.title('Diagrama de Confiabilidade')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Diagrama de confiabilidade salvo em: {save_path}")
        
        plt.close()
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }
    
    def calibrate_model_predictions(self, model: tf.keras.Model, 
                                  validation_data: Tuple[np.ndarray, np.ndarray],
                                  method: str = 'temperature') -> Dict:
        """
        Calibra predições de um modelo usando dados de validação
        """
        X_val, y_val = validation_data
        
        predictions = model.predict(X_val)
        
        if method == 'temperature':
            if np.max(predictions) > 1.0:  # Provavelmente logits
                optimal_temp = self.fit_temperature_scaling(predictions, y_val)
                calibrated_preds = self.temperature_scaling(predictions)
            else:  # Provavelmente probabilidades
                logits = np.log(predictions + 1e-8)
                optimal_temp = self.fit_temperature_scaling(logits, y_val)
                calibrated_preds = self.temperature_scaling(logits)
        
        elif method == 'platt':
            self.platt_scaling(predictions, y_val)
            calibrated_preds = self.apply_calibration(predictions, 'platt')
        
        elif method == 'isotonic':
            self.isotonic_regression(predictions, y_val)
            calibrated_preds = self.apply_calibration(predictions, 'isotonic')
        
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            max_probs_original = np.max(predictions, axis=1)
            max_probs_calibrated = np.max(calibrated_preds, axis=1)
            y_pred_original = np.argmax(predictions, axis=1)
            y_pred_calibrated = np.argmax(calibrated_preds, axis=1)
            
            ece_original = self.expected_calibration_error(
                (y_pred_original == y_val).astype(int), max_probs_original
            )
            ece_calibrated = self.expected_calibration_error(
                (y_pred_calibrated == y_val).astype(int), max_probs_calibrated
            )
        else:
            ece_original = self.expected_calibration_error(y_val, predictions[:, 1])
            ece_calibrated = self.expected_calibration_error(y_val, calibrated_preds[:, 1])
        
        results = {
            'method': method,
            'original_predictions': predictions,
            'calibrated_predictions': calibrated_preds,
            'ece_original': ece_original,
            'ece_calibrated': ece_calibrated,
            'improvement': ece_original - ece_calibrated
        }
        
        self.logger.info(f"Calibração usando {method}:")
        self.logger.info(f"  ECE original: {ece_original:.4f}")
        self.logger.info(f"  ECE calibrado: {ece_calibrated:.4f}")
        self.logger.info(f"  Melhoria: {results['improvement']:.4f}")
        
        return results

class ClinicalConfidenceThresholds:
    """
    Sistema de limiares de confiança para uso clínico
    """
    
    def __init__(self):
        self.thresholds = {
            'high_confidence': 0.9,    # Casos de alta confiança
            'medium_confidence': 0.7,  # Casos de confiança média
            'low_confidence': 0.5,     # Casos de baixa confiança
            'reject_threshold': 0.3    # Casos para rejeição/revisão humana
        }
        self.clinical_actions = {
            'high_confidence': 'automated_report',
            'medium_confidence': 'flagged_review',
            'low_confidence': 'manual_review',
            'reject': 'human_expert_required'
        }
    
    def classify_confidence_level(self, confidence: float) -> str:
        """
        Classifica nível de confiança para ação clínica
        """
        if confidence >= self.thresholds['high_confidence']:
            return 'high_confidence'
        elif confidence >= self.thresholds['medium_confidence']:
            return 'medium_confidence'
        elif confidence >= self.thresholds['low_confidence']:
            return 'low_confidence'
        else:
            return 'reject'
    
    def get_clinical_action(self, confidence: float) -> str:
        """
        Retorna ação clínica recomendada baseada na confiança
        """
        confidence_level = self.classify_confidence_level(confidence)
        return self.clinical_actions[confidence_level]
    
    def optimize_thresholds_for_sensitivity(self, y_true: np.ndarray, 
                                          y_prob: np.ndarray,
                                          target_sensitivity: float = 0.95) -> Dict:
        """
        Otimiza limiares para atingir sensibilidade alvo
        """
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        target_idx = np.argmin(np.abs(tpr - target_sensitivity))
        optimal_threshold = thresholds[target_idx]
        achieved_sensitivity = tpr[target_idx]
        achieved_specificity = 1 - fpr[target_idx]
        
        self.thresholds['high_confidence'] = min(0.95, optimal_threshold + 0.1)
        self.thresholds['medium_confidence'] = optimal_threshold
        self.thresholds['low_confidence'] = max(0.3, optimal_threshold - 0.2)
        
        return {
            'optimal_threshold': optimal_threshold,
            'achieved_sensitivity': achieved_sensitivity,
            'achieved_specificity': achieved_specificity,
            'updated_thresholds': self.thresholds.copy()
        }

def create_calibrated_model_wrapper(model: tf.keras.Model, 
                                  calibrator: ConfidenceCalibration,
                                  method: str = 'temperature') -> Callable:
    """
    Cria wrapper para modelo com calibração automática
    """
    def calibrated_predict(X):
        predictions = model.predict(X)
        
        calibrated_predictions = calibrator.apply_calibration(predictions, method)
        
        return calibrated_predictions
    
    return calibrated_predict

def integrate_with_existing_pipeline():
    """
    Integra sistema de calibração com pipeline existente
    """
    logger.info("Integrando sistema de calibração de confiança com pipeline existente")
    logger.info("Adicionando calibração para uso clínico seguro")
    
    return {
        'confidence_calibration': ConfidenceCalibration(),
        'clinical_thresholds': ClinicalConfidenceThresholds(),
        'calibrated_wrapper': create_calibrated_model_wrapper
    }
