#!/usr/bin/env python3
"""
Clinical Continuous Validator for Medical AI Models
Provides real-time clinical metrics monitoring and validation
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ClinicalValidator')

@dataclass
class ClinicalAlert:
    """Clinical performance alert"""
    timestamp: datetime
    alert_type: str
    severity: str
    metric: str
    current_value: float
    threshold: float
    message: str

@dataclass
class ValidationResult:
    """Clinical validation result"""
    timestamp: datetime
    model_name: str
    dataset: str
    sensitivity: float
    specificity: float
    accuracy: float
    auc: float
    clinical_ready: bool
    alerts: List[ClinicalAlert]

class ClinicalContinuousValidator:
    """Continuous validation system for clinical-grade medical AI models"""
    
    def __init__(self, clinical_thresholds: Optional[Dict[str, float]] = None):
        self.clinical_thresholds = clinical_thresholds or {
            'sensitivity': 0.85,
            'specificity': 0.85,
            'accuracy': 0.85,
            'auc': 0.85
        }
        self.validation_history = []
        self.alerts = []
        
    def validate_clinical_performance(self, 
                                    predictions: np.ndarray,
                                    ground_truth: np.ndarray,
                                    model_name: str,
                                    dataset: str) -> ValidationResult:
        """Validate clinical performance metrics"""
        
        metrics = self._calculate_clinical_metrics(predictions, ground_truth)
        
        clinical_ready = self._assess_clinical_readiness(metrics)
        
        alerts = self._generate_alerts(metrics, model_name)
        
        result = ValidationResult(
            timestamp=datetime.now(),
            model_name=model_name,
            dataset=dataset,
            sensitivity=metrics['sensitivity'],
            specificity=metrics['specificity'],
            accuracy=metrics['accuracy'],
            auc=metrics['auc'],
            clinical_ready=clinical_ready,
            alerts=alerts
        )
        
        self.validation_history.append(result)
        self.alerts.extend(alerts)
        
        return result
        
    def _calculate_clinical_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate clinical performance metrics"""
        
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)
            
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(ground_truth, predictions)
        except:
            auc = 0.5  # Random performance fallback
            
        return {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'tp': float(tp),
            'tn': float(tn),
            'fp': float(fp),
            'fn': float(fn)
        }
        
    def _assess_clinical_readiness(self, metrics: Dict[str, float]) -> bool:
        """Assess if model meets clinical readiness criteria"""
        
        for metric_name, threshold in self.clinical_thresholds.items():
            if metrics.get(metric_name, 0.0) < threshold:
                return False
                
        return True
        
    def _generate_alerts(self, metrics: Dict[str, float], model_name: str) -> List[ClinicalAlert]:
        """Generate clinical performance alerts"""
        
        alerts = []
        
        for metric_name, threshold in self.clinical_thresholds.items():
            current_value = metrics.get(metric_name, 0.0)
            
            if current_value < threshold:
                severity = "CRITICAL" if current_value < threshold * 0.8 else "WARNING"
                
                alert = ClinicalAlert(
                    timestamp=datetime.now(),
                    alert_type="PERFORMANCE_DEGRADATION",
                    severity=severity,
                    metric=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    message=f"{model_name}: {metric_name} ({current_value:.3f}) below clinical threshold ({threshold:.3f})"
                )
                
                alerts.append(alert)
                
        return alerts
        
    def get_clinical_summary(self) -> Dict[str, Any]:
        """Get clinical validation summary"""
        
        if not self.validation_history:
            return {"status": "No validation history"}
            
        recent_results = self.validation_history[-10:]  # Last 10 validations
        
        avg_metrics = {
            'sensitivity': np.mean([r.sensitivity for r in recent_results]),
            'specificity': np.mean([r.specificity for r in recent_results]),
            'accuracy': np.mean([r.accuracy for r in recent_results]),
            'auc': np.mean([r.auc for r in recent_results])
        }
        
        clinical_ready_count = sum(1 for r in recent_results if r.clinical_ready)
        
        return {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_results),
            'clinical_ready_rate': clinical_ready_count / len(recent_results),
            'average_metrics': avg_metrics,
            'active_alerts': len([a for a in self.alerts if a.severity == "CRITICAL"]),
            'clinical_thresholds': self.clinical_thresholds
        }
        
    def export_validation_report(self, filepath: str) -> bool:
        """Export validation report to file"""
        
        try:
            import json
            
            report = {
                'validation_summary': self.get_clinical_summary(),
                'validation_history': [
                    {
                        'timestamp': r.timestamp.isoformat(),
                        'model_name': r.model_name,
                        'dataset': r.dataset,
                        'sensitivity': r.sensitivity,
                        'specificity': r.specificity,
                        'accuracy': r.accuracy,
                        'auc': r.auc,
                        'clinical_ready': r.clinical_ready,
                        'alert_count': len(r.alerts)
                    }
                    for r in self.validation_history
                ],
                'alerts': [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'alert_type': a.alert_type,
                        'severity': a.severity,
                        'metric': a.metric,
                        'current_value': a.current_value,
                        'threshold': a.threshold,
                        'message': a.message
                    }
                    for a in self.alerts
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Validation report exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting validation report: {e}")
            return False

if __name__ == "__main__":
    validator = ClinicalContinuousValidator()
    
    predictions = np.random.rand(100)
    ground_truth = np.random.randint(0, 2, 100)
    
    result = validator.validate_clinical_performance(
        predictions, ground_truth, "TestModel", "TestDataset"
    )
    
    print(f"Validation result: {result}")
    print(f"Clinical summary: {validator.get_clinical_summary()}")
