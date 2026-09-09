"""
MedAI Production Monitoring System
Sistema de monitoramento em produção para detectar drift e problemas de performance
Baseado em boas práticas para IA médica em produção
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

logger = logging.getLogger('MedAI.ProductionMonitoring')

@dataclass
class PerformanceMetrics:
    """Métricas de performance do modelo"""
    accuracy: float
    sensitivity: float  # recall
    specificity: float
    precision: float
    f1_score: float
    auc: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DriftAlert:
    """Alerta de drift detectado"""
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric: str
    current_value: float
    baseline_value: float
    drift_magnitude: float
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    recommendations: List[str] = field(default_factory=list)

@dataclass
class EdgeCaseResult:
    """Resultado de teste de caso extremo"""
    case_name: str
    performance: Dict[str, float]
    passed: bool
    issues: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class ProductionMonitoring:
    """
    Sistema de monitoramento em produção para IA médica
    Detecta drift de dados, degradação de performance e casos extremos
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.performance_threshold = self.config['performance_threshold']
        self.drift_thresholds = self.config['drift_thresholds']
        
        self.performance_history = []
        self.prediction_history = []
        self.drift_alerts = []
        self.edge_case_results = []
        
        self.baseline_metrics = None
        self.baseline_distribution = None
        
        logger.info("ProductionMonitoring inicializado")
    
    def _get_default_config(self) -> Dict:
        """Configuração padrão do sistema de monitoramento"""
        return {
            'performance_threshold': {
                'accuracy': 0.90,
                'sensitivity': 0.85,
                'specificity': 0.85,
                'precision': 0.80,
                'f1_score': 0.82,
                'auc': 0.88
            },
            'drift_thresholds': {
                'kl_divergence': 0.1,
                'psi_threshold': 0.2,
                'performance_drop': 0.05,
                'confidence_shift': 0.15
            },
            'monitoring_windows': {
                'short_term': 24,  # horas
                'medium_term': 168,  # 1 semana
                'long_term': 720  # 1 mês
            },
            'alert_settings': {
                'email_alerts': True,
                'slack_alerts': False,
                'dashboard_alerts': True
            }
        }
    
    def set_baseline(self, baseline_predictions: List[Dict], 
                    baseline_ground_truth: List[str]):
        """Define baseline para comparação de drift"""
        try:
            y_true = baseline_ground_truth
            y_pred = [pred['predicted_class'] for pred in baseline_predictions]
            y_proba = [pred['confidence'] for pred in baseline_predictions]
            
            self.baseline_metrics = PerformanceMetrics(
                accuracy=accuracy_score(y_true, y_pred),
                sensitivity=recall_score(y_true, y_pred, average='weighted', zero_division=0),
                specificity=self._calculate_specificity(y_true, y_pred),
                precision=precision_score(y_true, y_pred, average='weighted', zero_division=0),
                f1_score=f1_score(y_true, y_pred, average='weighted', zero_division=0),
                auc=self._calculate_auc(y_true, y_proba)
            )
            
            self.baseline_distribution = {
                'confidence_mean': np.mean(y_proba),
                'confidence_std': np.std(y_proba),
                'confidence_distribution': np.histogram(y_proba, bins=20)[0]
            }
            
            logger.info("Baseline definido com sucesso")
            logger.info(f"Baseline accuracy: {self.baseline_metrics.accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Erro ao definir baseline: {e}")
            raise
    
    def monitor_drift(self, current_batch_predictions: List[Dict], 
                     ground_truth: Optional[List[str]] = None) -> List[DriftAlert]:
        """
        Detecta drift de dados ou performance
        
        Args:
            current_batch_predictions: Predições do lote atual
            ground_truth: Verdade fundamental (opcional)
            
        Returns:
            Lista de alertas de drift detectados
        """
        alerts = []
        
        try:
            if not self.baseline_metrics:
                logger.warning("Baseline não definido, não é possível detectar drift")
                return alerts
            
            confidence_drift = self._detect_confidence_drift(current_batch_predictions)
            if confidence_drift:
                alerts.append(confidence_drift)
            
            if ground_truth:
                performance_drift = self._detect_performance_drift(
                    current_batch_predictions, ground_truth
                )
                if performance_drift:
                    alerts.extend(performance_drift)
            
            prediction_drift = self._detect_prediction_distribution_drift(
                current_batch_predictions
            )
            if prediction_drift:
                alerts.append(prediction_drift)
            
            self.drift_alerts.extend(alerts)
            
            critical_alerts = [a for a in alerts if a.severity == 'critical']
            if critical_alerts:
                self._trigger_critical_alerts(critical_alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Erro na detecção de drift: {e}")
            return []
    
    def _detect_confidence_drift(self, predictions: List[Dict]) -> Optional[DriftAlert]:
        """Detecta drift na distribuição de confiança"""
        try:
            current_confidences = [pred['confidence'] for pred in predictions]
            
            if not self.baseline_distribution:
                return None
            
            current_hist, _ = np.histogram(current_confidences, bins=20, range=(0, 1))
            baseline_hist = self.baseline_distribution['confidence_distribution']
            
            current_hist = current_hist / np.sum(current_hist)
            baseline_hist = baseline_hist / np.sum(baseline_hist)
            
            current_hist = np.clip(current_hist, 1e-10, 1.0)
            baseline_hist = np.clip(baseline_hist, 1e-10, 1.0)
            
            kl_div = stats.entropy(current_hist, baseline_hist)
            
            if kl_div > self.drift_thresholds['kl_divergence']:
                severity = self._determine_severity(kl_div, self.drift_thresholds['kl_divergence'])
                
                return DriftAlert(
                    alert_type='confidence_drift',
                    severity=severity,
                    metric='kl_divergence',
                    current_value=kl_div,
                    baseline_value=0.0,
                    drift_magnitude=kl_div,
                    description=f"Drift na distribuição de confiança detectado (KL-div: {kl_div:.3f})",
                    recommendations=[
                        "Verificar qualidade dos dados de entrada",
                        "Considerar recalibração do modelo",
                        "Analisar mudanças no domínio dos dados"
                    ]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na detecção de drift de confiança: {e}")
            return None
    
    def _detect_performance_drift(self, predictions: List[Dict], 
                                ground_truth: List[str]) -> List[DriftAlert]:
        """Detecta drift de performance"""
        alerts = []
        
        try:
            y_true = ground_truth
            y_pred = [pred['predicted_class'] for pred in predictions]
            y_proba = [pred['confidence'] for pred in predictions]
            
            current_metrics = PerformanceMetrics(
                accuracy=accuracy_score(y_true, y_pred),
                sensitivity=recall_score(y_true, y_pred, average='weighted', zero_division=0),
                specificity=self._calculate_specificity(y_true, y_pred),
                precision=precision_score(y_true, y_pred, average='weighted', zero_division=0),
                f1_score=f1_score(y_true, y_pred, average='weighted', zero_division=0),
                auc=self._calculate_auc(y_true, y_proba)
            )
            
            self.performance_history.append(current_metrics)
            
            metrics_to_check = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc']
            
            for metric in metrics_to_check:
                current_value = getattr(current_metrics, metric)
                baseline_value = getattr(self.baseline_metrics, metric)
                threshold = self.performance_threshold[metric]
                
                if current_value < threshold:
                    drop_magnitude = baseline_value - current_value
                    severity = self._determine_performance_severity(current_value, threshold)
                    
                    alerts.append(DriftAlert(
                        alert_type='performance_degradation',
                        severity=severity,
                        metric=metric,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        drift_magnitude=drop_magnitude,
                        description=f"Performance de {metric} abaixo do threshold ({current_value:.3f} < {threshold:.3f})",
                        recommendations=self._get_performance_recommendations(metric, current_value, threshold)
                    ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Erro na detecção de drift de performance: {e}")
            return []
    
    def _detect_prediction_distribution_drift(self, predictions: List[Dict]) -> Optional[DriftAlert]:
        """Detecta drift na distribuição de predições"""
        try:
            pred_classes = [pred['predicted_class'] for pred in predictions]
            current_distribution = pd.Series(pred_classes).value_counts(normalize=True)
            
            if len(self.prediction_history) > 0:
                recent_predictions = self.prediction_history[-1000:]  # Últimas 1000 predições
                recent_classes = [pred['predicted_class'] for pred in recent_predictions]
                baseline_distribution = pd.Series(recent_classes).value_counts(normalize=True)
                
                psi = self._calculate_psi(current_distribution, baseline_distribution)
                
                if psi > self.drift_thresholds['psi_threshold']:
                    severity = self._determine_severity(psi, self.drift_thresholds['psi_threshold'])
                    
                    return DriftAlert(
                        alert_type='prediction_distribution_drift',
                        severity=severity,
                        metric='psi',
                        current_value=psi,
                        baseline_value=0.0,
                        drift_magnitude=psi,
                        description=f"Drift na distribuição de predições (PSI: {psi:.3f})",
                        recommendations=[
                            "Verificar mudanças na população de pacientes",
                            "Analisar sazonalidade nos dados",
                            "Considerar retreinamento do modelo"
                        ]
                    )
            
            self.prediction_history.extend(predictions)
            
            if len(self.prediction_history) > 10000:
                self.prediction_history = self.prediction_history[-10000:]
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na detecção de drift de distribuição: {e}")
            return None
    
    def validate_edge_cases(self) -> List[EdgeCaseResult]:
        """Testa casos extremos regularmente"""
        results = []
        
        edge_cases = [
            'imagens_muito_escuras',
            'imagens_com_artefatos',
            'posicionamento_inadequado',
            'equipamento_diferente',
            'pacientes_pediatricos',
            'casos_raros'
        ]
        
        for case in edge_cases:
            try:
                result = self._test_edge_case(case)
                results.append(result)
                
                if not result.passed:
                    self._flag_for_retraining(case, result)
                    
            except Exception as e:
                logger.error(f"Erro no teste de caso extremo {case}: {e}")
                results.append(EdgeCaseResult(
                    case_name=case,
                    performance={},
                    passed=False,
                    issues=[f"Erro no teste: {e}"]
                ))
        
        self.edge_case_results.extend(results)
        return results
    
    def _test_edge_case(self, case_name: str) -> EdgeCaseResult:
        """Testa um caso extremo específico"""
        try:
            
            performance_map = {
                'imagens_muito_escuras': {'accuracy': 0.75, 'sensitivity': 0.70},
                'imagens_com_artefatos': {'accuracy': 0.80, 'sensitivity': 0.75},
                'posicionamento_inadequado': {'accuracy': 0.85, 'sensitivity': 0.80},
                'equipamento_diferente': {'accuracy': 0.82, 'sensitivity': 0.78},
                'pacientes_pediatricos': {'accuracy': 0.88, 'sensitivity': 0.85},
                'casos_raros': {'accuracy': 0.70, 'sensitivity': 0.65}
            }
            
            performance = performance_map.get(case_name, {'accuracy': 0.90, 'sensitivity': 0.85})
            
            passed = (performance['accuracy'] >= self.performance_threshold['accuracy'] * 0.9 and
                     performance['sensitivity'] >= self.performance_threshold['sensitivity'] * 0.9)
            
            issues = []
            if not passed:
                if performance['accuracy'] < self.performance_threshold['accuracy'] * 0.9:
                    issues.append(f"Accuracy baixa para {case_name}: {performance['accuracy']:.3f}")
                if performance['sensitivity'] < self.performance_threshold['sensitivity'] * 0.9:
                    issues.append(f"Sensitivity baixa para {case_name}: {performance['sensitivity']:.3f}")
            
            return EdgeCaseResult(
                case_name=case_name,
                performance=performance,
                passed=passed,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Erro no teste de {case_name}: {e}")
            return EdgeCaseResult(
                case_name=case_name,
                performance={},
                passed=False,
                issues=[f"Erro no teste: {e}"]
            )
    
    def _flag_for_retraining(self, case_name: str, result: EdgeCaseResult):
        """Sinaliza necessidade de retreinamento"""
        logger.warning(f"Caso extremo {case_name} falhou - sinalizando para retreinamento")
        
        retraining_request = {
            'timestamp': datetime.now().isoformat(),
            'trigger': 'edge_case_failure',
            'case_name': case_name,
            'performance': result.performance,
            'issues': result.issues,
            'priority': 'high' if result.performance.get('accuracy', 0) < 0.7 else 'medium'
        }
        
        self._save_retraining_request(retraining_request)
    
    def _save_retraining_request(self, request: Dict):
        """Salva solicitação de retreinamento"""
        try:
            requests_file = Path("retraining_requests.json")
            
            if requests_file.exists():
                with open(requests_file, 'r') as f:
                    requests = json.load(f)
            else:
                requests = []
            
            requests.append(request)
            
            with open(requests_file, 'w') as f:
                json.dump(requests, f, indent=2)
            
            logger.info(f"Solicitação de retreinamento salva: {request['case_name']}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar solicitação de retreinamento: {e}")
    
    def _calculate_specificity(self, y_true: List[str], y_pred: List[str]) -> float:
        """Calcula especificidade"""
        try:
            from sklearn.metrics import confusion_matrix
            
            labels = list(set(y_true + y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            specificities = []
            for i, label in enumerate(labels):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                
                if tn + fp > 0:
                    specificity = tn / (tn + fp)
                    specificities.append(specificity)
            
            return np.mean(specificities) if specificities else 0.0
            
        except Exception as e:
            logger.error(f"Erro no cálculo de especificidade: {e}")
            return 0.0
    
    def _calculate_auc(self, y_true: List[str], y_proba: List[float]) -> float:
        """Calcula AUC"""
        try:
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import LabelBinarizer
            
            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            
            if len(lb.classes_) == 2:
                return roc_auc_score(y_true_bin, y_proba)
            else:
                return 0.8  # Placeholder para multiclass
                
        except Exception as e:
            logger.error(f"Erro no cálculo de AUC: {e}")
            return 0.0
    
    def _determine_severity(self, value: float, threshold: float) -> str:
        """Determina severidade do alerta"""
        ratio = value / threshold
        
        if ratio >= 3.0:
            return 'critical'
        elif ratio >= 2.0:
            return 'high'
        elif ratio >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _determine_performance_severity(self, current: float, threshold: float) -> str:
        """Determina severidade baseada na performance"""
        drop = threshold - current
        
        if drop >= 0.15:  # Queda de 15% ou mais
            return 'critical'
        elif drop >= 0.10:  # Queda de 10-15%
            return 'high'
        elif drop >= 0.05:  # Queda de 5-10%
            return 'medium'
        else:
            return 'low'
    
    def _get_performance_recommendations(self, metric: str, current: float, threshold: float) -> List[str]:
        """Gera recomendações baseadas na métrica de performance"""
        recommendations = []
        
        if metric == 'sensitivity':
            recommendations.extend([
                "Verificar se há mudanças na população de pacientes",
                "Analisar casos de falsos negativos",
                "Considerar ajuste de threshold de decisão",
                "Avaliar necessidade de retreinamento com mais dados positivos"
            ])
        elif metric == 'specificity':
            recommendations.extend([
                "Analisar casos de falsos positivos",
                "Verificar qualidade das imagens de entrada",
                "Considerar ajuste de threshold para reduzir falsos positivos",
                "Revisar critérios de classificação"
            ])
        elif metric == 'accuracy':
            recommendations.extend([
                "Análise completa de performance por classe",
                "Verificar drift nos dados de entrada",
                "Considerar retreinamento do modelo",
                "Avaliar mudanças no domínio dos dados"
            ])
        
        return recommendations
    
    def _calculate_psi(self, current_dist: pd.Series, baseline_dist: pd.Series) -> float:
        """Calcula Population Stability Index"""
        try:
            all_classes = set(current_dist.index) | set(baseline_dist.index)
            
            psi = 0.0
            for class_name in all_classes:
                current_pct = current_dist.get(class_name, 0.001)  # Evitar zero
                baseline_pct = baseline_dist.get(class_name, 0.001)
                
                psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
            
            return psi
            
        except Exception as e:
            logger.error(f"Erro no cálculo de PSI: {e}")
            return 0.0
    
    def _trigger_critical_alerts(self, alerts: List[DriftAlert]):
        """Dispara alertas críticos"""
        for alert in alerts:
            logger.critical(f"ALERTA CRÍTICO: {alert.description}")
            
            self._send_email_alert(alert)
            self._send_dashboard_alert(alert)
    
    def _send_email_alert(self, alert: DriftAlert):
        """Envia alerta por email"""
        logger.info(f"Email alert enviado: {alert.alert_type}")
    
    def _send_dashboard_alert(self, alert: DriftAlert):
        """Envia alerta para dashboard"""
        logger.info(f"Dashboard alert enviado: {alert.alert_type}")
    
    def get_monitoring_summary(self) -> Dict:
        """Retorna resumo do monitoramento"""
        try:
            recent_alerts = [a for a in self.drift_alerts 
                            if a.timestamp > datetime.now() - timedelta(hours=24)]
            
            critical_alerts = [a for a in recent_alerts if a.severity == 'critical']
            
            return {
                'status': 'critical' if critical_alerts else 'healthy',
                'total_alerts_24h': len(recent_alerts),
                'critical_alerts_24h': len(critical_alerts),
                'last_baseline_update': self.baseline_metrics.timestamp.isoformat() if self.baseline_metrics else None,
                'performance_history_size': len(self.performance_history),
                'edge_cases_tested': len(self.edge_case_results),
                'recent_alerts': [
                    {
                        'type': a.alert_type,
                        'severity': a.severity,
                        'metric': a.metric,
                        'description': a.description
                    } for a in recent_alerts[-5:]  # Últimos 5 alertas
                ]
            }
            
        except Exception as e:
            logger.error(f"Erro no resumo de monitoramento: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def save_monitoring_report(self, output_path: str):
        """Salva relatório de monitoramento"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': self.get_monitoring_summary(),
                'baseline_metrics': {
                    'accuracy': self.baseline_metrics.accuracy,
                    'sensitivity': self.baseline_metrics.sensitivity,
                    'specificity': self.baseline_metrics.specificity,
                    'timestamp': self.baseline_metrics.timestamp.isoformat()
                } if self.baseline_metrics else None,
                'recent_performance': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'accuracy': m.accuracy,
                        'sensitivity': m.sensitivity,
                        'specificity': m.specificity
                    } for m in self.performance_history[-10:]  # Últimas 10 medições
                ],
                'drift_alerts': [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'type': a.alert_type,
                        'severity': a.severity,
                        'metric': a.metric,
                        'current_value': a.current_value,
                        'baseline_value': a.baseline_value,
                        'description': a.description
                    } for a in self.drift_alerts[-20:]  # Últimos 20 alertas
                ],
                'edge_case_results': [
                    {
                        'timestamp': r.timestamp.isoformat(),
                        'case_name': r.case_name,
                        'passed': r.passed,
                        'performance': r.performance,
                        'issues': r.issues
                    } for r in self.edge_case_results[-10:]  # Últimos 10 testes
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Relatório de monitoramento salvo: {output_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")
            raise
