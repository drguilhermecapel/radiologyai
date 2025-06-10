"""
MedAI Clinical Evaluation - Sistema de avaliação clínica
Implementa métricas e benchmarks para avaliação de performance clínica
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger('MedAI.ClinicalEvaluation')

class ClinicalPerformanceEvaluator:
    """
    Avaliador de performance clínica para modelos de IA médica
    Implementa métricas específicas para radiologia
    """
    
    def __init__(self):
        self.evaluation_history = []
        self.benchmarks = {}
        logger.info("ClinicalPerformanceEvaluator inicializado")
    
    def evaluate_model_performance(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Avalia performance do modelo usando métricas clínicas
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições do modelo
            y_prob: Probabilidades das predições (opcional)
            
        Returns:
            Dicionário com métricas de performance
        """
        try:
            metrics = {}
            
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            metrics.update(self._calculate_clinical_metrics(y_true, y_pred))
            
            if y_prob is not None:
                try:
                    if len(np.unique(y_true)) == 2:  # Classificação binária
                        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
                    else:  # Classificação multi-classe
                        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
                except Exception as e:
                    logger.warning(f"Erro no cálculo do AUC: {e}")
                    metrics['auc_roc'] = 0.0
            
            metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação de performance: {e}")
            return {'error': str(e)}
    
    def _calculate_clinical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calcula métricas específicas para diagnóstico clínico"""
        try:
            metrics = {}
            
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            
            for class_label in unique_classes:
                y_true_binary = (y_true == class_label).astype(int)
                y_pred_binary = (y_pred == class_label).astype(int)
                
                tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
                tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
                fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                
                metrics[f'class_{class_label}_sensitivity'] = float(sensitivity)
                metrics[f'class_{class_label}_specificity'] = float(specificity)
                metrics[f'class_{class_label}_ppv'] = float(ppv)
                metrics[f'class_{class_label}_npv'] = float(npv)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de métricas clínicas: {e}")
            return {}
    
    def generate_clinical_report(self, metrics: Dict, model_name: str = "MedAI Model") -> str:
        """
        Gera relatório clínico de performance
        
        Args:
            metrics: Métricas calculadas
            model_name: Nome do modelo
            
        Returns:
            Relatório formatado como string
        """
        try:
            report = f"""

- Acurácia: {metrics.get('accuracy', 0):.3f}
- Precisão: {metrics.get('precision', 0):.3f}
- Sensibilidade (Recall): {metrics.get('recall', 0):.3f}
- F1-Score: {metrics.get('f1_score', 0):.3f}
- AUC-ROC: {metrics.get('auc_roc', 'N/A')}

"""
            
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= 0.95:
                report += "- Performance EXCELENTE para uso clínico\n"
            elif accuracy >= 0.90:
                report += "- Performance BOA para uso clínico com supervisão\n"
            elif accuracy >= 0.80:
                report += "- Performance MODERADA - requer validação adicional\n"
            else:
                report += "- Performance BAIXA - não recomendado para uso clínico\n"
            
            class_metrics = {k: v for k, v in metrics.items() if k.startswith('class_')}
            if class_metrics:
                report += "\n## Métricas por Classe\n"
                for metric_name, value in class_metrics.items():
                    report += f"- {metric_name}: {value:.3f}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Erro na geração do relatório clínico: {e}")
            return f"Erro na geração do relatório: {e}"

class RadiologyBenchmark:
    """
    Sistema de benchmark para modelos de radiologia
    Compara performance com padrões da literatura médica
    """
    
    def __init__(self):
        self.benchmarks = self._load_medical_benchmarks()
        logger.info("RadiologyBenchmark inicializado")
    
    def _load_medical_benchmarks(self) -> Dict:
        """Carrega benchmarks da literatura médica"""
        return {
            'chest_xray': {
                'pneumonia_detection': {
                    'accuracy': 0.92,
                    'sensitivity': 0.89,
                    'specificity': 0.94,
                    'reference': 'Rajpurkar et al. 2017'
                },
                'normal_vs_abnormal': {
                    'accuracy': 0.95,
                    'sensitivity': 0.93,
                    'specificity': 0.96,
                    'reference': 'Wang et al. 2017'
                }
            },
            'brain_ct': {
                'hemorrhage_detection': {
                    'accuracy': 0.94,
                    'sensitivity': 0.91,
                    'specificity': 0.96,
                    'reference': 'Arbabshirani et al. 2018'
                }
            },
            'bone_xray': {
                'fracture_detection': {
                    'accuracy': 0.90,
                    'sensitivity': 0.88,
                    'specificity': 0.92,
                    'reference': 'Lindsey et al. 2018'
                }
            }
        }
    
    def compare_with_benchmark(self, 
                             metrics: Dict, 
                             modality: str, 
                             task: str) -> Dict:
        """
        Compara métricas do modelo com benchmarks da literatura
        
        Args:
            metrics: Métricas do modelo
            modality: Modalidade (chest_xray, brain_ct, bone_xray)
            task: Tarefa específica
            
        Returns:
            Comparação com benchmark
        """
        try:
            if modality not in self.benchmarks:
                return {'error': f'Modalidade {modality} não encontrada nos benchmarks'}
            
            if task not in self.benchmarks[modality]:
                return {'error': f'Tarefa {task} não encontrada para {modality}'}
            
            benchmark = self.benchmarks[modality][task]
            comparison = {
                'benchmark_reference': benchmark['reference'],
                'comparisons': {}
            }
            
            for metric in ['accuracy', 'sensitivity', 'specificity']:
                if metric in metrics and metric in benchmark:
                    model_value = metrics[metric]
                    benchmark_value = benchmark[metric]
                    difference = model_value - benchmark_value
                    
                    comparison['comparisons'][metric] = {
                        'model_value': model_value,
                        'benchmark_value': benchmark_value,
                        'difference': difference,
                        'performance': 'superior' if difference > 0.01 else 'comparable' if abs(difference) <= 0.01 else 'inferior'
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Erro na comparação com benchmark: {e}")
            return {'error': str(e)}
    
    def generate_benchmark_report(self, comparison: Dict) -> str:
        """Gera relatório de comparação com benchmark"""
        try:
            if 'error' in comparison:
                return f"Erro na comparação: {comparison['error']}"
            
            report = f"""

Referência: {comparison['benchmark_reference']}

"""
            
            for metric, data in comparison['comparisons'].items():
                performance = data['performance']
                emoji = "🟢" if performance == 'superior' else "🟡" if performance == 'comparable' else "🔴"
                
                report += f"""
{emoji} **{metric.upper()}**
- Modelo: {data['model_value']:.3f}
- Benchmark: {data['benchmark_value']:.3f}
- Diferença: {data['difference']:+.3f}
- Performance: {performance}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Erro na geração do relatório de benchmark: {e}")
            return f"Erro na geração do relatório: {e}"

class ClinicalValidationFramework:
    """
    Framework para validação clínica de modelos de IA
    """
    
    def __init__(self):
        self.validation_protocols = {}
        logger.info("ClinicalValidationFramework inicializado")
    
    def validate_for_clinical_use(self, model_metrics: Dict) -> Dict:
        """
        Valida se o modelo está pronto para uso clínico
        
        Args:
            model_metrics: Métricas do modelo
            
        Returns:
            Resultado da validação clínica
        """
        try:
            validation_result = {
                'approved_for_clinical_use': False,
                'validation_criteria': {},
                'recommendations': []
            }
            
            min_accuracy = 0.85
            min_sensitivity = 0.80
            min_specificity = 0.80
            
            accuracy = model_metrics.get('accuracy', 0)
            sensitivity = model_metrics.get('recall', 0)  # Recall é sensibilidade
            specificity = self._calculate_average_specificity(model_metrics)
            
            validation_result['validation_criteria'] = {
                'accuracy': {
                    'value': accuracy,
                    'threshold': min_accuracy,
                    'passed': accuracy >= min_accuracy
                },
                'sensitivity': {
                    'value': sensitivity,
                    'threshold': min_sensitivity,
                    'passed': sensitivity >= min_sensitivity
                },
                'specificity': {
                    'value': specificity,
                    'threshold': min_specificity,
                    'passed': specificity >= min_specificity
                }
            }
            
            all_passed = all(criteria['passed'] for criteria in validation_result['validation_criteria'].values())
            validation_result['approved_for_clinical_use'] = all_passed
            
            if not all_passed:
                validation_result['recommendations'].append("Modelo não atende aos critérios mínimos para uso clínico")
                
                for metric, criteria in validation_result['validation_criteria'].items():
                    if not criteria['passed']:
                        validation_result['recommendations'].append(
                            f"Melhorar {metric}: atual {criteria['value']:.3f}, mínimo {criteria['threshold']:.3f}"
                        )
            else:
                validation_result['recommendations'].append("Modelo aprovado para uso clínico com supervisão adequada")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Erro na validação clínica: {e}")
            return {'error': str(e)}
    
    def _calculate_average_specificity(self, metrics: Dict) -> float:
        """Calcula especificidade média das classes"""
        try:
            specificity_values = []
            for key, value in metrics.items():
                if 'specificity' in key and isinstance(value, (int, float)):
                    specificity_values.append(value)
            
            return np.mean(specificity_values) if specificity_values else 0.0
            
        except Exception as e:
            logger.warning(f"Erro no cálculo da especificidade média: {e}")
            return 0.0
