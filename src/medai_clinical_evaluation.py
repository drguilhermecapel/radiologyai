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
    Implementa métricas específicas para radiologia baseadas no scientific guide
    Sensibilidade > 95% para condições críticas, > 90% para moderadas
    """
    
    def __init__(self, class_names: List[str] = None):
        self.evaluation_history = []
        self.benchmarks = {}
        self.class_names = class_names or ['normal', 'pneumonia', 'pleural_effusion', 'fracture', 'tumor']
        
        self.critical_conditions = ['pneumothorax', 'tumor', 'fracture']  # >95% sensibilidade
        self.moderate_conditions = ['pneumonia', 'pleural_effusion']      # >90% sensibilidade
        self.target_specificity = 0.90  # >90% especificidade para reduzir falsos positivos
        
        logger.info("ClinicalPerformanceEvaluator inicializado com validação clínica avançada")
    
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
        """
        Calcula métricas específicas para diagnóstico clínico baseadas no scientific guide
        Inclui validação de criticidade clínica e compliance com padrões médicos
        """
        try:
            metrics = {}
            clinical_compliance = {}
            
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            
            for class_idx, class_label in enumerate(unique_classes):
                y_true_binary = (y_true == class_label).astype(int)
                y_pred_binary = (y_pred == class_label).astype(int)
                
                tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
                tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
                fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Valor Preditivo Positivo
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Valor Preditivo Negativo
                
                f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
                
                balanced_accuracy = (sensitivity + specificity) / 2
                
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'class_{class_label}'
                
                metrics[f'{class_name}_sensitivity'] = float(sensitivity)
                metrics[f'{class_name}_specificity'] = float(specificity)
                metrics[f'{class_name}_ppv'] = float(ppv)
                metrics[f'{class_name}_npv'] = float(npv)
                metrics[f'{class_name}_f1_score'] = float(f1)
                metrics[f'{class_name}_balanced_accuracy'] = float(balanced_accuracy)
                
                clinical_category = self._get_clinical_category(class_name)
                meets_clinical_standards = self._validate_clinical_standards(
                    class_name, sensitivity, specificity
                )
                
                clinical_compliance[class_name] = {
                    'category': clinical_category,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'meets_standards': meets_clinical_standards,
                    'risk_level': self._assess_clinical_risk(class_name, sensitivity, specificity)
                }
            
            metrics['clinical_compliance'] = clinical_compliance
            metrics['overall_clinical_readiness'] = all(
                comp['meets_standards'] for comp in clinical_compliance.values()
            )
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de métricas clínicas: {e}")
            return {}
    
    def _get_clinical_category(self, class_name: str) -> str:
        """Determina a categoria clínica da condição"""
        class_lower = class_name.lower()
        
        if any(crit in class_lower for crit in self.critical_conditions):
            return "CRITICAL"
        elif any(mod in class_lower for mod in self.moderate_conditions):
            return "MODERATE"
        else:
            return "STANDARD"
    
    def _validate_clinical_standards(self, class_name: str, sensitivity: float, specificity: float) -> bool:
        """
        Valida se as métricas atendem aos padrões clínicos do scientific guide
        """
        category = self._get_clinical_category(class_name)
        
        if category == "CRITICAL":
            return sensitivity >= 0.95 and specificity >= self.target_specificity
        elif category == "MODERATE":
            return sensitivity >= 0.90 and specificity >= self.target_specificity
        else:
            return sensitivity >= 0.80 and specificity >= 0.85
    
    def _assess_clinical_risk(self, class_name: str, sensitivity: float, specificity: float) -> str:
        """Avalia o risco clínico baseado nas métricas"""
        category = self._get_clinical_category(class_name)
        
        if category == "CRITICAL":
            if sensitivity < 0.95:
                return "HIGH_RISK"  # Falsos negativos em condições críticas
            elif specificity < 0.90:
                return "MODERATE_RISK"  # Muitos falsos positivos
            else:
                return "LOW_RISK"
        elif category == "MODERATE":
            if sensitivity < 0.90:
                return "MODERATE_RISK"
            elif specificity < 0.90:
                return "MODERATE_RISK"
            else:
                return "LOW_RISK"
        else:
            if sensitivity < 0.80 or specificity < 0.85:
                return "MODERATE_RISK"
            else:
                return "LOW_RISK"
    
    def generate_clinical_report(self, metrics: Dict, model_name: str = "MedAI Model") -> str:
        """
        Gera relatório clínico de performance baseado no scientific guide
        Inclui análise de criticidade clínica e compliance com padrões médicos
        """
        try:
            report = f"""# Relatório de Avaliação Clínica - {model_name}

- **Acurácia Geral**: {metrics.get('accuracy', 0):.2%}
- **Precisão Média**: {metrics.get('precision', 0):.2%}
- **Sensibilidade Média**: {metrics.get('recall', 0):.2%}
- **F1-Score Médio**: {metrics.get('f1_score', 0):.2%}
- **AUC-ROC**: {metrics.get('auc_roc', 'N/A')}
- **Prontidão Clínica**: {'✅ APROVADO' if metrics.get('overall_clinical_readiness', False) else '❌ REQUER MELHORIAS'}


"""
            
            # Análise de compliance clínica
            clinical_compliance = metrics.get('clinical_compliance', {})
            
            if clinical_compliance:
                critical_classes = [name for name, data in clinical_compliance.items() 
                                  if data['category'] == 'CRITICAL']
                if critical_classes:
                    report += "### Condições Críticas (Sensibilidade > 95%)\n"
                    for class_name in critical_classes:
                        data = clinical_compliance[class_name]
                        status = "✅" if data['meets_standards'] else "❌"
                        risk = data['risk_level']
                        report += f"- **{class_name}**: {status} Sensibilidade {data['sensitivity']:.2%} | Especificidade {data['specificity']:.2%} | Risco: {risk}\n"
                
                moderate_classes = [name for name, data in clinical_compliance.items() 
                                  if data['category'] == 'MODERATE']
                if moderate_classes:
                    report += "\n### Condições Moderadas (Sensibilidade > 90%)\n"
                    for class_name in moderate_classes:
                        data = clinical_compliance[class_name]
                        status = "✅" if data['meets_standards'] else "⚠️"
                        risk = data['risk_level']
                        report += f"- **{class_name}**: {status} Sensibilidade {data['sensitivity']:.2%} | Especificidade {data['specificity']:.2%} | Risco: {risk}\n"
                
                standard_classes = [name for name, data in clinical_compliance.items() 
                                  if data['category'] == 'STANDARD']
                if standard_classes:
                    report += "\n### Condições Padrão (Sensibilidade > 80%)\n"
                    for class_name in standard_classes:
                        data = clinical_compliance[class_name]
                        status = "✅" if data['meets_standards'] else "⚠️"
                        risk = data['risk_level']
                        report += f"- **{class_name}**: {status} Sensibilidade {data['sensitivity']:.2%} | Especificidade {data['specificity']:.2%} | Risco: {risk}\n"
            
            report += "\n## Métricas Detalhadas por Classe\n\n"
            report += "| Classe | Sensibilidade | Especificidade | PPV | NPV | F1-Score | Acurácia Balanceada |\n"
            report += "|--------|---------------|----------------|-----|-----|----------|--------------------|\n"
            
            for class_name in self.class_names:
                sensitivity = metrics.get(f'{class_name}_sensitivity', 0)
                specificity = metrics.get(f'{class_name}_specificity', 0)
                ppv = metrics.get(f'{class_name}_ppv', 0)
                npv = metrics.get(f'{class_name}_npv', 0)
                f1 = metrics.get(f'{class_name}_f1_score', 0)
                balanced_acc = metrics.get(f'{class_name}_balanced_accuracy', 0)
                
                report += f"| {class_name} | {sensitivity:.2%} | {specificity:.2%} | {ppv:.2%} | {npv:.2%} | {f1:.2%} | {balanced_acc:.2%} |\n"
            
            # Recomendações clínicas
            report += "\n## Recomendações Clínicas\n\n"
            
            if metrics.get('overall_clinical_readiness', False):
                report += "✅ **Modelo aprovado para uso clínico** com as seguintes considerações:\n"
                report += "- Manter supervisão médica adequada\n"
                report += "- Realizar validação prospectiva em ambiente clínico\n"
                report += "- Monitorar performance continuamente\n"
            else:
                report += "❌ **Modelo requer melhorias** antes do uso clínico:\n"
                
                for class_name, data in clinical_compliance.items():
                    if not data['meets_standards']:
                        category = data['category']
                        sensitivity = data['sensitivity']
                        specificity = data['specificity']
                        
                        if category == "CRITICAL" and sensitivity < 0.95:
                            report += f"- 🔴 **{class_name}**: Aumentar sensibilidade para > 95% (atual: {sensitivity:.2%}) - Condição crítica\n"
                        elif category == "MODERATE" and sensitivity < 0.90:
                            report += f"- 🟡 **{class_name}**: Aumentar sensibilidade para > 90% (atual: {sensitivity:.2%}) - Condição moderada\n"
                        
                        if specificity < self.target_specificity:
                            report += f"- 🟡 **{class_name}**: Melhorar especificidade para > 90% (atual: {specificity:.2%}) - Reduzir falsos positivos\n"
            
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= 0.95:
                report += "\n📊 **Performance Geral**: EXCELENTE para uso clínico\n"
            elif accuracy >= 0.90:
                report += "\n📊 **Performance Geral**: BOA para uso clínico com supervisão\n"
            elif accuracy >= 0.80:
                report += "\n📊 **Performance Geral**: MODERADA - requer validação adicional\n"
            else:
                report += "\n📊 **Performance Geral**: BAIXA - não recomendado para uso clínico\n"
            
            report += f"\n---\n**Data do Relatório**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"**Padrões Aplicados**: Scientific Guide for Medical AI Validation\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Erro na geração do relatório clínico: {e}")
            return f"Erro na geração do relatório: {e}"
    
    def calculate_roc_auc_analysis(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Calcula análise ROC-AUC para cada classe patológica
        Baseado no scientific guide para validação clínica
        """
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
            
            roc_analysis = {
                'per_class_analysis': {},
                'overall_performance': {},
                'clinical_recommendations': []
            }
            
            all_auc_scores = []
            
            for i, class_name in enumerate(self.class_names):
                if i >= len(np.unique(y_true)):
                    continue
                    
                y_true_binary = (y_true == i).astype(int)
                y_prob_class = y_prob[:, i] if y_prob.ndim > 1 else y_prob
                
                fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_prob_class)
                precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_prob_class)
                
                auc_roc = auc(fpr, tpr)
                auc_pr = auc(recall, precision)
                
                all_auc_scores.append(auc_roc)
                
                clinical_category = self._get_clinical_category(class_name)
                performance_level = self._classify_auc_performance(auc_roc)
                
                roc_analysis['per_class_analysis'][class_name] = {
                    'auc_roc': auc_roc,
                    'auc_pr': auc_pr,
                    'clinical_category': clinical_category,
                    'performance_level': performance_level,
                    'clinical_interpretation': self._interpret_auc_clinically(class_name, auc_roc)
                }
                
                if auc_roc < 0.80:
                    roc_analysis['clinical_recommendations'].append(
                        f"🔴 {class_name}: AUC-ROC {auc_roc:.3f} < 0.80 - Modelo inadequado para uso clínico"
                    )
                elif auc_roc < 0.90 and clinical_category == "CRITICAL":
                    roc_analysis['clinical_recommendations'].append(
                        f"🟡 {class_name}: AUC-ROC {auc_roc:.3f} < 0.90 - Condição crítica requer maior precisão"
                    )
            
            if all_auc_scores:
                roc_analysis['overall_performance'] = {
                    'mean_auc_roc': np.mean(all_auc_scores),
                    'min_auc_roc': np.min(all_auc_scores),
                    'max_auc_roc': np.max(all_auc_scores),
                    'std_auc_roc': np.std(all_auc_scores),
                    'clinical_readiness': np.mean(all_auc_scores) >= 0.85
                }
            
            return roc_analysis
            
        except Exception as e:
            logger.error(f"Erro na análise ROC-AUC: {e}")
            return {'error': str(e)}
    
    def _classify_auc_performance(self, auc_score: float) -> str:
        """Classifica performance baseada no AUC-ROC"""
        if auc_score >= 0.95:
            return "EXCELLENT"
        elif auc_score >= 0.90:
            return "GOOD"
        elif auc_score >= 0.80:
            return "FAIR"
        elif auc_score >= 0.70:
            return "POOR"
        else:
            return "FAIL"
    
    def _interpret_auc_clinically(self, class_name: str, auc_score: float) -> str:
        """Interpreta AUC-ROC no contexto clínico"""
        category = self._get_clinical_category(class_name)
        
        if category == "CRITICAL":
            if auc_score >= 0.95:
                return "Excelente discriminação - Adequado para condição crítica"
            elif auc_score >= 0.90:
                return "Boa discriminação - Aceitável com monitoramento"
            else:
                return "Discriminação insuficiente - Inadequado para condição crítica"
        elif category == "MODERATE":
            if auc_score >= 0.90:
                return "Excelente discriminação - Adequado para uso clínico"
            elif auc_score >= 0.80:
                return "Boa discriminação - Aceitável com supervisão"
            else:
                return "Discriminação insuficiente - Requer melhorias"
        else:
            if auc_score >= 0.80:
                return "Discriminação adequada para uso clínico"
            else:
                return "Discriminação insuficiente - Não recomendado"
    
    def generate_confidence_based_recommendation(self, pred_class: int, confidence: float, class_name: str = None) -> str:
        """
        Gera recomendações baseadas na confiança do diagnóstico
        Implementa sistema de suporte à decisão clínica do scientific guide
        """
        try:
            if class_name is None:
                class_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f'class_{pred_class}'
            
            clinical_category = self._get_clinical_category(class_name)
            
            if clinical_category == "CRITICAL":
                high_confidence_threshold = 0.90
                moderate_confidence_threshold = 0.80
            elif clinical_category == "MODERATE":
                high_confidence_threshold = 0.85
                moderate_confidence_threshold = 0.75
            else:
                high_confidence_threshold = 0.80
                moderate_confidence_threshold = 0.70
            
            if confidence >= high_confidence_threshold:
                if class_name.lower() == 'normal':
                    return f"✅ **Alta Confiança ({confidence:.1%})**: Exame dentro dos padrões normais. Manter acompanhamento de rotina."
                else:
                    return f"🔴 **Alta Confiança ({confidence:.1%})**: {class_name} identificado. Encaminhar URGENTEMENTE para especialista."
            
            elif confidence >= moderate_confidence_threshold:
                if class_name.lower() == 'normal':
                    return f"🟡 **Confiança Moderada ({confidence:.1%})**: Provável normalidade. Considerar correlação clínica e histórico do paciente."
                else:
                    return f"🟡 **Confiança Moderada ({confidence:.1%})**: Possível {class_name}. Recomenda-se avaliação por especialista e exames complementares."
            
            else:
                return f"⚠️ **Baixa Confiança ({confidence:.1%})**: Resultado inconclusivo. Recomenda-se revisão por especialista, exames adicionais e correlação clínica."
            
        except Exception as e:
            logger.error(f"Erro na geração de recomendação: {e}")
            return "Erro na geração de recomendação clínica. Consultar especialista."

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
