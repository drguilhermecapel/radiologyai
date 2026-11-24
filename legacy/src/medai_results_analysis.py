# results_analysis.py - Sistema de análise de resultados e métricas médicas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
)
from scipy import stats
import statsmodels.stats.api as sms
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

logger = logging.getLogger('MedAI.ResultsAnalysis')

@dataclass
class ClinicalMetrics:
    """Métricas clínicas para avaliação médica"""
    sensitivity: float  # True Positive Rate (Recall)
    specificity: float  # True Negative Rate
    ppv: float  # Positive Predictive Value (Precision)
    npv: float  # Negative Predictive Value
    accuracy: float
    balanced_accuracy: float
    f1_score: float
    matthews_correlation: float
    cohen_kappa: float
    youden_index: float  # Sensitivity + Specificity - 1
    diagnostic_odds_ratio: float
    likelihood_ratio_positive: float
    likelihood_ratio_negative: float
    prevalence: float
    number_needed_to_diagnose: float
    
@dataclass
class ConfidenceInterval:
    """Intervalo de confiança"""
    mean: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    
@dataclass
class ModelPerformance:
    """Performance completa do modelo"""
    clinical_metrics: ClinicalMetrics
    confidence_intervals: Dict[str, ConfidenceInterval]
    roc_analysis: Dict[str, Any]
    pr_analysis: Dict[str, Any]
    calibration_metrics: Dict[str, float]
    reliability_metrics: Dict[str, float]
    subgroup_analysis: Optional[Dict[str, ClinicalMetrics]] = None
    
class MedicalMetricsAnalyzer:
    """
    Analisador de métricas médicas e performance de modelos
    Focado em métricas clinicamente relevantes e análise estatística
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 bootstrap_iterations: int = 1000):
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        
    def analyze_predictions(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_prob: np.ndarray,
                          class_names: List[str],
                          subgroups: Optional[Dict[str, np.ndarray]] = None) -> ModelPerformance:
        """
        Análise completa das predições do modelo
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições do modelo
            y_prob: Probabilidades preditas
            class_names: Nomes das classes
            subgroups: Dicionário com máscaras para análise de subgrupos
            
        Returns:
            Performance completa do modelo
        """
        logger.info("Iniciando análise de predições")
        
        # Calcular métricas clínicas
        clinical_metrics = self._calculate_clinical_metrics(y_true, y_pred)
        
        # Calcular intervalos de confiança
        confidence_intervals = self._calculate_confidence_intervals(
            y_true, y_pred, y_prob
        )
        
        # Análise ROC
        roc_analysis = self._perform_roc_analysis(y_true, y_prob, class_names)
        
        # Análise Precision-Recall
        pr_analysis = self._perform_pr_analysis(y_true, y_prob, class_names)
        
        # Métricas de calibração
        calibration_metrics = self._calculate_calibration_metrics(y_true, y_prob)
        
        # Métricas de confiabilidade
        reliability_metrics = self._calculate_reliability_metrics(y_true, y_prob)
        
        # Análise de subgrupos se fornecida
        subgroup_results = None
        if subgroups:
            subgroup_results = self._perform_subgroup_analysis(
                y_true, y_pred, subgroups
            )
        
        return ModelPerformance(
            clinical_metrics=clinical_metrics,
            confidence_intervals=confidence_intervals,
            roc_analysis=roc_analysis,
            pr_analysis=pr_analysis,
            calibration_metrics=calibration_metrics,
            reliability_metrics=reliability_metrics,
            subgroup_analysis=subgroup_results
        )
    
    def _calculate_clinical_metrics(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> ClinicalMetrics:
        """Calcula métricas clínicas relevantes"""
        # Matriz de confusão
        if len(np.unique(y_true)) == 2:
            # Binário
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Métricas básicas
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Métricas avançadas
            prevalence = (tp + fn) / (tp + tn + fp + fn)
            
            # Likelihood ratios
            lr_pos = sensitivity / (1 - specificity) if specificity < 1 else np.inf
            lr_neg = (1 - sensitivity) / specificity if specificity > 0 else 0
            
            # Diagnostic Odds Ratio
            if fp > 0 and fn > 0:
                dor = (tp * tn) / (fp * fn)
            else:
                dor = np.inf if fp == 0 or fn == 0 else 0
            
            # Number Needed to Diagnose
            nnd = 1 / (sensitivity - (1 - specificity)) if sensitivity > (1 - specificity) else np.inf
            
        else:
            # Multiclasse - calcular média
            cm = confusion_matrix(y_true, y_pred)
            
            # Calcular para cada classe e fazer média
            sensitivities = []
            specificities = []
            ppvs = []
            npvs = []
            
            for i in range(len(cm)):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp
                
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv_i = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv_i = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                sensitivities.append(sens)
                specificities.append(spec)
                ppvs.append(ppv_i)
                npvs.append(npv_i)
            
            sensitivity = np.mean(sensitivities)
            specificity = np.mean(specificities)
            ppv = np.mean(ppvs)
            npv = np.mean(npvs)
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            prevalence = 1 / len(cm)  # Assumir distribuição uniforme
            
            # Para multiclasse, usar valores médios simplificados
            lr_pos = sensitivity / (1 - specificity) if specificity < 1 else np.inf
            lr_neg = (1 - sensitivity) / specificity if specificity > 0 else 0
            dor = lr_pos / lr_neg if lr_neg > 0 else np.inf
            nnd = 1 / (sensitivity - (1 - specificity)) if sensitivity > (1 - specificity) else np.inf
        
        # Métricas gerais
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # F1 Score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Youden's Index
        youden = sensitivity + specificity - 1
        
        return ClinicalMetrics(
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            f1_score=f1,
            matthews_correlation=mcc,
            cohen_kappa=kappa,
            youden_index=youden,
            diagnostic_odds_ratio=dor,
            likelihood_ratio_positive=lr_pos,
            likelihood_ratio_negative=lr_neg,
            prevalence=prevalence,
            number_needed_to_diagnose=nnd
        )
    
    def _calculate_confidence_intervals(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      y_prob: np.ndarray) -> Dict[str, ConfidenceInterval]:
        """Calcula intervalos de confiança usando bootstrap"""
        metrics_ci = {}
        
        # Função para calcular métricas
        def calculate_metrics(y_t, y_p):
            cm = confusion_matrix(y_t, y_p)
            if len(cm) == 2:
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                acc = (tp + tn) / (tp + tn + fp + fn)
                return {'sensitivity': sens, 'specificity': spec, 'accuracy': acc}
            else:
                acc = np.sum(np.diag(cm)) / np.sum(cm)
                return {'accuracy': acc}
        
        # Bootstrap
        n_samples = len(y_true)
        bootstrap_metrics = {metric: [] for metric in ['sensitivity', 'specificity', 'accuracy']}
        
        for _ in range(self.bootstrap_iterations):
            # Resample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calcular métricas
            metrics = calculate_metrics(y_true_boot, y_pred_boot)
            for metric, value in metrics.items():
                bootstrap_metrics[metric].append(value)
        
        # Calcular intervalos de confiança
        alpha = 1 - self.confidence_level
        for metric, values in bootstrap_metrics.items():
            if values:
                values = np.array(values)
                mean_val = np.mean(values)
                lower = np.percentile(values, alpha/2 * 100)
                upper = np.percentile(values, (1 - alpha/2) * 100)
                
                metrics_ci[metric] = ConfidenceInterval(
                    mean=mean_val,
                    lower=lower,
                    upper=upper,
                    confidence_level=self.confidence_level
                )
        
        # Adicionar IC para AUC
        if len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_auc_score
            auc_scores = []
            
            for _ in range(self.bootstrap_iterations):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                try:
                    auc_boot = roc_auc_score(y_true[indices], y_prob[indices][:, 1])
                    auc_scores.append(auc_boot)
                except:
                    pass
            
            if auc_scores:
                metrics_ci['auc'] = ConfidenceInterval(
                    mean=np.mean(auc_scores),
                    lower=np.percentile(auc_scores, alpha/2 * 100),
                    upper=np.percentile(auc_scores, (1 - alpha/2) * 100),
                    confidence_level=self.confidence_level
                )
        
        return metrics_ci
    
    def _perform_roc_analysis(self,
                            y_true: np.ndarray,
                            y_prob: np.ndarray,
                            class_names: List[str]) -> Dict[str, Any]:
        """Análise ROC detalhada"""
        roc_data = {}
        
        n_classes = len(class_names)
        
        if n_classes == 2:
            # Binário
            fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            # Encontrar ponto ótimo (Youden)
            youden_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[youden_idx]
            
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': roc_auc,
                'optimal_threshold': optimal_threshold,
                'optimal_sensitivity': tpr[youden_idx],
                'optimal_specificity': 1 - fpr[youden_idx]
            }
            
        else:
            # Multiclasse - One-vs-Rest
            from sklearn.preprocessing import label_binarize
            
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            roc_data['per_class'] = {}
            
            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                
                roc_data['per_class'][class_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc
                }
            
            # Micro-average
            fpr_micro, tpr_micro, _ = roc_curve(
                y_true_bin.ravel(), 
                y_prob.ravel()
            )
            roc_data['micro_average'] = {
                'fpr': fpr_micro.tolist(),
                'tpr': tpr_micro.tolist(),
                'auc': auc(fpr_micro, tpr_micro)
            }
            
            # Macro-average
            roc_data['macro_average_auc'] = np.mean([
                roc_data['per_class'][c]['auc'] for c in class_names
            ])
        
        return roc_data
    
    def _perform_pr_analysis(self,
                           y_true: np.ndarray,
                           y_prob: np.ndarray,
                           class_names: List[str]) -> Dict[str, Any]:
        """Análise Precision-Recall detalhada"""
        pr_data = {}
        
        n_classes = len(class_names)
        
        if n_classes == 2:
            # Binário
            precision, recall, thresholds = precision_recall_curve(
                y_true, y_prob[:, 1]
            )
            avg_precision = average_precision_score(y_true, y_prob[:, 1])
            
            # F1 score para cada threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_f1_idx = np.argmax(f1_scores[:-1])  # Último elemento é artificial
            
            pr_data = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'average_precision': avg_precision,
                'best_f1_score': f1_scores[best_f1_idx],
                'best_f1_threshold': thresholds[best_f1_idx],
                'best_f1_precision': precision[best_f1_idx],
                'best_f1_recall': recall[best_f1_idx]
            }
            
        else:
            # Multiclasse
            from sklearn.preprocessing import label_binarize
            
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            pr_data['per_class'] = {}
            
            for i, class_name in enumerate(class_names):
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], 
                    y_prob[:, i]
                )
                avg_precision = average_precision_score(
                    y_true_bin[:, i], 
                    y_prob[:, i]
                )
                
                pr_data['per_class'][class_name] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'average_precision': avg_precision
                }
            
            # Micro-average
            precision_micro, recall_micro, _ = precision_recall_curve(
                y_true_bin.ravel(),
                y_prob.ravel()
            )
            pr_data['micro_average'] = {
                'precision': precision_micro.tolist(),
                'recall': recall_micro.tolist(),
                'average_precision': average_precision_score(
                    y_true_bin.ravel(),
                    y_prob.ravel(),
                    average="micro"
                )
            }
        
        return pr_data
    
    def _calculate_calibration_metrics(self,
                                     y_true: np.ndarray,
                                     y_prob: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de calibração do modelo"""
        from sklearn.calibration import calibration_curve
        
        calibration_metrics = {}
        
        if len(np.unique(y_true)) == 2:
            # Binário
            fraction_pos, mean_pred = calibration_curve(
                y_true, y_prob[:, 1], n_bins=10
            )
            
            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(fraction_pos - mean_pred))
            
            # Maximum Calibration Error (MCE)
            mce = np.max(np.abs(fraction_pos - mean_pred))
            
            # Brier Score
            from sklearn.metrics import brier_score_loss
            brier = brier_score_loss(y_true, y_prob[:, 1])
            
            calibration_metrics = {
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'brier_score': brier,
                'fraction_positives': fraction_pos.tolist(),
                'mean_predicted_prob': mean_pred.tolist()
            }
        
        return calibration_metrics
    
    def _calculate_reliability_metrics(self,
                                     y_true: np.ndarray,
                                     y_prob: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de confiabilidade e incerteza"""
        # Entropia das predições
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-10), axis=1)
        
        # Máxima probabilidade
        max_prob = np.max(y_prob, axis=1)
        
        # Margem (diferença entre top-2)
        sorted_probs = np.sort(y_prob, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        reliability_metrics = {
            'mean_entropy': float(np.mean(entropy)),
            'std_entropy': float(np.std(entropy)),
            'mean_max_probability': float(np.mean(max_prob)),
            'std_max_probability': float(np.std(max_prob)),
            'mean_margin': float(np.mean(margin)),
            'std_margin': float(np.std(margin)),
            'low_confidence_ratio': float(np.mean(max_prob < 0.5)),
            'high_confidence_ratio': float(np.mean(max_prob > 0.9))
        }
        
        return reliability_metrics
    
    def _perform_subgroup_analysis(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 subgroups: Dict[str, np.ndarray]) -> Dict[str, ClinicalMetrics]:
        """Análise de performance em subgrupos"""
        subgroup_results = {}
        
        for subgroup_name, mask in subgroups.items():
            if np.sum(mask) > 10:  # Mínimo de amostras
                y_true_sub = y_true[mask]
                y_pred_sub = y_pred[mask]
                
                metrics = self._calculate_clinical_metrics(y_true_sub, y_pred_sub)
                subgroup_results[subgroup_name] = metrics
            else:
                logger.warning(f"Subgrupo {subgroup_name} tem poucas amostras: {np.sum(mask)}")
        
        return subgroup_results
    
    def generate_clinical_report(self,
                               performance: ModelPerformance,
                               output_path: str,
                               include_plots: bool = True):
        """
        Gera relatório clínico detalhado
        
        Args:
            performance: Resultados da análise
            output_path: Caminho para salvar o relatório
            include_plots: Se deve incluir gráficos
        """
        # Criar figura com subplots
        if include_plots:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Curva ROC', 
                    'Curva Precision-Recall',
                    'Matriz de Confusão', 
                    'Calibração do Modelo',
                    'Distribuição de Confiança', 
                    'Análise de Subgrupos'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "scatter"}],
                    [{"type": "box"}, {"type": "bar"}]
                ]
            )
            
            # 1. Curva ROC
            if 'fpr' in performance.roc_analysis:
                fig.add_trace(
                    go.Scatter(
                        x=performance.roc_analysis['fpr'],
                        y=performance.roc_analysis['tpr'],
                        mode='lines',
                        name=f"ROC (AUC = {performance.roc_analysis['auc']:.3f})",
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # Linha diagonal
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # 2. Curva PR
            if 'precision' in performance.pr_analysis:
                fig.add_trace(
                    go.Scatter(
                        x=performance.pr_analysis['recall'],
                        y=performance.pr_analysis['precision'],
                        mode='lines',
                        name=f"PR (AP = {performance.pr_analysis['average_precision']:.3f})",
                        line=dict(color='green', width=2)
                    ),
                    row=1, col=2
                )
            
            # 3. Matriz de Confusão (placeholder - precisa dos dados originais)
            # Seria implementado com os dados reais
            
            # 4. Calibração
            if 'fraction_positives' in performance.calibration_metrics:
                fig.add_trace(
                    go.Scatter(
                        x=performance.calibration_metrics['mean_predicted_prob'],
                        y=performance.calibration_metrics['fraction_positives'],
                        mode='markers+lines',
                        name='Calibração',
                        marker=dict(size=8)
                    ),
                    row=2, col=2
                )
                
                # Linha perfeita
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Atualizar layout
            fig.update_xaxes(title_text="Taxa de Falso Positivo", row=1, col=1)
            fig.update_yaxes(title_text="Taxa de Verdadeiro Positivo", row=1, col=1)
            
            fig.update_xaxes(title_text="Recall", row=1, col=2)
            fig.update_yaxes(title_text="Precision", row=1, col=2)
            
            fig.update_xaxes(title_text="Probabilidade Média Predita", row=2, col=2)
            fig.update_yaxes(title_text="Fração de Positivos", row=2, col=2)
            
            fig.update_layout(
                height=1200,
                title_text="Análise de Performance do Modelo",
                showlegend=True
            )
            
            # Salvar figura
            fig.write_html(output_path.replace('.pdf', '_plots.html'))
        
        # Gerar relatório em texto
        report = self._generate_text_report(performance)
        
        # Salvar relatório
        report_path = Path(output_path).with_suffix('.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Relatório clínico salvo em: {report_path}")
    
    def _generate_text_report(self, performance: ModelPerformance) -> str:
        """Gera relatório em texto com interpretação clínica"""
        metrics = performance.clinical_metrics
        ci = performance.confidence_intervals
        
        report = f"""
RELATÓRIO DE ANÁLISE DE PERFORMANCE - MODELO DE IA MÉDICA
========================================================
Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}

1. MÉTRICAS DE PERFORMANCE DIAGNÓSTICA
--------------------------------------
Sensibilidade (Recall): {metrics.sensitivity:.3f} {self._format_ci(ci.get('sensitivity'))}
   - Capacidade de identificar corretamente casos positivos
   - {self._interpret_sensitivity(metrics.sensitivity)}

Especificidade: {metrics.specificity:.3f} {self._format_ci(ci.get('specificity'))}
   - Capacidade de identificar corretamente casos negativos
   - {self._interpret_specificity(metrics.specificity)}

Valor Preditivo Positivo (Precisão): {metrics.ppv:.3f}
   - Probabilidade de um resultado positivo ser verdadeiro
   - {self._interpret_ppv(metrics.ppv, metrics.prevalence)}

Valor Preditivo Negativo: {metrics.npv:.3f}
   - Probabilidade de um resultado negativo ser verdadeiro
   - {self._interpret_npv(metrics.npv, metrics.prevalence)}

Acurácia Global: {metrics.accuracy:.3f} {self._format_ci(ci.get('accuracy'))}
Acurácia Balanceada: {metrics.balanced_accuracy:.3f}

2. MÉTRICAS DE CONCORDÂNCIA E CONFIABILIDADE
-------------------------------------------
F1-Score: {metrics.f1_score:.3f}
   - Média harmônica entre precisão e recall

Coeficiente de Matthews (MCC): {metrics.matthews_correlation:.3f}
   - Medida de qualidade geral (-1 a +1)
   - {self._interpret_mcc(metrics.matthews_correlation)}

Kappa de Cohen: {metrics.cohen_kappa:.3f}
   - Concordância além do acaso
   - {self._interpret_kappa(metrics.cohen_kappa)}

Índice de Youden: {metrics.youden_index:.3f}
   - Performance diagnóstica geral
   - {self._interpret_youden(metrics.youden_index)}

3. RAZÕES DE VEROSSIMILHANÇA E UTILIDADE CLÍNICA
-----------------------------------------------
Razão de Verossimilhança Positiva (LR+): {metrics.likelihood_ratio_positive:.2f}
   - {self._interpret_lr_positive(metrics.likelihood_ratio_positive)}

Razão de Verossimilhança Negativa (LR-): {metrics.likelihood_ratio_negative:.2f}
   - {self._interpret_lr_negative(metrics.likelihood_ratio_negative)}

Odds Ratio Diagnóstico (DOR): {metrics.diagnostic_odds_ratio:.2f}
   - Razão entre odds de verdadeiros e falsos resultados

Number Needed to Diagnose (NND): {metrics.number_needed_to_diagnose:.1f}
   - Número de pacientes a examinar para um diagnóstico correto adicional

4. ANÁLISE ROC E DISCRIMINAÇÃO
-----------------------------
AUC-ROC: {performance.roc_analysis.get('auc', 'N/A'):.3f} {self._format_ci(ci.get('auc'))}
   - Capacidade discriminativa do modelo
   - {self._interpret_auc(performance.roc_analysis.get('auc', 0))}

Limiar Ótimo: {performance.roc_analysis.get('optimal_threshold', 'N/A'):.3f}
Sensibilidade no Limiar Ótimo: {performance.roc_analysis.get('optimal_sensitivity', 'N/A'):.3f}
Especificidade no Limiar Ótimo: {performance.roc_analysis.get('optimal_specificity', 'N/A'):.3f}

5. CALIBRAÇÃO E CONFIABILIDADE
-----------------------------
Erro de Calibração Esperado (ECE): {performance.calibration_metrics.get('expected_calibration_error', 'N/A'):.3f}
Erro de Calibração Máximo (MCE): {performance.calibration_metrics.get('maximum_calibration_error', 'N/A'):.3f}
Brier Score: {performance.calibration_metrics.get('brier_score', 'N/A'):.3f}

Entropia Média das Predições: {performance.reliability_metrics.get('mean_entropy', 'N/A'):.3f}
Proporção de Baixa Confiança (<50%): {performance.reliability_metrics.get('low_confidence_ratio', 'N/A'):.2%}
Proporção de Alta Confiança (>90%): {performance.reliability_metrics.get('high_confidence_ratio', 'N/A'):.2%}

6. INTERPRETAÇÃO CLÍNICA
-----------------------
{self._generate_clinical_interpretation(metrics)}

7. RECOMENDAÇÕES
---------------
{self._generate_recommendations(performance)}

8. LIMITAÇÕES E CONSIDERAÇÕES
----------------------------
- Este modelo deve ser usado como ferramenta auxiliar de diagnóstico
- A decisão final deve sempre ser tomada por um profissional médico qualificado
- Performance pode variar em diferentes populações e contextos clínicos
- Recomenda-se validação externa antes do uso clínico

"""
        
        # Adicionar análise de subgrupos se disponível
        if performance.subgroup_analysis:
            report += "\n9. ANÁLISE DE SUBGRUPOS\n"
            report += "-----------------------\n"
            for subgroup, metrics in performance.subgroup_analysis.items():
                report += f"\n{subgroup}:\n"
                report += f"  Sensibilidade: {metrics.sensitivity:.3f}\n"
                report += f"  Especificidade: {metrics.specificity:.3f}\n"
                report += f"  Acurácia: {metrics.accuracy:.3f}\n"
        
        return report
    
    def _format_ci(self, ci: Optional[ConfidenceInterval]) -> str:
        """Formata intervalo de confiança"""
        if ci:
            return f"(IC {ci.confidence_level:.0%}: {ci.lower:.3f}-{ci.upper:.3f})"
        return ""
    
    def _interpret_sensitivity(self, value: float) -> str:
        """Interpreta valor de sensibilidade"""
        if value >= 0.95:
            return "Excelente - Muito poucos falsos negativos"
        elif value >= 0.90:
            return "Muito bom - Poucos falsos negativos"
        elif value >= 0.80:
            return "Bom - Taxa aceitável de falsos negativos"
        elif value >= 0.70:
            return "Moderado - Considerar impacto de falsos negativos"
        else:
            return "Baixo - Alto risco de falsos negativos"
    
    def _interpret_specificity(self, value: float) -> str:
        """Interpreta valor de especificidade"""
        if value >= 0.95:
            return "Excelente - Muito poucos falsos positivos"
        elif value >= 0.90:
            return "Muito bom - Poucos falsos positivos"
        elif value >= 0.80:
            return "Bom - Taxa aceitável de falsos positivos"
        elif value >= 0.70:
            return "Moderado - Considerar impacto de falsos positivos"
        else:
            return "Baixo - Alto risco de falsos positivos"
    
    def _interpret_ppv(self, ppv: float, prevalence: float) -> str:
        """Interpreta PPV considerando prevalência"""
        if ppv >= 0.90:
            return "Muito confiável para confirmar diagnóstico"
        elif ppv >= 0.70:
            return "Boa confiabilidade, mas considerar testes confirmatórios"
        else:
            adj = " (considerar baixa prevalência)" if prevalence < 0.1 else ""
            return f"Moderada confiabilidade{adj}"
    
    def _interpret_npv(self, npv: float, prevalence: float) -> str:
        """Interpreta NPV considerando prevalência"""
        if npv >= 0.95:
            return "Excelente para descartar diagnóstico"
        elif npv >= 0.90:
            return "Muito bom para descartar diagnóstico"
        else:
            adj = " (considerar prevalência)" if prevalence > 0.3 else ""
            return f"Moderado para descartar{adj}"
    
    def _interpret_mcc(self, mcc: float) -> str:
        """Interpreta Matthews Correlation Coefficient"""
        if mcc >= 0.7:
            return "Correlação forte"
        elif mcc >= 0.5:
            return "Correlação moderada"
        elif mcc >= 0.3:
            return "Correlação fraca"
        else:
            return "Correlação muito fraca"
    
    def _interpret_kappa(self, kappa: float) -> str:
        """Interpreta Kappa de Cohen"""
        if kappa >= 0.81:
            return "Concordância quase perfeita"
        elif kappa >= 0.61:
            return "Concordância substancial"
        elif kappa >= 0.41:
            return "Concordância moderada"
        elif kappa >= 0.21:
            return "Concordância razoável"
        else:
            return "Concordância fraca"
    
    def _interpret_youden(self, youden: float) -> str:
        """Interpreta índice de Youden"""
        if youden >= 0.8:
            return "Performance diagnóstica excelente"
        elif youden >= 0.6:
            return "Performance diagnóstica boa"
        elif youden >= 0.4:
            return "Performance diagnóstica moderada"
        else:
            return "Performance diagnóstica limitada"
    
    def _interpret_lr_positive(self, lr_pos: float) -> str:
        """Interpreta LR+"""
        if lr_pos > 10:
            return "Forte evidência para confirmar diagnóstico"
        elif lr_pos > 5:
            return "Moderada evidência para confirmar diagnóstico"
        elif lr_pos > 2:
            return "Fraca evidência para confirmar diagnóstico"
        else:
            return "Evidência mínima para confirmar diagnóstico"
    
    def _interpret_lr_negative(self, lr_neg: float) -> str:
        """Interpreta LR-"""
        if lr_neg < 0.1:
            return "Forte evidência para descartar diagnóstico"
        elif lr_neg < 0.2:
            return "Moderada evidência para descartar diagnóstico"
        elif lr_neg < 0.5:
            return "Fraca evidência para descartar diagnóstico"
        else:
            return "Evidência mínima para descartar diagnóstico"
    
    def _interpret_auc(self, auc: float) -> str:
        """Interpreta AUC"""
        if auc >= 0.9:
            return "Discriminação excelente"
        elif auc >= 0.8:
            return "Discriminação boa"
        elif auc >= 0.7:
            return "Discriminação aceitável"
        elif auc >= 0.6:
            return "Discriminação pobre"
        else:
            return "Discriminação falha"
    
    def _generate_clinical_interpretation(self, metrics: ClinicalMetrics) -> str:
        """Gera interpretação clínica geral"""
        interpretations = []
        
        # Cenário de triagem
        if metrics.sensitivity > 0.9 and metrics.npv > 0.95:
            interpretations.append(
                "• TRIAGEM: Modelo adequado para triagem inicial devido à alta "
                "sensibilidade e VPN. Poucos casos serão perdidos."
            )
        
        # Cenário de confirmação
        if metrics.specificity > 0.9 and metrics.ppv > 0.8:
            interpretations.append(
                "• CONFIRMAÇÃO: Modelo útil para confirmação diagnóstica devido "
                "à alta especificidade e VPP."
            )
        
        # Equilíbrio
        if 0.8 <= metrics.sensitivity <= 0.9 and 0.8 <= metrics.specificity <= 0.9:
            interpretations.append(
                "• USO GERAL: Modelo bem balanceado, adequado para uso clínico "
                "geral com supervisão médica."
            )
        
        # Limitações
        if metrics.sensitivity < 0.7:
            interpretations.append(
                "⚠ ATENÇÃO: Baixa sensibilidade - risco significativo de falsos "
                "negativos. Não recomendado para triagem."
            )
        
        if metrics.specificity < 0.7:
            interpretations.append(
                "⚠ ATENÇÃO: Baixa especificidade - alto número de falsos positivos "
                "pode gerar ansiedade e exames desnecessários."
            )
        
        return "\n".join(interpretations) if interpretations else "Interpretação pendente de mais análises."
    
    def _generate_recommendations(self, performance: ModelPerformance) -> str:
        """Gera recomendações baseadas na performance"""
        recommendations = []
        metrics = performance.clinical_metrics
        
        # Baseado na performance geral
        if metrics.sensitivity < 0.8:
            recommendations.append(
                "1. Considerar retreinamento com foco em casos difíceis/raros"
            )
            recommendations.append(
                "2. Implementar sistema de dupla leitura para casos borderline"
            )
        
        if metrics.specificity < 0.8:
            recommendations.append(
                "3. Ajustar limiar de decisão para reduzir falsos positivos"
            )
            recommendations.append(
                "4. Considerar features adicionais para melhor discriminação"
            )
        
        # Baseado na calibração
        if performance.calibration_metrics.get('expected_calibration_error', 0) > 0.1:
            recommendations.append(
                "5. Aplicar recalibração (Platt scaling ou isotonic regression)"
            )
        
        # Baseado na confiabilidade
        if performance.reliability_metrics.get('low_confidence_ratio', 0) > 0.3:
            recommendations.append(
                "6. Implementar sistema de triagem para casos de baixa confiança"
            )
        
        # Sempre incluir
        recommendations.extend([
            "7. Realizar validação externa em população independente",
            "8. Monitorar performance continuamente em produção",
            "9. Estabelecer protocolo de atualização periódica do modelo"
        ])
        
        return "\n".join(recommendations)
    
    def compare_models(self,
                      models_performance: Dict[str, ModelPerformance],
                      test_type: str = 'mcnemar') -> pd.DataFrame:
        """
        Compara estatisticamente múltiplos modelos
        
        Args:
            models_performance: Dicionário com performances dos modelos
            test_type: Tipo de teste estatístico ('mcnemar', 'delong')
            
        Returns:
            DataFrame com resultados da comparação
        """
        model_names = list(models_performance.keys())
        n_models = len(model_names)
        
        # Criar matriz de p-values
        p_values = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1 = models_performance[model_names[i]]
                model2 = models_performance[model_names[j]]
                
                # Aqui seria implementado o teste estatístico real
                # Por enquanto, placeholder
                p_value = 0.05  # Placeholder
                
                p_values[i, j] = p_value
                p_values[j, i] = p_value
        
        # Criar DataFrame
        comparison_df = pd.DataFrame(
            p_values,
            index=model_names,
            columns=model_names
        )
        
        return comparison_df
    
    def generate_latex_table(self, 
                           performance: ModelPerformance,
                           caption: str = "Performance do Modelo") -> str:
        """Gera tabela LaTeX com métricas para publicação"""
        metrics = performance.clinical_metrics
        ci = performance.confidence_intervals
        
        latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Métrica}} & \\textbf{{Valor}} & \\textbf{{IC 95\\%}} \\\\
\\hline
Sensibilidade & {metrics.sensitivity:.3f} & {self._format_ci_latex(ci.get('sensitivity'))} \\\\
Especificidade & {metrics.specificity:.3f} & {self._format_ci_latex(ci.get('specificity'))} \\\\
VPP & {metrics.ppv:.3f} & - \\\\
VPN & {metrics.npv:.3f} & - \\\\
Acurácia & {metrics.accuracy:.3f} & {self._format_ci_latex(ci.get('accuracy'))} \\\\
AUC-ROC & {performance.roc_analysis.get('auc', '-'):.3f} & {self._format_ci_latex(ci.get('auc'))} \\\\
F1-Score & {metrics.f1_score:.3f} & - \\\\
MCC & {metrics.matthews_correlation:.3f} & - \\\\
Kappa & {metrics.cohen_kappa:.3f} & - \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
        return latex_table
    
    def _format_ci_latex(self, ci: Optional[ConfidenceInterval]) -> str:
        """Formata IC para LaTeX"""
        if ci:
            return f"({ci.lower:.3f}-{ci.upper:.3f})"
        return "-"
