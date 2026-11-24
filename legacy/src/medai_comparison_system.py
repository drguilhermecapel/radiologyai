# comparison_system.py - Sistema de comparação de imagens e análise temporal

import numpy as np
import cv2
from scipy import ndimage
from skimage import metrics as skmetrics
from skimage import transform as sktransform
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import pandas as pd
import json

logger = logging.getLogger('MedAI.Comparison')

@dataclass
class ComparisonResult:
    """Resultado de comparação entre imagens"""
    similarity_score: float
    structural_similarity: float
    pixel_difference: np.ndarray
    regions_changed: List[Dict]
    metrics: Dict[str, float]
    alignment_transform: Optional[np.ndarray] = None
    
@dataclass
class TemporalAnalysis:
    """Análise temporal de uma série de exames"""
    patient_id: str
    exam_dates: List[datetime]
    predictions_timeline: List[Dict]
    progression_metrics: Dict[str, List[float]]
    trend_analysis: Dict[str, str]
    risk_score: float

class ImageComparisonSystem:
    """
    Sistema para comparação de imagens médicas e análise de progressão temporal
    Identifica mudanças, mede similaridade e rastreia evolução de condições
    """
    
    def __init__(self):
        self.registration_methods = {
            'rigid': self._rigid_registration,
            'affine': self._affine_registration,
            'elastic': self._elastic_registration
        }
        
    def compare_images(self,
                      image1: np.ndarray,
                      image2: np.ndarray,
                      metadata1: Optional[Dict] = None,
                      metadata2: Optional[Dict] = None,
                      auto_align: bool = True) -> ComparisonResult:
        """
        Compara duas imagens médicas
        
        Args:
            image1: Primeira imagem (referência)
            image2: Segunda imagem (comparação)
            metadata1: Metadados da primeira imagem
            metadata2: Metadados da segunda imagem
            auto_align: Se deve alinhar automaticamente as imagens
            
        Returns:
            Resultado da comparação
        """
        logger.info("Iniciando comparação de imagens")
        
        # Garantir mesmo tipo de dados
        img1 = image1.astype(np.float32)
        img2 = image2.astype(np.float32)
        
        # Alinhar imagens se necessário
        if auto_align and self._needs_alignment(img1, img2):
            img2, transform_matrix = self._align_images(img1, img2)
        else:
            transform_matrix = None
        
        # Redimensionar se necessário
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calcular métricas de similaridade
        metrics = self._calculate_similarity_metrics(img1, img2)
        
        # Calcular diferença de pixels
        pixel_diff = self._calculate_pixel_difference(img1, img2)
        
        # Detectar regiões com mudanças significativas
        changed_regions = self._detect_changed_regions(pixel_diff)
        
        # Calcular score geral de similaridade
        similarity_score = self._calculate_overall_similarity(metrics)
        
        return ComparisonResult(
            similarity_score=similarity_score,
            structural_similarity=metrics['ssim'],
            pixel_difference=pixel_diff,
            regions_changed=changed_regions,
            metrics=metrics,
            alignment_transform=transform_matrix
        )
    
    def _needs_alignment(self, img1: np.ndarray, img2: np.ndarray) -> bool:
        """Verifica se as imagens precisam de alinhamento"""
        # Verificar se tamanhos são muito diferentes
        if img1.shape != img2.shape:
            return True
        
        # Calcular correlação rápida
        correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
        
        # Se correlação baixa, provavelmente precisa alinhamento
        return correlation < 0.8
    
    def _align_images(self, 
                     reference: np.ndarray, 
                     moving: np.ndarray,
                     method: str = 'affine') -> Tuple[np.ndarray, np.ndarray]:
        """
        Alinha duas imagens usando registro de imagem
        
        Args:
            reference: Imagem de referência
            moving: Imagem a ser alinhada
            method: Método de registro ('rigid', 'affine', 'elastic')
            
        Returns:
            Tupla com imagem alinhada e matriz de transformação
        """
        logger.info(f"Alinhando imagens usando método: {method}")
        
        if method in self.registration_methods:
            return self.registration_methods[method](reference, moving)
        else:
            raise ValueError(f"Método de registro desconhecido: {method}")
    
    def _rigid_registration(self, 
                           reference: np.ndarray, 
                           moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Registro rígido (translação + rotação)"""
        # Detectar features
        detector = cv2.SIFT_create()
        
        kp1, des1 = detector.detectAndCompute(reference.astype(np.uint8), None)
        kp2, des2 = detector.detectAndCompute(moving.astype(np.uint8), None)
        
        # Match features
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Filtrar bons matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) >= 4:
            # Extrair pontos
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            
            # Calcular transformação rígida
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            
            # Aplicar transformação
            aligned = cv2.warpAffine(
                moving, M, 
                (reference.shape[1], reference.shape[0])
            )
            
            return aligned, M
        else:
            logger.warning("Poucos matches encontrados para alinhamento")
            return moving, np.eye(3)
    
    def _affine_registration(self,
                           reference: np.ndarray,
                           moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Registro afim (translação + rotação + escala + cisalhamento)"""
        # Usar ECC (Enhanced Correlation Coefficient)
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Definir critério de parada
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                   1000, 1e-6)
        
        try:
            # Calcular transformação
            _, warp_matrix = cv2.findTransformECC(
                reference.astype(np.float32),
                moving.astype(np.float32),
                warp_matrix,
                warp_mode,
                criteria
            )
            
            # Aplicar transformação
            aligned = cv2.warpAffine(
                moving,
                warp_matrix,
                (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            
            return aligned, warp_matrix
            
        except cv2.error:
            logger.warning("Falha no registro ECC, usando método alternativo")
            return self._rigid_registration(reference, moving)
    
    def _elastic_registration(self,
                            reference: np.ndarray,
                            moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Registro elástico (deformação não-linear)"""
        # Implementação simplificada usando optical flow
        
        # Calcular optical flow denso
        flow = cv2.calcOpticalFlowFarneback(
            reference.astype(np.uint8),
            moving.astype(np.uint8),
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Criar grid de coordenadas
        h, w = reference.shape[:2]
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                flow_map[y, x] = [x + flow[y, x, 0], y + flow[y, x, 1]]
        
        # Aplicar remapeamento
        aligned = cv2.remap(
            moving,
            flow_map,
            None,
            cv2.INTER_LINEAR
        )
        
        return aligned, flow
    
    def _calculate_similarity_metrics(self,
                                    img1: np.ndarray,
                                    img2: np.ndarray) -> Dict[str, float]:
        """Calcula várias métricas de similaridade"""
        metrics = {}
        
        # Normalizar imagens
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min())
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min())
        
        # SSIM (Structural Similarity Index)
        metrics['ssim'] = skmetrics.structural_similarity(
            img1_norm, img2_norm, data_range=1.0
        )
        
        # MSE (Mean Squared Error)
        metrics['mse'] = skmetrics.mean_squared_error(img1_norm, img2_norm)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        if metrics['mse'] > 0:
            metrics['psnr'] = skmetrics.peak_signal_noise_ratio(
                img1_norm, img2_norm, data_range=1.0
            )
        else:
            metrics['psnr'] = float('inf')
        
        # Correlação de Pearson
        metrics['correlation'] = np.corrcoef(
            img1_norm.flatten(), 
            img2_norm.flatten()
        )[0, 1]
        
        # Mutual Information
        metrics['mutual_info'] = skmetrics.normalized_mutual_information(
            img1_norm.astype(np.uint8), 
            img2_norm.astype(np.uint8)
        )
        
        # Dice Coefficient (para imagens binárias ou segmentadas)
        if len(np.unique(img1)) <= 10:  # Assumir que é segmentação
            intersection = np.logical_and(img1 > 0, img2 > 0).sum()
            dice = 2 * intersection / (np.sum(img1 > 0) + np.sum(img2 > 0))
            metrics['dice'] = dice
        
        return metrics
    
    def _calculate_pixel_difference(self,
                                  img1: np.ndarray,
                                  img2: np.ndarray) -> np.ndarray:
        """Calcula diferença absoluta entre pixels"""
        # Diferença absoluta
        diff = np.abs(img1 - img2)
        
        # Aplicar filtro gaussiano para suavizar ruído
        diff_smooth = cv2.GaussianBlur(diff, (5, 5), 1.0)
        
        return diff_smooth
    
    def _detect_changed_regions(self,
                              diff_image: np.ndarray,
                              threshold: float = 0.1) -> List[Dict]:
        """Detecta regiões com mudanças significativas"""
        # Binarizar diferença
        _, binary = cv2.threshold(
            (diff_image * 255).astype(np.uint8),
            int(threshold * 255),
            255,
            cv2.THRESH_BINARY
        )
        
        # Operações morfológicas para limpar ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analisar cada região
        regions = []
        min_area = 100  # Área mínima para considerar
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # Calcular bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calcular centróide
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Calcular intensidade média da mudança
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_change = cv2.mean(diff_image, mask=mask)[0]
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'area': area,
                    'mean_change': mean_change,
                    'contour': contour.tolist()
                })
        
        # Ordenar por área (maiores primeiro)
        regions.sort(key=lambda r: r['area'], reverse=True)
        
        return regions
    
    def _calculate_overall_similarity(self, metrics: Dict[str, float]) -> float:
        """Calcula score geral de similaridade ponderado"""
        # Pesos para cada métrica
        weights = {
            'ssim': 0.4,
            'correlation': 0.3,
            'mutual_info': 0.2,
            'mse': 0.1  # Invertido
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                if metric == 'mse':
                    # MSE é melhor quando menor
                    value = 1.0 - min(metrics[metric], 1.0)
                else:
                    value = metrics[metric]
                
                score += value * weight
                total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        
        return max(0.0, min(1.0, score))
    
    def analyze_temporal_series(self,
                              images: List[np.ndarray],
                              dates: List[datetime],
                              predictions: List[Dict],
                              patient_id: str) -> TemporalAnalysis:
        """
        Analisa série temporal de exames
        
        Args:
            images: Lista de imagens ao longo do tempo
            dates: Datas dos exames
            predictions: Predições de IA para cada exame
            patient_id: ID do paciente
            
        Returns:
            Análise temporal completa
        """
        logger.info(f"Analisando série temporal de {len(images)} exames")
        
        if len(images) < 2:
            raise ValueError("Necessário pelo menos 2 exames para análise temporal")
        
        # Ordenar por data
        sorted_indices = np.argsort(dates)
        images = [images[i] for i in sorted_indices]
        dates = [dates[i] for i in sorted_indices]
        predictions = [predictions[i] for i in sorted_indices]
        
        # Calcular métricas de progressão
        progression_metrics = self._calculate_progression_metrics(
            images, dates, predictions
        )
        
        # Analisar tendências
        trend_analysis = self._analyze_trends(progression_metrics)
        
        # Calcular score de risco
        risk_score = self._calculate_risk_score(
            progression_metrics, trend_analysis
        )
        
        return TemporalAnalysis(
            patient_id=patient_id,
            exam_dates=dates,
            predictions_timeline=predictions,
            progression_metrics=progression_metrics,
            trend_analysis=trend_analysis,
            risk_score=risk_score
        )
    
    def _calculate_progression_metrics(self,
                                     images: List[np.ndarray],
                                     dates: List[datetime],
                                     predictions: List[Dict]) -> Dict[str, List[float]]:
        """Calcula métricas de progressão ao longo do tempo"""
        metrics = {
            'volume_change': [],
            'intensity_change': [],
            'texture_change': [],
            'confidence_trend': [],
            'days_between_exams': []
        }
        
        # Primeira imagem como referência
        ref_image = images[0]
        ref_date = dates[0]
        
        for i in range(1, len(images)):
            current_image = images[i]
            current_date = dates[i]
            
            # Dias entre exames
            days_diff = (current_date - ref_date).days
            metrics['days_between_exams'].append(days_diff)
            
            # Comparar com imagem anterior
            comparison = self.compare_images(
                images[i-1], current_image, auto_align=True
            )
            
            # Volume de mudança (área total alterada)
            total_changed_area = sum(r['area'] for r in comparison.regions_changed)
            total_area = current_image.shape[0] * current_image.shape[1]
            volume_change = total_changed_area / total_area
            metrics['volume_change'].append(volume_change)
            
            # Mudança de intensidade média
            intensity_change = np.mean(comparison.pixel_difference)
            metrics['intensity_change'].append(intensity_change)
            
            # Mudança de textura (usando desvio padrão local)
            texture_ref = cv2.Laplacian(images[i-1], cv2.CV_64F).std()
            texture_cur = cv2.Laplacian(current_image, cv2.CV_64F).std()
            texture_change = abs(texture_cur - texture_ref) / texture_ref
            metrics['texture_change'].append(texture_change)
            
            # Tendência de confiança das predições
            if i < len(predictions):
                confidence = predictions[i].get('confidence', 0)
                metrics['confidence_trend'].append(confidence)
        
        return metrics
    
    def _analyze_trends(self, 
                       metrics: Dict[str, List[float]]) -> Dict[str, str]:
        """Analisa tendências nas métricas de progressão"""
        trends = {}
        
        for metric_name, values in metrics.items():
            if len(values) < 2:
                trends[metric_name] = 'insufficient_data'
                continue
            
            # Calcular tendência linear
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            
            # Calcular variação percentual
            if values[0] != 0:
                percent_change = ((values[-1] - values[0]) / values[0]) * 100
            else:
                percent_change = 0
            
            # Classificar tendência
            if abs(slope) < 0.01:
                trend = 'stable'
            elif slope > 0.05:
                trend = 'increasing_fast'
            elif slope > 0:
                trend = 'increasing_slow'
            elif slope < -0.05:
                trend = 'decreasing_fast'
            else:
                trend = 'decreasing_slow'
            
            trends[metric_name] = {
                'direction': trend,
                'slope': float(slope),
                'percent_change': float(percent_change)
            }
        
        return trends
    
    def _calculate_risk_score(self,
                            metrics: Dict[str, List[float]],
                            trends: Dict[str, Dict]) -> float:
        """Calcula score de risco baseado na progressão"""
        risk_factors = []
        
        # Fator 1: Volume de mudança crescente
        if 'volume_change' in trends:
            volume_trend = trends['volume_change']
            if volume_trend['direction'].startswith('increasing'):
                risk_factors.append(0.3 * (1 + volume_trend['slope']))
        
        # Fator 2: Intensidade de mudança alta
        if 'intensity_change' in metrics and metrics['intensity_change']:
            avg_intensity_change = np.mean(metrics['intensity_change'])
            if avg_intensity_change > 0.2:
                risk_factors.append(0.25 * avg_intensity_change)
        
        # Fator 3: Confiança decrescente
        if 'confidence_trend' in trends:
            conf_trend = trends['confidence_trend']
            if conf_trend['direction'].startswith('decreasing'):
                risk_factors.append(0.2 * abs(conf_trend['slope']))
        
        # Fator 4: Alta variabilidade
        for metric_values in metrics.values():
            if len(metric_values) > 2:
                cv = np.std(metric_values) / (np.mean(metric_values) + 1e-6)
                if cv > 0.5:
                    risk_factors.append(0.15 * cv)
        
        # Calcular score final (0-1)
        if risk_factors:
            risk_score = min(1.0, np.mean(risk_factors) * 2)
        else:
            risk_score = 0.0
        
        return risk_score
    
    def visualize_comparison(self,
                           img1: np.ndarray,
                           img2: np.ndarray,
                           comparison_result: ComparisonResult,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Visualiza comparação entre duas imagens"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Imagem 1
        axes[0, 0].imshow(img1, cmap='gray')
        axes[0, 0].set_title('Imagem de Referência')
        axes[0, 0].axis('off')
        
        # Imagem 2
        axes[0, 1].imshow(img2, cmap='gray')
        axes[0, 1].set_title('Imagem de Comparação')
        axes[0, 1].axis('off')
        
        # Diferença
        axes[0, 2].imshow(comparison_result.pixel_difference, cmap='hot')
        axes[0, 2].set_title('Mapa de Diferenças')
        axes[0, 2].axis('off')
        
        # Regiões alteradas
        overlay = img2.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        for region in comparison_result.regions_changed[:5]:  # Top 5 regiões
            x, y, w, h = region['bbox']
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Adicionar label
            label = f"{region['mean_change']:.2f}"
            cv2.putText(overlay, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Regiões Alteradas')
        axes[1, 0].axis('off')
        
        # Métricas
        metrics_text = f"""
Similaridade Geral: {comparison_result.similarity_score:.2%}
SSIM: {comparison_result.structural_similarity:.3f}
MSE: {comparison_result.metrics.get('mse', 0):.4f}
Correlação: {comparison_result.metrics.get('correlation', 0):.3f}
Regiões Alteradas: {len(comparison_result.regions_changed)}
"""
        axes[1, 1].text(0.1, 0.5, metrics_text, 
                       transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="lightgray"))
        axes[1, 1].axis('off')
        
        # Histograma de diferenças
        axes[1, 2].hist(comparison_result.pixel_difference.flatten(), 
                       bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 2].set_xlabel('Intensidade da Diferença')
        axes[1, 2].set_ylabel('Frequência')
        axes[1, 2].set_title('Distribuição das Diferenças')
        
        plt.suptitle('Análise Comparativa de Imagens Médicas', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {save_path}")
        
        return fig
    
    def generate_temporal_report(self,
                               temporal_analysis: TemporalAnalysis,
                               output_path: str):
        """Gera relatório de análise temporal"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Converter datas para números para plotagem
        dates_numeric = [(d - temporal_analysis.exam_dates[0]).days 
                        for d in temporal_analysis.exam_dates]
        
        # 1. Volume de mudança ao longo do tempo
        if 'volume_change' in temporal_analysis.progression_metrics:
            axes[0, 0].plot(dates_numeric[1:], 
                          temporal_analysis.progression_metrics['volume_change'],
                          'b-o', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Dias desde o primeiro exame')
            axes[0, 0].set_ylabel('Volume de Mudança (%)')
            axes[0, 0].set_title('Progressão do Volume de Alterações')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Tendência de confiança
        if 'confidence_trend' in temporal_analysis.progression_metrics:
            confidences = [p.get('confidence', 0) 
                          for p in temporal_analysis.predictions_timeline]
            axes[0, 1].plot(dates_numeric[:len(confidences)], 
                          confidences,
                          'g-s', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Dias desde o primeiro exame')
            axes[0, 1].set_ylabel('Confiança da Predição')
            axes[0, 1].set_title('Evolução da Confiança do Modelo')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribuição de classes ao longo do tempo
        class_timeline = []
        for pred in temporal_analysis.predictions_timeline:
            class_timeline.append(pred.get('predicted_class', 'Unknown'))
        
        unique_classes = list(set(class_timeline))
        class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        
        for i, cls in enumerate(unique_classes):
            class_dates = [dates_numeric[j] for j, c in enumerate(class_timeline) 
                          if c == cls]
            axes[1, 0].scatter(class_dates, [cls] * len(class_dates),
                             c=[class_colors[i]], s=100, label=cls)
        
        axes[1, 0].set_xlabel('Dias desde o primeiro exame')
        axes[1, 0].set_ylabel('Classe Predita')
        axes[1, 0].set_title('Timeline de Classificações')
        axes[1, 0].legend()
        
        # 4. Mudança de intensidade
        if 'intensity_change' in temporal_analysis.progression_metrics:
            axes[1, 1].plot(dates_numeric[1:],
                          temporal_analysis.progression_metrics['intensity_change'],
                          'r-^', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('Dias desde o primeiro exame')
            axes[1, 1].set_ylabel('Mudança de Intensidade Média')
            axes[1, 1].set_title('Evolução da Intensidade')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Score de risco
        risk_color = 'red' if temporal_analysis.risk_score > 0.7 else \
                    'orange' if temporal_analysis.risk_score > 0.4 else 'green'
        
        axes[2, 0].bar(['Score de Risco'], [temporal_analysis.risk_score],
                      color=risk_color, alpha=0.7)
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].set_ylabel('Score')
        axes[2, 0].set_title(f'Score de Risco: {temporal_analysis.risk_score:.2f}')
        
        # Adicionar linha de referência
        axes[2, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        axes[2, 0].axhline(y=0.4, color='orange', linestyle='--', alpha=0.5)
        
        # 6. Resumo de tendências
        trend_text = "Resumo de Tendências:\n\n"
        for metric, trend_info in temporal_analysis.trend_analysis.items():
            if isinstance(trend_info, dict):
                direction = trend_info['direction'].replace('_', ' ').title()
                change = trend_info.get('percent_change', 0)
                trend_text += f"{metric.replace('_', ' ').title()}:\n"
                trend_text += f"  Direção: {direction}\n"
                trend_text += f"  Mudança: {change:.1f}%\n\n"
        
        axes[2, 1].text(0.1, 0.5, trend_text,
                       transform=axes[2, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", 
                                facecolor="lightblue", alpha=0.8))
        axes[2, 1].axis('off')
        
        # Título geral
        plt.suptitle(f'Análise Temporal - Paciente {temporal_analysis.patient_id}',
                    fontsize=16)
        plt.tight_layout()
        
        # Salvar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Relatório temporal salvo em: {output_path}")
        
        plt.close()
