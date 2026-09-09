# gui_methods.py - Métodos da interface gráfica

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from pathlib import Path
import cv2
import pydicom
from typing import Optional, List, Dict
import json
from datetime import datetime
import logging

# Continuação da classe MedAIMainWindow

class MedAIMainWindow_Methods:
    """Métodos de funcionalidade para a interface principal"""
    
    # Métodos de arquivo
    def open_image(self):
        """Abre diálogo para selecionar imagem"""
        file_filter = "Imagens Médicas (*.dcm *.dicom *.png *.jpg *.jpeg *.nii *.nii.gz);;DICOM (*.dcm *.dicom);;Imagens (*.png *.jpg *.jpeg);;NIfTI (*.nii *.nii.gz);;Todos (*.*)"
        
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir Imagem Médica",
            "",
            file_filter
        )
        
        if filename:
            self.load_image(filename)
    
    def open_folder(self):
        """Abre pasta com séries DICOM"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Selecionar Pasta DICOM",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if folder:
            self.load_dicom_series(folder)
    
    def load_image(self, filepath: str):
        """Carrega imagem do arquivo"""
        try:
            self.log_activity(f"Carregando imagem: {Path(filepath).name}")
            
            # Atualizar caminho na interface
            self.file_path_edit.setText(filepath)
            
            # Detectar tipo de arquivo
            if filepath.lower().endswith(('.dcm', '.dicom')):
                self.load_dicom_file(filepath)
            elif filepath.lower().endswith(('.nii', '.nii.gz')):
                self.load_nifti_file(filepath)
            else:
                self.load_standard_image(filepath)
            
            # Adicionar ao histórico
            self.add_to_history(filepath)
            
            # Habilitar controles
            self.enable_controls(True)
            
            self.log_activity("Imagem carregada com sucesso")
            
        except Exception as e:
            self.log_activity(f"Erro ao carregar imagem: {str(e)}")
            QMessageBox.critical(self, "Erro", f"Não foi possível carregar a imagem:\n{str(e)}")
    
    def load_dicom_file(self, filepath: str):
        """Carrega arquivo DICOM"""
        # Inicializar processador DICOM se necessário
        if not self.dicom_processor:
            from dicom_processor import DICOMProcessor
            self.dicom_processor = DICOMProcessor(anonymize=True)
        
        # Ler DICOM
        ds = self.dicom_processor.read_dicom(filepath)
        
        # Converter para array
        image_array = self.dicom_processor.dicom_to_array(ds)
        
        # Extrair metadados
        metadata = self.dicom_processor.extract_metadata(ds)
        
        # Atualizar interface com metadados
        self.update_patient_info(metadata)
        
        # Exibir imagem
        self.display_image(image_array)
        
        # Armazenar dados
        self.current_image = image_array
        self.current_metadata = metadata
    
    def load_standard_image(self, filepath: str):
        """Carrega imagem padrão (PNG, JPG, etc)"""
        # Ler imagem
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError("Não foi possível ler a imagem")
        
        # Exibir imagem
        self.display_image(image)
        
        # Armazenar
        self.current_image = image
        self.current_metadata = {
            'Modality': 'Unknown',
            'StudyDate': datetime.now().strftime('%Y%m%d')
        }
    
    def load_nifti_file(self, filepath: str):
        """Carrega arquivo NIfTI"""
        import nibabel as nib
        
        # Carregar NIfTI
        nifti = nib.load(filepath)
        image_data = nifti.get_fdata()
        
        # Se for 3D, pegar slice do meio
        if len(image_data.shape) == 3:
            slice_idx = image_data.shape[2] // 2
            image = image_data[:, :, slice_idx]
        else:
            image = image_data
        
        # Normalizar
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Exibir
        self.display_image(image)
        self.current_image = image
    
    def load_dicom_series(self, folder: str):
        """Carrega série DICOM de uma pasta"""
        dicom_files = []
        
        # Buscar arquivos DICOM
        for file in Path(folder).glob('**/*'):
            if file.is_file():
                try:
                    pydicom.dcmread(str(file), stop_before_pixels=True)
                    dicom_files.append(str(file))
                except:
                    pass
        
        if not dicom_files:
            QMessageBox.warning(self, "Aviso", "Nenhum arquivo DICOM encontrado na pasta")
            return
        
        self.log_activity(f"Encontrados {len(dicom_files)} arquivos DICOM")
        
        # Criar diálogo de seleção de série
        dialog = SeriesSelectionDialog(dicom_files, self)
        if dialog.exec_():
            selected_file = dialog.get_selected_file()
            if selected_file:
                self.load_image(selected_file)
    
    # Métodos de análise
    def analyze_image(self):
        """Analisa imagem atual com modelo de IA"""
        if self.current_image is None:
            QMessageBox.warning(self, "Aviso", "Nenhuma imagem carregada")
            return
        
        try:
            # Mostrar diálogo de progresso
            progress = QProgressDialog("Analisando imagem...", "Cancelar", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Inicializar engine de inferência se necessário
            if not self.inference_engine:
                self.load_inference_engine()
            
            progress.setValue(30)
            
            # Preparar imagem
            model_type = self.model_combo.currentText()
            
            # Realizar predição
            self.log_activity(f"Iniciando análise com modelo: {model_type}")
            
            # Simular análise (substituir por chamada real)
            QApplication.processEvents()
            
            # Realizar inferência real
            result = self.inference_engine.predict_single(
                self.current_image,
                return_attention=True,
                metadata=self.current_metadata
            )
            
            progress.setValue(80)
            
            # Processar resultados
            self.process_analysis_results(result)
            
            progress.setValue(100)
            
            self.log_activity("Análise concluída com sucesso")
            
        except Exception as e:
            self.log_activity(f"Erro na análise: {str(e)}")
            QMessageBox.critical(self, "Erro", f"Erro durante análise:\n{str(e)}")
        finally:
            progress.close()
    
    def process_analysis_results(self, result):
        """Processa e exibe resultados da análise"""
        self.current_result = result
        
        # Atualizar texto de resultados
        results_text = f"""
<b>Análise Concluída</b><br>
<b>Classe Predita:</b> {result.predicted_class}<br>
<b>Confiança:</b> {result.confidence:.1%}<br>
<b>Tempo de Processamento:</b> {result.processing_time:.3f}s<br>
<br>
<b>Probabilidades por Classe:</b><br>
"""
        
        for class_name, prob in sorted(result.predictions.items(), 
                                      key=lambda x: x[1], reverse=True):
            color = "green" if prob > 0.7 else "orange" if prob > 0.3 else "red"
            results_text += f'<span style="color: {color}">• {class_name}: {prob:.1%}</span><br>'
        
        self.results_text.setHtml(results_text)
        
        # Atualizar gráfico de probabilidades
        self.update_probability_plot(result.predictions)
        
        # Atualizar métricas
        self.confidence_bar.setValue(int(result.confidence * 100))
        
        # Calcular e exibir incerteza
        if hasattr(self.inference_engine, 'analyze_uncertainty'):
            uncertainty = self.inference_engine.analyze_uncertainty(
                np.array(list(result.predictions.values())),
                len(result.predictions)
            )
            self.uncertainty_label.setText(f"{uncertainty['normalized_entropy']:.2%}")
        
        self.processing_time_label.setText(f"{result.processing_time:.3f}s")
        
        # Exibir heatmap se disponível
        if result.heatmap is not None:
            self.display_heatmap(result.heatmap)
            self.view_tabs.setCurrentIndex(1)  # Mudar para aba de heatmap
    
    def update_probability_plot(self, predictions: Dict[str, float]):
        """Atualiza gráfico de barras com probabilidades"""
        self.prob_plot.clear()
        
        # Preparar dados
        classes = list(predictions.keys())
        probs = list(predictions.values())
        
        # Criar barras horizontais
        y_pos = np.arange(len(classes))
        
        # Criar gráfico de barras
        bg = pg.BarGraphItem(
            x0=0, 
            x1=probs, 
            y=y_pos, 
            height=0.8,
            brush=[(0, 150, 255) if p > 0.5 else (255, 150, 0) for p in probs]
        )
        
        self.prob_plot.addItem(bg)
        
        # Configurar eixos
        self.prob_plot.setXRange(0, 1)
        self.prob_plot.setYRange(-0.5, len(classes) - 0.5)
        
        # Adicionar labels
        axis = self.prob_plot.getAxis('left')
        axis.setTicks([[(i, classes[i]) for i in range(len(classes))]])
    
    # Métodos de visualização
    def display_image(self, image: np.ndarray):
        """Exibe imagem no visualizador principal"""
        # Transpor se necessário para pyqtgraph
        if len(image.shape) == 2:
            image = image.T
        
        self.image_view.setImage(image)
        self.current_display_image = image.copy()
    
    def display_heatmap(self, heatmap: np.ndarray):
        """Exibe mapa de calor"""
        # Aplicar colormap
        colored_heatmap = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Sobrepor na imagem original
        if self.current_image is not None:
            # Redimensionar heatmap se necessário
            if heatmap.shape[:2] != self.current_image.shape[:2]:
                heatmap = cv2.resize(heatmap, 
                                   (self.current_image.shape[1], 
                                    self.current_image.shape[0]))
            
            # Converter imagem para RGB se necessário
            if len(self.current_image.shape) == 2:
                image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = self.current_image
            
            # Sobrepor
            overlay = cv2.addWeighted(image_rgb, 0.6, colored_heatmap, 0.4, 0)
            
            # Exibir
            self.heatmap_view.setImage(overlay.transpose(1, 0, 2))
        else:
            self.heatmap_view.setImage(colored_heatmap.transpose(1, 0, 2))
    
    def adjust_brightness(self, value: int):
        """Ajusta brilho da imagem"""
        if self.current_display_image is None:
            return
        
        # Aplicar ajuste
        adjusted = cv2.convertScaleAbs(
            self.current_display_image, 
            alpha=1.0, 
            beta=value
        )
        
        self.image_view.setImage(adjusted)
    
    def adjust_contrast(self, value: int):
        """Ajusta contraste da imagem"""
        if self.current_display_image is None:
            return
        
        # Calcular fator de contraste
        factor = (value + 100) / 100.0
        
        # Aplicar ajuste
        adjusted = cv2.convertScaleAbs(
            self.current_display_image, 
            alpha=factor, 
            beta=0
        )
        
        self.image_view.setImage(adjusted)
    
    def reset_adjustments(self):
        """Reseta ajustes de visualização"""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        
        if self.current_display_image is not None:
            self.image_view.setImage(self.current_display_image)
    
    # Métodos de zoom e navegação
    def zoom_in(self):
        """Aumenta zoom"""
        self.image_view.getView().scaleBy((0.8, 0.8))
        self.update_zoom_label()
    
    def zoom_out(self):
        """Diminui zoom"""
        self.image_view.getView().scaleBy((1.25, 1.25))
        self.update_zoom_label()
    
    def fit_to_window(self):
        """Ajusta imagem à janela"""
        self.image_view.autoRange()
        self.update_zoom_label()
    
    def update_zoom_label(self):
        """Atualiza label de zoom na status bar"""
        # Calcular zoom aproximado
        view_range = self.image_view.getView().viewRange()
        if self.current_image is not None:
            x_zoom = self.current_image.shape[1] / (view_range[0][1] - view_range[0][0])
            zoom_percent = int(x_zoom * 100)
            self.zoom_label.setText(f"Zoom: {zoom_percent}%")
    
    # Métodos de ferramentas
    def toggle_ruler(self, checked: bool):
        """Ativa/desativa ferramenta de régua"""
        if checked:
            self.log_activity("Régua ativada")
            # Implementar ferramenta de régua
            self.enable_ruler_tool()
        else:
            self.log_activity("Régua desativada")
            self.disable_ruler_tool()
    
    def toggle_roi(self, checked: bool):
        """Ativa/desativa ferramenta de ROI"""
        if checked:
            self.log_activity("ROI ativado")
            # Implementar ferramenta de ROI
            self.enable_roi_tool()
        else:
            self.log_activity("ROI desativado")
            self.disable_roi_tool()
    
    def enable_ruler_tool(self):
        """Habilita ferramenta de medição"""
        # Criar linha para medição
        self.ruler_line = pg.LineSegmentROI(
            [[100, 100], [200, 200]], 
            pen=pg.mkPen(color='y', width=2)
        )
        self.image_view.addItem(self.ruler_line)
        
        # Conectar para atualizar medidas
        self.ruler_line.sigRegionChangeFinished.connect(self.update_ruler_measurement)
    
    def disable_ruler_tool(self):
        """Desabilita ferramenta de medição"""
        if hasattr(self, 'ruler_line'):
            self.image_view.removeItem(self.ruler_line)
            del self.ruler_line
    
    def update_ruler_measurement(self):
        """Atualiza medida da régua"""
        if hasattr(self, 'ruler_line'):
            # Obter pontos
            points = self.ruler_line.getHandles()
            p1 = points[0].pos()
            p2 = points[1].pos()
            
            # Calcular distância
            distance = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
            
            # Converter para mm se houver metadados
            if hasattr(self, 'current_metadata') and 'PixelSpacing' in self.current_metadata:
                try:
                    pixel_spacing = float(self.current_metadata['PixelSpacing'].split('\\')[0])
                    distance_mm = distance * pixel_spacing
                    self.status_bar.showMessage(f"Distância: {distance_mm:.2f} mm", 3000)
                except:
                    self.status_bar.showMessage(f"Distância: {distance:.2f} pixels", 3000)
            else:
                self.status_bar.showMessage(f"Distância: {distance:.2f} pixels", 3000)
    
    def enable_roi_tool(self):
        """Habilita ferramenta de ROI"""
        # Criar ROI retangular
        self.roi = pg.RectROI(
            [100, 100], [100, 100],
            pen=pg.mkPen(color='r', width=2)
        )
        self.image_view.addItem(self.roi)
        
        # Conectar para análise de ROI
        self.roi.sigRegionChangeFinished.connect(self.analyze_roi)
    
    def disable_roi_tool(self):
        """Desabilita ferramenta de ROI"""
        if hasattr(self, 'roi'):
            self.image_view.removeItem(self.roi)
            del self.roi
    
    def analyze_roi(self):
        """Analisa região de interesse"""
        if hasattr(self, 'roi') and self.current_image is not None:
            # Obter dados da ROI
            roi_data = self.roi.getArrayRegion(
                self.current_image.T, 
                self.image_view.getImageItem()
            )
            
            # Calcular estatísticas
            mean_val = np.mean(roi_data)
            std_val = np.std(roi_data)
            min_val = np.min(roi_data)
            max_val = np.max(roi_data)
            
            # Mostrar estatísticas
            stats_text = f"ROI - Média: {mean_val:.2f}, Desvio: {std_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}"
            self.status_bar.showMessage(stats_text, 5000)
    
    # Métodos de relatório
    def save_report(self):
        """Gera e salva relatório médico"""
        if self.current_result is None:
            QMessageBox.warning(self, "Aviso", "Nenhuma análise disponível para gerar relatório")
            return
        
        # Diálogo para salvar arquivo
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar Relatório",
            f"relatorio_medai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf);;HTML Files (*.html)"
        )
        
        if filename:
            try:
                if filename.endswith('.pdf'):
                    self.generate_pdf_report(filename)
                else:
                    self.generate_html_report(filename)
                
                self.log_activity(f"Relatório salvo: {Path(filename).name}")
                QMessageBox.information(self, "Sucesso", "Relatório salvo com sucesso!")
                
            except Exception as e:
                self.log_activity(f"Erro ao salvar relatório: {str(e)}")
                QMessageBox.critical(self, "Erro", f"Erro ao salvar relatório:\n{str(e)}")
    
    def export_results(self):
        """Exporta resultados em formato JSON"""
        if self.current_result is None:
            QMessageBox.warning(self, "Aviso", "Nenhum resultado para exportar")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Resultados",
            f"resultados_medai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'image_path': self.file_path_edit.text(),
                    'model': self.model_combo.currentText(),
                    'predictions': self.current_result.predictions,
                    'predicted_class': self.current_result.predicted_class,
                    'confidence': self.current_result.confidence,
                    'processing_time': self.current_result.processing_time,
                    'metadata': self.current_result.metadata
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=4)
                
                self.log_activity(f"Resultados exportados: {Path(filename).name}")
                
            except Exception as e:
                self.log_activity(f"Erro ao exportar: {str(e)}")
                QMessageBox.critical(self, "Erro", f"Erro ao exportar:\n{str(e)}")
