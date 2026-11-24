# gui_main.py - Interface gráfica principal do MedAI

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from pathlib import Path
import numpy as np
import logging
from typing import Optional, Dict, List
import json
from datetime import datetime

logger = logging.getLogger('MedAI.GUI')

class MedAIMainWindow(QMainWindow):
    """
    Janela principal do sistema MedAI
    Interface moderna e intuitiva para análise de imagens médicas
    """
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_result = None
        self.dicom_processor = None
        self.inference_engine = None
        self.history = []
        
        self.init_ui()
        self.setup_connections()
        self.load_settings()
        
    def init_ui(self):
        """Inicializa interface do usuário"""
        self.setWindowTitle("MedAI - Sistema de Análise Radiológica por IA")
        self.setGeometry(100, 100, 1400, 900)
        
        # Tema futurista inspirado em radiologia e IA
        self.setStyleSheet("""
            QMainWindow {
                background-color: #101820;
                color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #1e90ff;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #1e90ff;
            }
            QPushButton {
                background-color: #1e90ff;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00c2ff;
            }
            QPushButton:pressed {
                background-color: #0080ff;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #aaaaaa;
            }
            QLabel {
                color: #e0e0e0;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #1e90ff;
                border-radius: 4px;
                background-color: #1b1b1b;
                color: #e0e0e0;
            }
            QProgressBar {
                border: 1px solid #1e90ff;
                border-radius: 4px;
                text-align: center;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
                border-radius: 3px;
            }
        """)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Painel esquerdo
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 2)
        
        # Painel central
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, 3)
        
        # Painel direito
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
        # Menu bar
        self.create_menu_bar()
        
        # Status bar
        self.create_status_bar()
        
        # Toolbar
        self.create_toolbar()
        
    def create_menu_bar(self):
        """Cria barra de menu"""
        menubar = self.menuBar()
        
        # Menu Arquivo
        file_menu = menubar.addMenu('&Arquivo')
        
        open_action = QAction(QIcon('icons/open.png'), 'Abrir Imagem', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction(QIcon('icons/folder.png'), 'Abrir Pasta', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        save_report_action = QAction(QIcon('icons/save.png'), 'Salvar Relatório', self)
        save_report_action.setShortcut('Ctrl+S')
        save_report_action.triggered.connect(self.save_report)
        file_menu.addAction(save_report_action)
        
        export_action = QAction('Exportar Resultados', self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Sair', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Ferramentas
        tools_menu = menubar.addMenu('&Ferramentas')
        
        batch_action = QAction('Processamento em Lote', self)
        batch_action.triggered.connect(self.batch_processing)
        tools_menu.addAction(batch_action)
        
        compare_action = QAction('Comparar Imagens', self)
        compare_action.triggered.connect(self.compare_images)
        tools_menu.addAction(compare_action)
        
        tools_menu.addSeparator()
        
        settings_action = QAction('Configurações', self)
        settings_action.setShortcut('Ctrl+,')
        settings_action.triggered.connect(self.open_settings)
        tools_menu.addAction(settings_action)
        
        # Menu Modelos
        models_menu = menubar.addMenu('&Modelos')
        
        load_model_action = QAction('Carregar Modelo', self)
        load_model_action.triggered.connect(self.load_model)
        models_menu.addAction(load_model_action)
        
        train_model_action = QAction('Treinar Modelo', self)
        train_model_action.triggered.connect(self.train_model)
        models_menu.addAction(train_model_action)
        
        models_menu.addSeparator()
        
        model_info_action = QAction('Informações do Modelo', self)
        model_info_action.triggered.connect(self.show_model_info)
        models_menu.addAction(model_info_action)
        
        # Menu Ajuda
        help_menu = menubar.addMenu('&Ajuda')
        
        help_action = QAction('Documentação', self)
        help_action.setShortcut('F1')
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        about_action = QAction('Sobre', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_toolbar(self):
        """Cria barra de ferramentas"""
        toolbar = self.addToolBar('Principal')
        toolbar.setMovable(False)
        
        # Botões principais
        open_btn = QAction(QIcon('icons/open.png'), 'Abrir', self)
        open_btn.triggered.connect(self.open_image)
        toolbar.addAction(open_btn)
        
        analyze_btn = QAction(QIcon('icons/analyze.png'), 'Analisar', self)
        analyze_btn.triggered.connect(self.analyze_image)
        toolbar.addAction(analyze_btn)
        
        toolbar.addSeparator()
        
        # Ferramentas de visualização
        zoom_in_btn = QAction(QIcon('icons/zoom_in.png'), 'Zoom +', self)
        zoom_in_btn.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_btn)
        
        zoom_out_btn = QAction(QIcon('icons/zoom_out.png'), 'Zoom -', self)
        zoom_out_btn.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_btn)
        
        fit_btn = QAction(QIcon('icons/fit.png'), 'Ajustar', self)
        fit_btn.triggered.connect(self.fit_to_window)
        toolbar.addAction(fit_btn)
        
        toolbar.addSeparator()
        
        # Ferramentas de medição
        ruler_btn = QAction(QIcon('icons/ruler.png'), 'Régua', self)
        ruler_btn.setCheckable(True)
        ruler_btn.triggered.connect(self.toggle_ruler)
        toolbar.addAction(ruler_btn)
        
        roi_btn = QAction(QIcon('icons/roi.png'), 'ROI', self)
        roi_btn.setCheckable(True)
        roi_btn.triggered.connect(self.toggle_roi)
        toolbar.addAction(roi_btn)
        
    def create_status_bar(self):
        """Cria barra de status"""
        self.status_bar = self.statusBar()
        
        # Labels permanentes
        self.coord_label = QLabel("X: 0, Y: 0")
        self.pixel_label = QLabel("Valor: 0")
        self.zoom_label = QLabel("Zoom: 100%")
        self.model_label = QLabel("Modelo: Nenhum")
        
        self.status_bar.addPermanentWidget(self.coord_label)
        self.status_bar.addPermanentWidget(self.pixel_label)
        self.status_bar.addPermanentWidget(self.zoom_label)
        self.status_bar.addPermanentWidget(self.model_label)
        
    def create_left_panel(self) -> QWidget:
        """Cria painel esquerdo com controles"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Seleção de arquivo
        file_group = QGroupBox("Arquivo")
        file_layout = QVBoxLayout()
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("Nenhum arquivo selecionado")
        file_layout.addWidget(self.file_path_edit)
        
        open_btn = QPushButton("Abrir Imagem")
        open_btn.clicked.connect(self.open_image)
        file_layout.addWidget(open_btn)
        
        open_folder_btn = QPushButton("Abrir Pasta DICOM")
        open_folder_btn.clicked.connect(self.open_folder)
        file_layout.addWidget(open_folder_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Seleção de modelo
        model_group = QGroupBox("Modelo de IA")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Raio-X Torácico",
            "CT Cerebral",
            "Detecção de Fraturas",
            "Segmentação Pulmonar",
            "Detecção de Nódulos"
        ])
        model_layout.addWidget(QLabel("Tipo de Análise:"))
        model_layout.addWidget(self.model_combo)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(70)
        self.threshold_label = QLabel("Limiar: 70%")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"Limiar: {v}%")
        )
        
        model_layout.addWidget(self.threshold_label)
        model_layout.addWidget(self.threshold_slider)
        
        analyze_btn = QPushButton("Analisar Imagem")
        analyze_btn.clicked.connect(self.analyze_image)
        model_layout.addWidget(analyze_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Informações do paciente
        patient_group = QGroupBox("Informações do Paciente")
        patient_layout = QFormLayout()
        
        self.patient_id_label = QLabel("-")
        self.patient_age_label = QLabel("-")
        self.study_date_label = QLabel("-")
        self.modality_label = QLabel("-")
        
        patient_layout.addRow("ID:", self.patient_id_label)
        patient_layout.addRow("Idade:", self.patient_age_label)
        patient_layout.addRow("Data:", self.study_date_label)
        patient_layout.addRow("Modalidade:", self.modality_label)
        
        patient_group.setLayout(patient_layout)
        layout.addWidget(patient_group)
        
        # Histórico
        history_group = QGroupBox("Histórico")
        history_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        history_layout.addWidget(self.history_list)
        
        clear_history_btn = QPushButton("Limpar Histórico")
        clear_history_btn.clicked.connect(self.clear_history)
        history_layout.addWidget(clear_history_btn)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        layout.addStretch()
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        """Cria painel central com visualizador de imagem"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs para diferentes visualizações
        self.view_tabs = QTabWidget()
        
        # Tab de imagem principal
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.view_tabs.addTab(self.image_view, "Imagem Original")
        
        # Tab de heatmap
        self.heatmap_view = pg.ImageView()
        self.heatmap_view.ui.histogram.hide()
        self.heatmap_view.ui.roiBtn.hide()
        self.heatmap_view.ui.menuBtn.hide()
        self.view_tabs.addTab(self.heatmap_view, "Mapa de Calor")
        
        # Tab de comparação
        self.compare_widget = QWidget()
        compare_layout = QHBoxLayout(self.compare_widget)
        
        self.compare_view1 = pg.ImageView()
        self.compare_view1.ui.histogram.hide()
        self.compare_view1.ui.roiBtn.hide()
        self.compare_view1.ui.menuBtn.hide()
        
        self.compare_view2 = pg.ImageView()
        self.compare_view2.ui.histogram.hide()
        self.compare_view2.ui.roiBtn.hide()
        self.compare_view2.ui.menuBtn.hide()
        
        compare_layout.addWidget(self.compare_view1)
        compare_layout.addWidget(self.compare_view2)
        
        self.view_tabs.addTab(self.compare_widget, "Comparação")
        
        layout.addWidget(self.view_tabs)
        
        # Controles de visualização
        controls_layout = QHBoxLayout()
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        
        controls_layout.addWidget(QLabel("Brilho:"))
        controls_layout.addWidget(self.brightness_slider)
        controls_layout.addWidget(QLabel("Contraste:"))
        controls_layout.addWidget(self.contrast_slider)
        
        reset_btn = QPushButton("Resetar")
        reset_btn.clicked.connect(self.reset_adjustments)
        controls_layout.addWidget(reset_btn)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Cria painel direito com resultados"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Resultados da análise
        results_group = QGroupBox("Resultados da Análise")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        
        # Gráfico de probabilidades
        self.prob_plot = pg.PlotWidget()
        self.prob_plot.setLabel('left', 'Classe')
        self.prob_plot.setLabel('bottom', 'Probabilidade (%)')
        self.prob_plot.setMaximumHeight(200)
        results_layout.addWidget(self.prob_plot)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Métricas de confiança
        metrics_group = QGroupBox("Métricas de Confiança")
        metrics_layout = QFormLayout()
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.uncertainty_label = QLabel("-")
        self.processing_time_label = QLabel("-")
        
        metrics_layout.addRow("Confiança:", self.confidence_bar)
        metrics_layout.addRow("Incerteza:", self.uncertainty_label)
        metrics_layout.addRow("Tempo:", self.processing_time_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Ações
        actions_group = QGroupBox("Ações")
        actions_layout = QVBoxLayout()
        
        save_report_btn = QPushButton("Gerar Relatório")
        save_report_btn.clicked.connect(self.save_report)
        actions_layout.addWidget(save_report_btn)
        
        export_btn = QPushButton("Exportar Resultados")
        export_btn.clicked.connect(self.export_results)
        actions_layout.addWidget(export_btn)
        
        second_opinion_btn = QPushButton("Segunda Opinião")
        second_opinion_btn.clicked.connect(self.get_second_opinion)
        actions_layout.addWidget(second_opinion_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Log de atividades
        log_group = QGroupBox("Log de Atividades")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return panel
    
    def setup_connections(self):
        """Configura conexões de sinais"""
        # Conectar eventos do visualizador de imagem
        self.image_view.getImageItem().sigImageChanged.connect(self.on_image_changed)
        
    def load_settings(self):
        """Carrega configurações salvas"""
        settings = QSettings('MedAI', 'RadiologyAnalyzer')
        
        # Restaurar geometria da janela
        geometry = settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restaurar último modelo usado
        last_model = settings.value('last_model', 0)
        self.model_combo.setCurrentIndex(int(last_model))
        
        # Restaurar threshold
        threshold = settings.value('threshold', 70)
        self.threshold_slider.setValue(int(threshold))
    
    def save_settings(self):
        """Salva configurações"""
        settings = QSettings('MedAI', 'RadiologyAnalyzer')
        
        # Salvar geometria
        settings.setValue('geometry', self.saveGeometry())
        
        # Salvar modelo atual
        settings.setValue('last_model', self.model_combo.currentIndex())
        
        # Salvar threshold
        settings.setValue('threshold', self.threshold_slider.value())
    
    def closeEvent(self, event):
        """Evento ao fechar aplicação"""
        self.save_settings()
        event.accept()
        
    def log_activity(self, message: str):
        """Adiciona mensagem ao log de atividades"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll para o final
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
