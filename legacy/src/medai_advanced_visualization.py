# advanced_visualization.py - Sistema de visualização avançada e ferramentas de análise

import numpy as np
import cv2
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph.opengl as gl
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.filters import frangi, hessian
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger('MedAI.Visualization')

@dataclass
class MeasurementResult:
    """Resultado de medição"""
    measurement_type: str  # distance, area, volume, angle, density
    value: float
    unit: str
    points: List[Tuple[float, float, float]]
    metadata: Dict[str, Any]

class AdvancedVisualizationWidget(QWidget):
    """
    Widget de visualização avançada para imagens médicas
    Suporta 2D, 3D, MPR, MIP e ferramentas de análise
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.current_volume = None
        self.measurements = []
        self.annotations = []
        self.window_level = (0, 255)
        self.current_slice = 0
        
        self.init_ui()
        self.setup_vtk_pipeline()
        
    def init_ui(self):
        """Inicializa interface do usuário"""
        layout = QVBoxLayout(self)
        
        # Toolbar de visualização
        toolbar = self.create_visualization_toolbar()
        layout.addWidget(toolbar)
        
        # Área principal com splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Painel de visualização 2D/3D
        self.view_stack = QStackedWidget()
        
        # Vista 2D
        self.view_2d = self.create_2d_view()
        self.view_stack.addWidget(self.view_2d)
        
        # Vista 3D
        self.view_3d = self.create_3d_view()
        self.view_stack.addWidget(self.view_3d)
        
        # Vista MPR
        self.view_mpr = self.create_mpr_view()
        self.view_stack.addWidget(self.view_mpr)
        
        splitter.addWidget(self.view_stack)
        
        # Painel lateral com ferramentas
        self.tools_panel = self.create_tools_panel()
        splitter.addWidget(self.tools_panel)
        
        splitter.setSizes([800, 300])
        layout.addWidget(splitter)
        
        # Barra de status
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        
    def create_visualization_toolbar(self) -> QToolBar:
        """Cria toolbar de visualização"""
        toolbar = QToolBar()
        
        # Modos de visualização
        view_group = QActionGroup(self)
        
        view_2d_action = QAction(QIcon('icons/view_2d.png'), '2D', self)
        view_2d_action.setCheckable(True)
        view_2d_action.setChecked(True)
        view_2d_action.triggered.connect(lambda: self.set_view_mode('2D'))
        view_group.addAction(view_2d_action)
        toolbar.addAction(view_2d_action)
        
        view_3d_action = QAction(QIcon('icons/view_3d.png'), '3D', self)
        view_3d_action.setCheckable(True)
        view_3d_action.triggered.connect(lambda: self.set_view_mode('3D'))
        view_group.addAction(view_3d_action)
        toolbar.addAction(view_3d_action)
        
        view_mpr_action = QAction(QIcon('icons/view_mpr.png'), 'MPR', self)
        view_mpr_action.setCheckable(True)
        view_mpr_action.triggered.connect(lambda: self.set_view_mode('MPR'))
        view_group.addAction(view_mpr_action)
        toolbar.addAction(view_mpr_action)
        
        toolbar.addSeparator()
        
        # Ferramentas de medição
        measure_distance = QAction(QIcon('icons/ruler.png'), 'Distância', self)
        measure_distance.triggered.connect(self.activate_distance_tool)
        toolbar.addAction(measure_distance)
        
        measure_area = QAction(QIcon('icons/area.png'), 'Área', self)
        measure_area.triggered.connect(self.activate_area_tool)
        toolbar.addAction(measure_area)
        
        measure_angle = QAction(QIcon('icons/angle.png'), 'Ângulo', self)
        measure_angle.triggered.connect(self.activate_angle_tool)
        toolbar.addAction(measure_angle)
        
        toolbar.addSeparator()
        
        # Ferramentas de análise
        roi_tool = QAction(QIcon('icons/roi.png'), 'ROI', self)
        roi_tool.triggered.connect(self.activate_roi_tool)
        toolbar.addAction(roi_tool)
        
        histogram_tool = QAction(QIcon('icons/histogram.png'), 'Histograma', self)
        histogram_tool.triggered.connect(self.show_histogram)
        toolbar.addAction(histogram_tool)
        
        profile_tool = QAction(QIcon('icons/profile.png'), 'Perfil', self)
        profile_tool.triggered.connect(self.activate_profile_tool)
        toolbar.addAction(profile_tool)
        
        toolbar.addSeparator()
        
        # Filtros
        filters_menu = QMenu()
        
        smooth_action = QAction('Suavização', self)
        smooth_action.triggered.connect(lambda: self.apply_filter('smooth'))
        filters_menu.addAction(smooth_action)
        
        sharpen_action = QAction('Nitidez', self)
        sharpen_action.triggered.connect(lambda: self.apply_filter('sharpen'))
        filters_menu.addAction(sharpen_action)
        
        edge_action = QAction('Detecção de Bordas', self)
        edge_action.triggered.connect(lambda: self.apply_filter('edge'))
        filters_menu.addAction(edge_action)
        
        vessel_action = QAction('Realce de Vasos', self)
        vessel_action.triggered.connect(lambda: self.apply_filter('vessel'))
        filters_menu.addAction(vessel_action)
        
        filters_btn = QToolButton()
        filters_btn.setText('Filtros')
        filters_btn.setMenu(filters_menu)
        filters_btn.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(filters_btn)
        
        return toolbar
    
    def create_2d_view(self) -> QWidget:
        """Cria visualizador 2D"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Canvas matplotlib para visualização 2D
        self.figure_2d = Figure(figsize=(8, 8))
        self.canvas_2d = FigureCanvas(self.figure_2d)
        self.ax_2d = self.figure_2d.add_subplot(111)
        self.ax_2d.axis('off')
        
        layout.addWidget(self.canvas_2d)
        
        # Controles de slice para volumes
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setVisible(False)
        self.slice_slider.valueChanged.connect(self.update_slice)
        
        self.slice_label = QLabel("Slice: 0/0")
        self.slice_label.setVisible(False)
        
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Slice:"))
        slice_layout.addWidget(self.slice_slider)
        slice_layout.addWidget(self.slice_label)
        
        layout.addLayout(slice_layout)
        
        # Conectar eventos do mouse
        self.canvas_2d.mpl_connect('button_press_event', self.on_mouse_press_2d)
        self.canvas_2d.mpl_connect('motion_notify_event', self.on_mouse_move_2d)
        self.canvas_2d.mpl_connect('button_release_event', self.on_mouse_release_2d)
        self.canvas_2d.mpl_connect('scroll_event', self.on_mouse_scroll_2d)
        
        return widget
    
    def create_3d_view(self) -> QWidget:
        """Cria visualizador 3D"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Widget OpenGL para visualização 3D
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=200)
        layout.addWidget(self.gl_widget)
        
        # Controles 3D
        controls_layout = QHBoxLayout()
        
        # Threshold para renderização de volume
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.update_3d_threshold)
        
        controls_layout.addWidget(QLabel("Threshold:"))
        controls_layout.addWidget(self.threshold_slider)
        
        # Opacidade
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self.update_3d_opacity)
        
        controls_layout.addWidget(QLabel("Opacidade:"))
        controls_layout.addWidget(self.opacity_slider)
        
        layout.addLayout(controls_layout)
        
        return widget
    
    def create_mpr_view(self) -> QWidget:
        """Cria visualizador MPR (Multi-Planar Reconstruction)"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Três vistas ortogonais + 3D
        self.mpr_views = {
            'axial': self.create_mpr_plane_view('Axial'),
            'sagittal': self.create_mpr_plane_view('Sagital'),
            'coronal': self.create_mpr_plane_view('Coronal'),
            '3d': self.create_mpr_3d_view()
        }
        
        layout.addWidget(self.mpr_views['axial'], 0, 0)
        layout.addWidget(self.mpr_views['sagittal'], 0, 1)
        layout.addWidget(self.mpr_views['coronal'], 1, 0)
        layout.addWidget(self.mpr_views['3d'], 1, 1)
        
        return widget
    
    def create_mpr_plane_view(self, plane_name: str) -> QWidget:
        """Cria vista de plano MPR individual"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Label do plano
        label = QLabel(plane_name)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold; background-color: #333; color: white; padding: 5px;")
        layout.addWidget(label)
        
        # Canvas para o plano
        figure = Figure(figsize=(4, 4))
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        ax.axis('off')
        
        layout.addWidget(canvas)
        
        # Armazenar referências
        setattr(self, f'mpr_{plane_name.lower()}_figure', figure)
        setattr(self, f'mpr_{plane_name.lower()}_canvas', canvas)
        setattr(self, f'mpr_{plane_name.lower()}_ax', ax)
        
        return widget
    
    def create_mpr_3d_view(self) -> QWidget:
        """Cria vista 3D para MPR"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("3D")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold; background-color: #333; color: white; padding: 5px;")
        layout.addWidget(label)
        
        # Renderizador VTK
        from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        
        self.mpr_vtk_widget = QVTKRenderWindowInteractor(widget)
        layout.addWidget(self.mpr_vtk_widget)
        
        # Configurar renderizador
        self.mpr_renderer = vtk.vtkRenderer()
        self.mpr_vtk_widget.GetRenderWindow().AddRenderer(self.mpr_renderer)
        
        return widget
    
    def create_tools_panel(self) -> QWidget:
        """Cria painel de ferramentas lateral"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Window/Level
        wl_group = QGroupBox("Window/Level")
        wl_layout = QFormLayout()
        
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setRange(1, 4096)
        self.window_slider.setValue(400)
        self.window_slider.valueChanged.connect(self.update_window_level)
        
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(-1024, 3072)
        self.level_slider.setValue(40)
        self.level_slider.valueChanged.connect(self.update_window_level)
        
        wl_layout.addRow("Window:", self.window_slider)
        wl_layout.addRow("Level:", self.level_slider)
        
        # Presets
        preset_combo = QComboBox()
        preset_combo.addItems(['Pulmão', 'Osso', 'Tecido Mole', 'Cérebro', 'Abdômen'])
        preset_combo.currentTextChanged.connect(self.apply_window_preset)
        wl_layout.addRow("Preset:", preset_combo)
        
        wl_group.setLayout(wl_layout)
        layout.addWidget(wl_group)
        
        # Medições
        measurements_group = QGroupBox("Medições")
        measurements_layout = QVBoxLayout()
        
        self.measurements_list = QListWidget()
        self.measurements_list.setMaximumHeight(150)
        measurements_layout.addWidget(self.measurements_list)
        
        clear_measurements_btn = QPushButton("Limpar Medições")
        clear_measurements_btn.clicked.connect(self.clear_measurements)
        measurements_layout.addWidget(clear_measurements_btn)
        
        measurements_group.setLayout(measurements_layout)
        layout.addWidget(measurements_group)
        
        # Anotações
        annotations_group = QGroupBox("Anotações")
        annotations_layout = QVBoxLayout()
        
        self.annotation_text = QTextEdit()
        self.annotation_text.setMaximumHeight(100)
        annotations_layout.addWidget(self.annotation_text)
        
        add_annotation_btn = QPushButton("Adicionar Anotação")
        add_annotation_btn.clicked.connect(self.add_annotation)
        annotations_layout.addWidget(add_annotation_btn)
        
        annotations_group.setLayout(annotations_layout)
        layout.addWidget(annotations_group)
        
        # Ferramentas avançadas
        advanced_group = QGroupBox("Ferramentas Avançadas")
        advanced_layout = QVBoxLayout()
        
        segment_btn = QPushButton("Segmentação Automática")
        segment_btn.clicked.connect(self.auto_segment)
        advanced_layout.addWidget(segment_btn)
        
        vessel_btn = QPushButton("Análise de Vasos")
        vessel_btn.clicked.connect(self.analyze_vessels)
        advanced_layout.addWidget(vessel_btn)
        
        texture_btn = QPushButton("Análise de Textura")
        texture_btn.clicked.connect(self.analyze_texture)
        advanced_layout.addWidget(texture_btn)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        
        return panel
    
    def setup_vtk_pipeline(self):
        """Configura pipeline VTK para renderização 3D"""
        # Criar componentes VTK
        self.vtk_image_data = vtk.vtkImageData()
        self.vtk_volume_mapper = vtk.vtkSmartVolumeMapper()
        self.vtk_volume = vtk.vtkVolume()
        self.vtk_renderer = vtk.vtkRenderer()
        
        # Configurar propriedades de volume
        self.vtk_volume_property = vtk.vtkVolumeProperty()
        self.vtk_volume_property.ShadeOn()
        self.vtk_volume_property.SetInterpolationTypeToLinear()
        
        # Funções de transferência
        self.vtk_color_func = vtk.vtkColorTransferFunction()
        self.vtk_opacity_func = vtk.vtkPiecewiseFunction()
        
        # Configurar funções padrão
        self.setup_default_transfer_functions()
        
        # Conectar pipeline
        self.vtk_volume_mapper.SetInputData(self.vtk_image_data)
        self.vtk_volume.SetMapper(self.vtk_volume_mapper)
        self.vtk_volume.SetProperty(self.vtk_volume_property)
    
    def setup_default_transfer_functions(self):
        """Configura funções de transferência padrão"""
        # Função de cor (grayscale)
        self.vtk_color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)
        self.vtk_color_func.AddRGBPoint(64, 0.3, 0.3, 0.3)
        self.vtk_color_func.AddRGBPoint(128, 0.6, 0.6, 0.6)
        self.vtk_color_func.AddRGBPoint(192, 0.9, 0.9, 0.9)
        self.vtk_color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)
        
        # Função de opacidade
        self.vtk_opacity_func.AddPoint(0, 0.0)
        self.vtk_opacity_func.AddPoint(128, 0.5)
        self.vtk_opacity_func.AddPoint(255, 1.0)
        
        self.vtk_volume_property.SetColor(self.vtk_color_func)
        self.vtk_volume_property.SetScalarOpacity(self.vtk_opacity_func)
    
    def load_image(self, image: np.ndarray, metadata: Optional[Dict] = None):
        """Carrega imagem 2D"""
        self.current_image = image
        self.current_metadata = metadata or {}
        
        # Atualizar visualização 2D
        self.ax_2d.clear()
        self.ax_2d.imshow(image, cmap='gray', aspect='equal')
        self.ax_2d.axis('off')
        self.canvas_2d.draw()
        
        # Atualizar controles de window/level
        self.window_slider.setRange(1, int(image.max() - image.min()))
        self.level_slider.setRange(int(image.min()), int(image.max()))
        
        # Resetar medições e anotações
        self.measurements.clear()
        self.annotations.clear()
        self.update_measurements_list()
        
        logger.info(f"Imagem carregada: {image.shape}")
    
    def load_volume(self, volume: np.ndarray, spacing: Tuple[float, float, float] = (1, 1, 1)):
        """Carrega volume 3D"""
        self.current_volume = volume
        self.volume_spacing = spacing
        
        # Configurar slider de slice
        self.slice_slider.setRange(0, volume.shape[0] - 1)
        self.slice_slider.setValue(volume.shape[0] // 2)
        self.slice_slider.setVisible(True)
        self.slice_label.setVisible(True)
        
        # Atualizar visualização 2D com slice central
        self.current_slice = volume.shape[0] // 2
        self.update_slice()
        
        # Preparar dados para VTK
        self.prepare_vtk_volume(volume, spacing)
        
        # Atualizar MPR se visível
        if self.view_stack.currentIndex() == 2:
            self.update_mpr_views()
        
        logger.info(f"Volume carregado: {volume.shape}, spacing: {spacing}")
    
    def prepare_vtk_volume(self, volume: np.ndarray, spacing: Tuple[float, float, float]):
        """Prepara volume para renderização VTK"""
        # Converter para formato VTK
        vtk_data_array = numpy_support.numpy_to_vtk(
            volume.ravel(), 
            deep=True, 
            array_type=vtk.VTK_FLOAT
        )
        
        # Configurar vtkImageData
        self.vtk_image_data.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])
        self.vtk_image_data.SetSpacing(spacing[2], spacing[1], spacing[0])
        self.vtk_image_data.GetPointData().SetScalars(vtk_data_array)
        
        # Atualizar mapper
        self.vtk_volume_mapper.SetInputData(self.vtk_image_data)
    
    def set_view_mode(self, mode: str):
        """Altera modo de visualização"""
        if mode == '2D':
            self.view_stack.setCurrentIndex(0)
        elif mode == '3D':
            self.view_stack.setCurrentIndex(1)
            self.render_3d_volume()
        elif mode == 'MPR':
            self.view_stack.setCurrentIndex(2)
            if self.current_volume is not None:
                self.update_mpr_views()
    
    def render_3d_volume(self):
        """Renderiza volume 3D"""
        if self.current_volume is None:
            return
        
        # Limpar visualização anterior
        self.gl_widget.clear()
        
        # Criar isosuperfície
        threshold = self.threshold_slider.value()
        
        # Usar marching cubes para extrair superfície
        verts, faces, _, _ = measure.marching_cubes(
            self.current_volume, 
            level=threshold,
            spacing=self.volume_spacing
        )
        
        # Criar mesh
        mesh_data = gl.MeshData(vertexes=verts, faces=faces)
        mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=True,
            shader='shaded',
            glOptions='opaque'
        )
        
        # Definir cor e opacidade
        opacity = self.opacity_slider.value() / 100.0
        mesh.setColor((1, 1, 1, opacity))
        
        # Adicionar à cena
        self.gl_widget.addItem(mesh)
        
        # Adicionar eixos
        axis = gl.GLAxisItem()
        self.gl_widget.addItem(axis)
    
    def update_mpr_views(self):
        """Atualiza vistas MPR"""
        if self.current_volume is None:
            return
        
        shape = self.current_volume.shape
        
        # Slice central de cada plano
        axial_slice = self.current_volume[shape[0]//2, :, :]
        sagittal_slice = self.current_volume[:, shape[1]//2, :]
        coronal_slice = self.current_volume[:, :, shape[2]//2]
        
        # Atualizar cada vista
        self.mpr_axial_ax.clear()
        self.mpr_axial_ax.imshow(axial_slice, cmap='gray', aspect='equal')
        self.mpr_axial_ax.axis('off')
        self.mpr_axial_canvas.draw()
        
        self.mpr_sagittal_ax.clear()
        self.mpr_sagittal_ax.imshow(sagittal_slice, cmap='gray', aspect='equal')
        self.mpr_sagittal_ax.axis('off')
        self.mpr_sagittal_canvas.draw()
        
        self.mpr_coronal_ax.clear()
        self.mpr_coronal_ax.imshow(coronal_slice, cmap='gray', aspect='equal')
        self.mpr_coronal_ax.axis('off')
        self.mpr_coronal_canvas.draw()
        
        # Atualizar vista 3D VTK
        self.update_mpr_3d()
    
    def update_mpr_3d(self):
        """Atualiza vista 3D do MPR"""
        # Adicionar volume ao renderizador
        self.mpr_renderer.RemoveAllViewProps()
        self.mpr_renderer.AddVolume(self.vtk_volume)
        
        # Adicionar planos de corte
        self.add_cutting_planes()
        
        # Configurar câmera
        self.mpr_renderer.ResetCamera()
        self.mpr_vtk_widget.GetRenderWindow().Render()
    
    def add_cutting_planes(self):
        """Adiciona planos de corte ao MPR"""
        bounds = self.vtk_image_data.GetBounds()
        center = self.vtk_image_data.GetCenter()
        
        # Plano axial
        axial_plane = vtk.vtkPlaneSource()
        axial_plane.SetOrigin(bounds[0], bounds[2], center[2])
        axial_plane.SetPoint1(bounds[1], bounds[2], center[2])
        axial_plane.SetPoint2(bounds[0], bounds[3], center[2])
        axial_plane.Update()
        
        axial_mapper = vtk.vtkPolyDataMapper()
        axial_mapper.SetInputConnection(axial_plane.GetOutputPort())
        
        axial_actor = vtk.vtkActor()
        axial_actor.SetMapper(axial_mapper)
        axial_actor.GetProperty().SetColor(1, 0, 0)  # Vermelho
        axial_actor.GetProperty().SetOpacity(0.3)
        
        self.mpr_renderer.AddActor(axial_actor)
        
        # Repetir para planos sagital e coronal...
    
    def update_slice(self):
        """Atualiza slice atual"""
        if self.current_volume is None:
            return
        
        slice_idx = self.slice_slider.value()
        self.current_slice = slice_idx
        
        # Obter slice
        slice_image = self.current_volume[slice_idx, :, :]
        
        # Atualizar visualização
        self.ax_2d.clear()
        self.ax_2d.imshow(slice_image, cmap='gray', aspect='equal')
        self.ax_2d.axis('off')
        
        # Redesenhar medições e anotações
        self.redraw_overlays()
        
        self.canvas_2d.draw()
        
        # Atualizar label
        self.slice_label.setText(f"Slice: {slice_idx}/{self.current_volume.shape[0]-1}")
    
    def update_window_level(self):
        """Atualiza window/level"""
        window = self.window_slider.value()
        level = self.level_slider.value()
        
        # Calcular limites
        min_val = level - window / 2
        max_val = level + window / 2
        
        # Aplicar à imagem atual
        if self.view_stack.currentIndex() == 0:  # Vista 2D
            self.ax_2d.images[0].set_clim(min_val, max_val)
            self.canvas_2d.draw()
    
    def apply_window_preset(self, preset: str):
        """Aplica preset de window/level"""
        presets = {
            'Pulmão': {'window': 1500, 'level': -600},
            'Osso': {'window': 2000, 'level': 300},
            'Tecido Mole': {'window': 400, 'level': 40},
            'Cérebro': {'window': 80, 'level': 40},
            'Abdômen': {'window': 350, 'level': 50}
        }
        
        if preset in presets:
            self.window_slider.setValue(presets[preset]['window'])
            self.level_slider.setValue(presets[preset]['level'])
    
    def activate_distance_tool(self):
        """Ativa ferramenta de medição de distância"""
        self.current_tool = 'distance'
        self.tool_points = []
        self.canvas_2d.setCursor(Qt.CrossCursor)
        self.status_bar.showMessage("Clique em dois pontos para medir distância")
    
    def activate_area_tool(self):
        """Ativa ferramenta de medição de área"""
        self.current_tool = 'area'
        self.tool_points = []
        self.canvas_2d.setCursor(Qt.CrossCursor)
        self.status_bar.showMessage("Clique para definir pontos do polígono, duplo clique para finalizar")
    
    def activate_angle_tool(self):
        """Ativa ferramenta de medição de ângulo"""
        self.current_tool = 'angle'
        self.tool_points = []
        self.canvas_2d.setCursor(Qt.CrossCursor)
        self.status_bar.showMessage("Clique em três pontos para medir ângulo")
    
    def activate_roi_tool(self):
        """Ativa ferramenta de ROI"""
        self.current_tool = 'roi'
        self.roi_start = None
        self.roi_end = None
        self.canvas_2d.setCursor(Qt.CrossCursor)
        self.status_bar.showMessage("Arraste para definir ROI")
    
    def activate_profile_tool(self):
        """Ativa ferramenta de perfil de intensidade"""
        self.current_tool = 'profile'
        self.tool_points = []
        self.canvas_2d.setCursor(Qt.CrossCursor)
        self.status_bar.showMessage("Clique em dois pontos para traçar perfil")
    
    def on_mouse_press_2d(self, event):
        """Manipula clique do mouse na vista 2D"""
        if event.inaxes != self.ax_2d:
            return
        
        x, y = event.xdata, event.ydata
        
        if hasattr(self, 'current_tool'):
            if self.current_tool == 'distance':
                self.tool_points.append((x, y))
                if len(self.tool_points) == 2:
                    self.measure_distance()
            
            elif self.current_tool == 'area':
                if event.dblclick:
                    self.measure_area()
                else:
                    self.tool_points.append((x, y))
                    self.draw_temp_polygon()
            
            elif self.current_tool == 'angle':
                self.tool_points.append((x, y))
                if len(self.tool_points) == 3:
                    self.measure_angle()
            
            elif self.current_tool == 'roi':
                self.roi_start = (x, y)
            
            elif self.current_tool == 'profile':
                self.tool_points.append((x, y))
                if len(self.tool_points) == 2:
                    self.show_intensity_profile()
    
    def on_mouse_move_2d(self, event):
        """Manipula movimento do mouse na vista 2D"""
        if event.inaxes != self.ax_2d:
            return
        
        x, y = event.xdata, event.ydata
        
        # Atualizar coordenadas e valor do pixel
        if self.current_image is not None:
            row, col = int(y), int(x)
            if 0 <= row < self.current_image.shape[0] and 0 <= col < self.current_image.shape[1]:
                value = self.current_image[row, col]
                self.status_bar.showMessage(f"Posição: ({col}, {row}), Valor: {value:.2f}")
        
        # Desenhar ROI temporário
        if hasattr(self, 'current_tool') and self.current_tool == 'roi' and self.roi_start:
            self.roi_end = (x, y)
            self.draw_temp_roi()
    
    def on_mouse_release_2d(self, event):
        """Manipula soltura do mouse na vista 2D"""
        if hasattr(self, 'current_tool') and self.current_tool == 'roi' and self.roi_start and self.roi_end:
            self.analyze_roi()
    
    def on_mouse_scroll_2d(self, event):
        """Manipula scroll do mouse para navegar em slices"""
        if self.current_volume is not None:
            if event.button == 'up':
                new_value = min(self.slice_slider.value() + 1, self.slice_slider.maximum())
            else:
                new_value = max(self.slice_slider.value() - 1, self.slice_slider.minimum())
            
            self.slice_slider.setValue(new_value)
    
    def measure_distance(self):
        """Mede distância entre dois pontos"""
        p1, p2 = self.tool_points[-2:]
        
        # Calcular distância
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Converter para unidades físicas se disponível
        if 'PixelSpacing' in self.current_metadata:
            pixel_spacing = self.current_metadata['PixelSpacing']
            distance_mm = distance * float(pixel_spacing[0])
            unit = 'mm'
            value = distance_mm
        else:
            unit = 'pixels'
            value = distance
        
        # Adicionar medição
        measurement = MeasurementResult(
            measurement_type='distance',
            value=value,
            unit=unit,
            points=[p1, p2],
            metadata={'pixels': distance}
        )
        
        self.measurements.append(measurement)
        self.update_measurements_list()
        
        # Desenhar linha
        self.ax_2d.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
        self.ax_2d.text((p1[0] + p2[0])/2, (p1[1] + p2[1])/2, 
                        f'{value:.2f} {unit}', 
                        color='red', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        self.canvas_2d.draw()
        
        # Resetar ferramenta
        self.current_tool = None
        self.canvas_2d.setCursor(Qt.ArrowCursor)
    
    def measure_area(self):
        """Mede área de polígono"""
        if len(self.tool_points) < 3:
            return
        
        # Calcular área usando shoelace formula
        points = np.array(self.tool_points)
        area = 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - 
                           np.dot(points[:, 1], np.roll(points[:, 0], 1)))
        
        # Converter para unidades físicas
        if 'PixelSpacing' in self.current_metadata:
            pixel_spacing = self.current_metadata['PixelSpacing']
            area_mm2 = area * float(pixel_spacing[0]) * float(pixel_spacing[1])
            unit = 'mm²'
            value = area_mm2
        else:
            unit = 'pixels²'
            value = area
        
        # Adicionar medição
        measurement = MeasurementResult(
            measurement_type='area',
            value=value,
            unit=unit,
            points=self.tool_points.copy(),
            metadata={'pixels': area}
        )
        
        self.measurements.append(measurement)
        self.update_measurements_list()
        
        # Desenhar polígono
        polygon = plt.Polygon(self.tool_points, fill=False, edgecolor='r', linewidth=2)
        self.ax_2d.add_patch(polygon)
        
        # Adicionar texto
        center = np.mean(points, axis=0)
        self.ax_2d.text(center[0], center[1], f'{value:.2f} {unit}',
                       color='red', fontsize=10, ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        self.canvas_2d.draw()
        
        # Resetar
        self.current_tool = None
        self.tool_points = []
        self.canvas_2d.setCursor(Qt.ArrowCursor)
    
    def measure_angle(self):
        """Mede ângulo entre três pontos"""
        p1, p2, p3 = self.tool_points[-3:]
        
        # Vetores
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        # Calcular ângulo
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Adicionar medição
        measurement = MeasurementResult(
            measurement_type='angle',
            value=angle_deg,
            unit='degrees',
            points=[p1, p2, p3],
            metadata={'radians': angle_rad}
        )
        
        self.measurements.append(measurement)
        self.update_measurements_list()
        
        # Desenhar ângulo
        self.ax_2d.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
        self.ax_2d.plot([p2[0], p3[0]], [p2[1], p3[1]], 'r-', linewidth=2)
        
        # Arco do ângulo
        angle_radius = 30
        angle_start = np.degrees(np.arctan2(v1[1], v1[0]))
        angle_end = np.degrees(np.arctan2(v2[1], v2[0]))
        
        arc = plt.Circle(p2, angle_radius, fill=False, color='red')
        self.ax_2d.add_patch(arc)
        
        # Texto
        text_pos = np.array(p2) + angle_radius * 0.7 * (v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2))
        self.ax_2d.text(text_pos[0], text_pos[1], f'{angle_deg:.1f}°',
                       color='red', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        self.canvas_2d.draw()
        
        # Resetar
        self.current_tool = None
        self.tool_points = []
        self.canvas_2d.setCursor(Qt.ArrowCursor)
    
    def analyze_roi(self):
        """Analisa região de interesse"""
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        
        # Garantir ordem correta
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Extrair ROI
        roi = self.current_image[int(y1):int(y2), int(x1):int(x2)]
        
        # Calcular estatísticas
        stats = {
            'mean': np.mean(roi),
            'std': np.std(roi),
            'min': np.min(roi),
            'max': np.max(roi),
            'median': np.median(roi)
        }
        
        # Mostrar resultados
        result_text = f"ROI Statistics:\n"
        result_text += f"Mean: {stats['mean']:.2f}\n"
        result_text += f"Std: {stats['std']:.2f}\n"
        result_text += f"Min: {stats['min']:.2f}\n"
        result_text += f"Max: {stats['max']:.2f}\n"
        result_text += f"Median: {stats['median']:.2f}"
        
        QMessageBox.information(self, "ROI Analysis", result_text)
        
        # Desenhar ROI
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='yellow', linewidth=2)
        self.ax_2d.add_patch(rect)
        self.canvas_2d.draw()
        
        # Resetar
        self.current_tool = None
        self.roi_start = None
        self.roi_end = None
        self.canvas_2d.setCursor(Qt.ArrowCursor)
    
    def show_histogram(self):
        """Mostra histograma da imagem"""
        if self.current_image is None:
            return
        
        # Criar diálogo com histograma
        dialog = QDialog(self)
        dialog.setWindowTitle("Histograma")
        dialog.setModal(False)
        layout = QVBoxLayout(dialog)
        
        # Criar figura
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Plotar histograma
        data = self.current_image.flatten()
        ax.hist(data, bins=256, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Intensidade')
        ax.set_ylabel('Frequência')
        ax.set_title('Histograma de Intensidades')
        ax.grid(True, alpha=0.3)
        
        # Adicionar estatísticas
        stats_text = f'Mean: {np.mean(data):.2f}\n'
        stats_text += f'Std: {np.std(data):.2f}\n'
        stats_text += f'Min: {np.min(data):.2f}\n'
        stats_text += f'Max: {np.max(data):.2f}'
        
        ax.text(0.7, 0.95, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               verticalalignment='top')
        
        layout.addWidget(canvas)
        dialog.resize(800, 600)
        dialog.show()
    
    def show_intensity_profile(self):
        """Mostra perfil de intensidade ao longo de uma linha"""
        if len(self.tool_points) < 2:
            return
        
        p1, p2 = self.tool_points[-2:]
        
        # Extrair perfil
        num_points = int(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
        x_coords = np.linspace(p1[0], p2[0], num_points)
        y_coords = np.linspace(p1[1], p2[1], num_points)
        
        # Interpolação bilinear para obter valores
        from scipy.interpolate import interp2d
        y_range = np.arange(self.current_image.shape[0])
        x_range = np.arange(self.current_image.shape[1])
        
        f = interp2d(x_range, y_range, self.current_image, kind='linear')
        profile = [f(x, y)[0] for x, y in zip(x_coords, y_coords)]
        
        # Mostrar perfil
        dialog = QDialog(self)
        dialog.setWindowTitle("Perfil de Intensidade")
        dialog.setModal(False)
        layout = QVBoxLayout(dialog)
        
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        distances = np.linspace(0, num_points, num_points)
        ax.plot(distances, profile, 'b-', linewidth=2)
        ax.set_xlabel('Distância (pixels)')
        ax.set_ylabel('Intensidade')
        ax.set_title('Perfil de Intensidade')
        ax.grid(True, alpha=0.3)
        
        layout.addWidget(canvas)
        dialog.resize(800, 600)
        dialog.show()
        
        # Desenhar linha no visualizador
        self.ax_2d.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2)
        self.canvas_2d.draw()
        
        # Resetar
        self.current_tool = None
        self.tool_points = []
        self.canvas_2d.setCursor(Qt.ArrowCursor)
    
    def apply_filter(self, filter_type: str):
        """Aplica filtro à imagem"""
        if self.current_image is None:
            return
        
        filtered = self.current_image.copy()
        
        if filter_type == 'smooth':
            # Filtro Gaussiano
            filtered = cv2.GaussianBlur(filtered, (5, 5), 1.0)
        
        elif filter_type == 'sharpen':
            # Unsharp masking
            gaussian = cv2.GaussianBlur(filtered, (0, 0), 2.0)
            filtered = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
        
        elif filter_type == 'edge':
            # Detecção de bordas Canny
            filtered = cv2.Canny(filtered.astype(np.uint8), 50, 150)
        
        elif filter_type == 'vessel':
            # Filtro de Frangi para realce de vasos
            filtered = frangi(filtered, sigmas=range(1, 10, 2), black_ridges=False)
            filtered = (filtered * 255).astype(np.uint8)
        
        # Atualizar visualização
        self.ax_2d.clear()
        self.ax_2d.imshow(filtered, cmap='gray', aspect='equal')
        self.ax_2d.axis('off')
        self.canvas_2d.draw()
        
        # Armazenar imagem filtrada
        self.filtered_image = filtered
    
    def auto_segment(self):
        """Realiza segmentação automática"""
        if self.current_image is None:
            return
        
        # Diálogo para escolher método
        methods = ['Threshold', 'Region Growing', 'Watershed', 'Active Contours']
        method, ok = QInputDialog.getItem(self, "Segmentação", 
                                         "Escolha o método:", methods, 0, False)
        
        if not ok:
            return
        
        segmented = None
        
        if method == 'Threshold':
            # Threshold de Otsu
            threshold = cv2.threshold(self.current_image.astype(np.uint8), 
                                    0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            segmented = self.current_image > threshold
        
        elif method == 'Region Growing':
            # Implementação simplificada de region growing
            seed_point = (self.current_image.shape[0]//2, self.current_image.shape[1]//2)
            segmented = self.region_growing(self.current_image, seed_point)
        
        elif method == 'Watershed':
            # Watershed
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_max
            from scipy import ndimage as ndi
            
            # Calcular gradiente
            gradient = cv2.morphologyEx(self.current_image.astype(np.uint8), 
                                      cv2.MORPH_GRADIENT, np.ones((3,3)))
            
            # Encontrar marcadores
            distance = ndi.distance_transform_edt(self.current_image > 100)
            coords = peak_local_max(distance, min_distance=20, indices=False)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[coords] = True
            markers, _ = ndi.label(mask)
            
            segmented = watershed(gradient, markers)
        
        elif method == 'Active Contours':
            # Snake/Active contours
            from skimage.segmentation import active_contour
            
            # Criar contorno inicial circular
            s = np.linspace(0, 2*np.pi, 100)
            center = np.array(self.current_image.shape) / 2
            radius = min(self.current_image.shape) / 4
            init = np.array([center[0] + radius*np.sin(s), 
                           center[1] + radius*np.cos(s)]).T
            
            # Aplicar active contour
            snake = active_contour(self.current_image, init, alpha=0.015, beta=10)
            
            # Criar máscara a partir do contorno
            segmented = np.zeros_like(self.current_image, dtype=bool)
            rr, cc = measure.grid_points_in_poly(segmented.shape, snake)
            segmented[rr, cc] = True
        
        if segmented is not None:
            # Mostrar resultado
            self.ax_2d.clear()
            self.ax_2d.imshow(self.current_image, cmap='gray', alpha=0.5)
            self.ax_2d.imshow(segmented, cmap='jet', alpha=0.5)
            self.ax_2d.axis('off')
            self.canvas_2d.draw()
            
            # Calcular estatísticas
            area = np.sum(segmented)
            QMessageBox.information(self, "Segmentação", 
                                  f"Segmentação concluída\nÁrea segmentada: {area} pixels")
    
    def region_growing(self, image: np.ndarray, seed: Tuple[int, int], 
                      tolerance: float = 10) -> np.ndarray:
        """Implementa region growing simples"""
        segmented = np.zeros_like(image, dtype=bool)
        seed_value = image[seed]
        
        # Fila de pixels a verificar
        queue = [seed]
        
        while queue:
            y, x = queue.pop(0)
            
            if segmented[y, x]:
                continue
            
            # Verificar se o pixel está dentro da tolerância
            if abs(image[y, x] - seed_value) <= tolerance:
                segmented[y, x] = True
                
                # Adicionar vizinhos à fila
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < image.shape[0] and 0 <= nx < image.shape[1] 
                        and not segmented[ny, nx]):
                        queue.append((ny, nx))
        
        return segmented
    
    def analyze_vessels(self):
        """Analisa estruturas vasculares"""
        if self.current_image is None:
            return
        
        # Aplicar filtro de Frangi
        sigmas = range(1, 10, 2)
        vessel_enhanced = frangi(self.current_image, sigmas=sigmas, black_ridges=False)
        
        # Binarizar
        threshold = 0.1 * vessel_enhanced.max()
        vessels_binary = vessel_enhanced > threshold
        
        # Análise morfológica
        skeleton = morphology.skeletonize(vessels_binary)
        
        # Calcular métricas
        vessel_pixels = np.sum(vessels_binary)
        skeleton_pixels = np.sum(skeleton)
        
        # Detectar bifurcações
        kernel = np.array([[1,1,1], [1,10,1], [1,1,1]])
        convolved = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        bifurcations = (convolved > 12) & skeleton
        num_bifurcations = np.sum(bifurcations)
        
        # Mostrar resultado
        self.ax_2d.clear()
        self.ax_2d.imshow(self.current_image, cmap='gray', alpha=0.5)
        self.ax_2d.imshow(skeleton, cmap='hot', alpha=0.7)
        
        # Marcar bifurcações
        y_coords, x_coords = np.where(bifurcations)
        self.ax_2d.scatter(x_coords, y_coords, c='red', s=50, marker='o')
        
        self.ax_2d.axis('off')
        self.canvas_2d.draw()
        
        # Mostrar estatísticas
        stats_text = f"Análise de Vasos:\n"
        stats_text += f"Área vascular: {vessel_pixels} pixels\n"
        stats_text += f"Comprimento total: {skeleton_pixels} pixels\n"
        stats_text += f"Pontos de bifurcação: {num_bifurcations}"
        
        QMessageBox.information(self, "Análise de Vasos", stats_text)
    
    def analyze_texture(self):
        """Analisa textura da imagem"""
        if self.current_image is None:
            return
        
        from skimage.feature import greycomatrix, greycoprops
        
        # Converter para uint8
        image_uint8 = (self.current_image / self.current_image.max() * 255).astype(np.uint8)
        
        # Calcular GLCM
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = greycomatrix(image_uint8, distances, angles, 256, symmetric=True, normed=True)
        
        # Extrair features
        features = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features[prop] = greycoprops(glcm, prop).mean()
        
        # Adicionar outras features
        features['entropy'] = -np.sum(glcm * np.log2(glcm + 1e-10))
        features['mean'] = np.mean(image_uint8)
        features['std'] = np.std(image_uint8)
        
        # Mostrar resultados
        result_text = "Análise de Textura:\n\n"
        for feature, value in features.items():
            result_text += f"{feature.capitalize()}: {value:.4f}\n"
        
        QMessageBox.information(self, "Análise de Textura", result_text)
    
    def update_measurements_list(self):
        """Atualiza lista de medições"""
        self.measurements_list.clear()
        
        for i, measurement in enumerate(self.measurements):
            text = f"{measurement.measurement_type}: {measurement.value:.2f} {measurement.unit}"
            self.measurements_list.addItem(text)
    
    def clear_measurements(self):
        """Limpa todas as medições"""
        self.measurements.clear()
        self.update_measurements_list()
        
        # Redesenhar imagem sem overlays
        if self.current_image is not None:
            self.ax_2d.clear()
            self.ax_2d.imshow(self.current_image, cmap='gray', aspect='equal')
            self.ax_2d.axis('off')
            self.canvas_2d.draw()
    
    def add_annotation(self):
        """Adiciona anotação à imagem"""
        text = self.annotation_text.toPlainText()
        if not text:
            return
        
        # Pedir posição para anotação
        QMessageBox.information(self, "Anotação", 
                              "Clique na imagem onde deseja adicionar a anotação")
        
        self.pending_annotation = text
        self.canvas_2d.setCursor(Qt.CrossCursor)
    
    def redraw_overlays(self):
        """Redesenha todas as medições e anotações"""
        # Implementar redesenho de overlays após mudança de slice
        pass
    
    def draw_temp_polygon(self):
        """Desenha polígono temporário durante criação"""
        if len(self.tool_points) < 2:
            return
        
        # Limpar linhas temporárias anteriores
        for line in self.ax_2d.lines[:]:
            if line.get_linestyle() == '--':
                line.remove()
        
        # Desenhar linhas
        points = self.tool_points + [self.tool_points[0]]  # Fechar polígono
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        self.ax_2d.plot(x_coords, y_coords, 'r--', linewidth=1)
        self.canvas_2d.draw()
    
    def draw_temp_roi(self):
        """Desenha ROI temporário durante criação"""
        if not self.roi_start or not self.roi_end:
            return
        
        # Limpar retângulos temporários
        for patch in self.ax_2d.patches[:]:
            if isinstance(patch, plt.Rectangle) and patch.get_linestyle() == '--':
                patch.remove()
        
        # Desenhar retângulo
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        width = x2 - x1
        height = y2 - y1
        
        rect = plt.Rectangle((x1, y1), width, height, 
                           fill=False, edgecolor='yellow', 
                           linewidth=1, linestyle='--')
        self.ax_2d.add_patch(rect)
        self.canvas_2d.draw()
    
    def update_3d_threshold(self, value: int):
        """Atualiza threshold para renderização 3D"""
        if self.view_stack.currentIndex() == 1:  # Vista 3D
            self.render_3d_volume()
    
    def update_3d_opacity(self, value: int):
        """Atualiza opacidade para renderização 3D"""
        if self.view_stack.currentIndex() == 1:  # Vista 3D
            self.render_3d_volume()
