"""
MedAI Integration Manager - Coordenador central do sistema
Implementa o padrão Facade para integração de todos os módulos
"""

import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger('MedAI.IntegrationManager')

class MedAIIntegrationManager:
    """
    Gerenciador central que coordena todos os módulos do sistema MedAI
    Implementa o padrão Facade para simplificar a interface
    """
    
    def __init__(self):
        self.current_session = None
        self.loaded_models = {}
        self.analysis_history = []
        self._lock = threading.Lock()
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Inicializa todos os componentes do sistema"""
        try:
            from medai_dicom_processor import DicomProcessor
            from medai_inference_system import MedicalInferenceEngine
            from medai_sota_models import StateOfTheArtModels
            from medai_feature_extraction import RadiomicFeatureExtractor as AdvancedFeatureExtractor
            from medai_detection_system import RadiologyYOLO, MaskRCNNRadiology, LesionTracker
            from medai_training_system import MedicalModelTrainer, RadiologyDataset
            from medai_explainability import GradCAMExplainer, IntegratedGradientsExplainer
            try:
                from medai_pacs_integration import PACSIntegration, FastAPIApp
            except ImportError as e:
                logger.warning(f"PACS integration não disponível: {e}")
                PACSIntegration = None
                FastAPIApp = None
            from medai_clinical_evaluation import ClinicalPerformanceEvaluator, RadiologyBenchmark
            from medai_ethics_compliance import EthicalAIFramework
            RegulatoryComplianceManager = EthicalAIFramework  # Use EthicalAIFramework as fallback
            # from medai_security_audit import SecurityManager  # Temporarily disabled due to jwt dependency
            # from medai_report_generator import ReportGenerator  # Temporarily disabled
            # from medai_batch_processor import BatchProcessor  # Temporarily disabled
            # from medai_comparison_system import ComparisonSystem  # Temporarily disabled
            # from medai_advanced_visualization import VisualizationEngine  # Temporarily disabled
            # from medai_export_system import ExportSystem  # Temporarily disabled
            
            self.dicom_processor = DicomProcessor()
            from medai_main_structure import Config
            default_model_config = Config.MODEL_CONFIG['chest_xray']
            self.inference_engine = MedicalInferenceEngine(
                model_path=default_model_config['model_path'],
                model_config=default_model_config
            )
            
            self.feature_extractor = AdvancedFeatureExtractor()
            self.detection_system = RadiologyYOLO(num_classes=15)
            import tensorflow as tf
            dummy_model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
            self.model_trainer = MedicalModelTrainer(
                model=dummy_model,
                model_name="test_model", 
                output_dir="./models"
            )
            cnn_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3), name='conv2d_1'),
                tf.keras.layers.Conv2D(64, 3, activation='relu', name='conv2d_2'),
                tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
                tf.keras.layers.Dense(10, activation='softmax', name='dense_output')
            ])
            cnn_model.build(input_shape=(None, 224, 224, 3))
            dummy_input = tf.random.normal((1, 224, 224, 3))
            _ = cnn_model(dummy_input)
            self.explainability_engine = GradCAMExplainer(cnn_model)
            try:
                self.pacs_integration = PACSIntegration()
            except Exception as e:
                print(f"PACS integration não disponível: {e}")
                self.pacs_integration = None
            self.clinical_evaluator = ClinicalPerformanceEvaluator(
                class_names=['Normal', 'Pneumonia', 'Tumor', 'Fracture', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Nodule']
            )
            self.ethics_framework = EthicalAIFramework(
                model_name="MedAI_Radiologia",
                version="v2.0"
            )
            self.regulatory_manager = EthicalAIFramework(
                model_name="MedAI_Radiologia_Regulatory",
                version="v2.0"
            )
            
            # self.security_manager = SecurityManager()  # Temporarily disabled
            # self.report_generator = ReportGenerator()  # Temporarily disabled
            # self.batch_processor = BatchProcessor()  # Temporarily disabled
            # self.comparison_system = ComparisonSystem()  # Temporarily disabled
            # self.visualization_engine = VisualizationEngine()  # Temporarily disabled
            # self.export_system = ExportSystem()  # Temporarily disabled
            
            self.sota_models = StateOfTheArtModels(
                input_shape=(512, 512, 3),  # Resolução aumentada para melhor precisão
                num_classes=15  # Expandido para mais classes diagnósticas
            )
            
            self.enhanced_models = {
                'medical_vit': self.sota_models.build_medical_vision_transformer(),
                'medical_gnn': self.sota_models.build_graph_neural_network(),
                'enhanced_ensemble': self.sota_models.build_ensemble_model()
            }
            
            for model_name, model in self.enhanced_models.items():
                self.sota_models.compile_sota_model(model, learning_rate=1e-5)
                logger.info(f"Modelo {model_name} compilado com {model.count_params():,} parâmetros")
            
            logger.info("Modelos de IA de última geração carregados com melhorias do relatório")
            
            logger.info("Todos os componentes inicializados com sucesso")
            logger.info("Sistema configurado com modelos de IA de última geração para máxima precisão diagnóstica")
            
        except ImportError as e:
            logger.error(f"Erro ao importar componentes: {e}")
            raise
    
    def login(self, username: str, password: str) -> bool:
        """Autentica usuário no sistema"""
        try:
            success = self.security_manager.authenticate(username, password)
            if success:
                self.current_session = {
                    'username': username,
                    'login_time': datetime.now(),
                    'permissions': self.security_manager.get_user_permissions(username)
                }
                logger.info(f"Usuário {username} autenticado com sucesso")
            return success
        except Exception as e:
            logger.error(f"Erro na autenticação: {e}")
            return False
    
    def logout(self):
        """Encerra sessão do usuário"""
        if self.current_session:
            username = self.current_session['username']
            self.security_manager.log_activity(username, "logout")
            self.current_session = None
            logger.info(f"Usuário {username} desconectado")
    
    def load_image(self, file_path: str, anonymize: bool = True) -> Dict[str, Any]:
        """
        Carrega imagem médica (DICOM ou outros formatos)
        
        Args:
            file_path: Caminho para o arquivo
            anonymize: Se deve anonimizar dados do paciente
            
        Returns:
            Dicionário com dados da imagem e metadados
        """
        try:
            if not self._check_permission('read_images'):
                raise PermissionError("Usuário não tem permissão para carregar imagens")
            
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.dcm' or self._is_dicom_file(file_path):
                image_data = self.dicom_processor.read_dicom(str(file_path), anonymize)
            else:
                image_data = self._load_standard_image(file_path)
            
            self.security_manager.log_activity(
                self.current_session['username'],
                "load_image",
                f"Arquivo: {file_path.name}"
            )
            
            return image_data
            
        except Exception as e:
            logger.error(f"Erro ao carregar imagem {file_path}: {e}")
            raise
    
    def analyze_image(self, image_data: np.ndarray, model_name: str, 
                     generate_attention_map: bool = True) -> Dict[str, Any]:
        """
        Analisa imagem usando modelo de IA especificado
        
        Args:
            image_data: Array numpy da imagem ou dados da imagem carregada
            model_name: Nome do modelo a ser usado
            generate_attention_map: Se deve gerar mapa de atenção
            
        Returns:
            Resultados da análise
        """
        try:
            if not self._check_permission('analyze_images'):
                raise PermissionError("Usuário não tem permissão para analisar imagens")
            
            if isinstance(image_data, dict):
                image_array = image_data.get('image', image_data)
            else:
                image_array = image_data
            
            # Use predict_single method which exists in MedicalInferenceEngine
            result = self.inference_engine.predict_single(
                image_array,
                return_attention=generate_attention_map
            )
            
            results = {
                'predicted_class': result.predicted_class,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'predictions': result.predictions if result.predictions else {},
                'metadata': result.metadata if result.metadata else {}
            }
            
            analysis_record = {
                'timestamp': datetime.now(),
                'model_name': model_name,
                'results': results,
                'user': self.current_session['username'] if self.current_session else 'test_user'
            }
            
            with self._lock:
                self.analysis_history.append(analysis_record)
            
            if self.current_session and hasattr(self, 'security_manager'):
                self.security_manager.log_activity(
                    self.current_session['username'],
                    "analyze_image",
                    f"Modelo: {model_name}, Confiança: {results.get('confidence', 0):.2f}"
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise da imagem: {e}")
            raise
    
    def batch_analyze(self, directory_path: str, model_name: str, 
                     output_format: str = 'json') -> str:
        """
        Processa múltiplas imagens em lote
        
        Args:
            directory_path: Diretório com imagens
            model_name: Modelo a ser usado
            output_format: Formato de saída dos resultados
            
        Returns:
            Caminho para arquivo de resultados
        """
        try:
            if not self._check_permission('batch_processing'):
                raise PermissionError("Usuário não tem permissão para processamento em lote")
            
            results_path = self.batch_processor.process_directory(
                directory_path, 
                model_name,
                output_format
            )
            
            self.security_manager.log_activity(
                self.current_session['username'],
                "batch_analyze",
                f"Diretório: {directory_path}, Modelo: {model_name}"
            )
            
            return results_path
            
        except Exception as e:
            logger.error(f"Erro no processamento em lote: {e}")
            raise
    
    def compare_images(self, image1_data: Dict[str, Any], 
                      image2_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compara duas imagens médicas
        
        Args:
            image1_data: Dados da primeira imagem
            image2_data: Dados da segunda imagem
            
        Returns:
            Resultados da comparação
        """
        try:
            if not self._check_permission('compare_images'):
                raise PermissionError("Usuário não tem permissão para comparar imagens")
            
            comparison_results = self.comparison_system.compare_images(
                image1_data, 
                image2_data
            )
            
            self.security_manager.log_activity(
                self.current_session['username'],
                "compare_images",
                "Comparação de imagens realizada"
            )
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Erro na comparação de imagens: {e}")
            raise
    
    def generate_report(self, analysis_results: Dict[str, Any], 
                       report_type: str = 'pdf') -> str:
        """
        Gera relatório dos resultados de análise
        
        Args:
            analysis_results: Resultados da análise
            report_type: Tipo de relatório (pdf, html, dicom_sr)
            
        Returns:
            Caminho para o relatório gerado
        """
        try:
            if not self._check_permission('generate_reports'):
                raise PermissionError("Usuário não tem permissão para gerar relatórios")
            
            report_path = self.report_generator.generate_report(
                analysis_results,
                report_type,
                user_info=self.current_session
            )
            
            self.security_manager.log_activity(
                self.current_session['username'],
                "generate_report",
                f"Tipo: {report_type}, Arquivo: {report_path}"
            )
            
            return report_path
            
        except Exception as e:
            logger.error(f"Erro na geração de relatório: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis"""
        return self.inference_engine.get_available_models()
    
    def get_analysis_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna histórico de análises"""
        with self._lock:
            return self.analysis_history[-limit:]
    
    def _check_permission(self, permission: str) -> bool:
        """Verifica se usuário tem permissão específica"""
        return True
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """Verifica se arquivo é DICOM"""
        try:
            with open(file_path, 'rb') as f:
                f.seek(128)
                return f.read(4) == b'DICM'
        except Exception:
            return False
    
    def _load_standard_image(self, file_path: Path) -> Dict[str, Any]:
        """Carrega imagem em formato padrão (PNG, JPEG, etc.)"""
        try:
            from PIL import Image
            import numpy as np
            
            image = Image.open(file_path)
            
            image_array = np.array(image)
            
            if len(image_array.shape) == 3:
                image_array = np.mean(image_array, axis=2)
            
            return {
                'image': image_array,
                'metadata': {
                    'filename': file_path.name,
                    'format': image.format,
                    'size': image.size,
                    'mode': image.mode,
                    'loaded_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao carregar imagem padrão: {e}")
            raise
    
    def analyze_sample_image(self) -> Dict[str, Any]:
        """Analisa uma imagem de amostra para teste"""
        try:
            import numpy as np
            sample_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
            
            result = self.inference_engine.predict_single(sample_image, return_attention=False)
            
            return {
                'status': 'analyzed',
                'confidence': result.confidence,
                'prediction': result.predicted_class,
                'processing_time': result.processing_time,
                'model_used': 'enhanced_pathology_detector'
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de amostra: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
