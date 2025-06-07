# MedAI Radiologia - Sistema de Análise de Imagens Médicas por IA
# Arquivo: main.py

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Configuração de diretórios
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
TEMP_DIR = BASE_DIR / "temp"

# Criar diretórios necessários
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'medai_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('MedAI')

# Configuração global do sistema
class Config:
    """Configurações globais do sistema MedAI"""
    
    # Configurações de aplicação
    APP_NAME = "MedAI Radiologia"
    APP_VERSION = "1.0.0"
    
    # Configurações de modelos - Estado da Arte para máxima precisão
    MODEL_CONFIG = {
        'chest_xray': {
            'model_path': MODELS_DIR / 'chest_xray_efficientnetv2_model.h5',
            'input_size': (384, 384),  # Resolução aumentada para melhor precisão
            'classes': ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculose', 'Cardiomegalia', 'Derrame Pleural'],
            'threshold': 0.85,  # Threshold mais alto para maior confiabilidade
            'architecture': 'efficientnetv2'  # Modelo de última geração
        },
        'brain_ct': {
            'model_path': MODELS_DIR / 'brain_ct_vision_transformer_model.h5',
            'input_size': (384, 384),  # Resolução aumentada
            'classes': ['Normal', 'Hemorragia', 'Isquemia', 'Tumor', 'Edema', 'Hidrocefalia'],
            'threshold': 0.90,  # Threshold alto para diagnósticos críticos
            'architecture': 'vision_transformer'  # ViT para máxima precisão
        },
        'bone_fracture': {
            'model_path': MODELS_DIR / 'bone_fracture_convnext_model.h5',
            'input_size': (384, 384),  # Resolução aumentada
            'classes': ['Normal', 'Fratura', 'Luxação', 'Osteoporose', 'Artrite', 'Osteomielite'],
            'threshold': 0.82,  # Threshold otimizado para detecção óssea
            'architecture': 'convnext'  # ConvNeXt para análise óssea
        }
    }
    
    # Configurações DICOM
    DICOM_TAGS = {
        'PatientName': (0x0010, 0x0010),
        'PatientID': (0x0010, 0x0020),
        'StudyDate': (0x0008, 0x0020),
        'Modality': (0x0008, 0x0060),
        'StudyDescription': (0x0008, 0x1030),
        'SeriesDescription': (0x0008, 0x103E)
    }
    
    # Configurações de segurança
    ANONYMIZE_DATA = True
    ENCRYPTION_KEY = os.environ.get('MEDAI_ENCRYPTION_KEY', 'default_key')
    
    # Configurações de performance
    MAX_BATCH_SIZE = 32
    GPU_ENABLED = True
    CACHE_SIZE = 100  # Número de imagens em cache

# requirements.txt
"""
numpy==1.24.3
tensorflow==2.13.0
pydicom==2.4.3
opencv-python==4.8.1
Pillow==10.1.0
PyQt5==5.15.9
matplotlib==3.7.2
scikit-learn==1.3.0
pandas==2.0.3
h5py==3.9.0
SimpleITK==2.3.0
nibabel==5.1.0
scikit-image==0.21.0
reportlab==4.0.4
cryptography==41.0.4
python-gdcm==3.0.22
pyqtgraph==0.13.3
vtk==9.2.6
"""

if __name__ == "__main__":
    logger.info(f"Iniciando {Config.APP_NAME} v{Config.APP_VERSION}")
    logger.info(f"Diretório base: {BASE_DIR}")
    logger.info("Sistema configurado com sucesso")
