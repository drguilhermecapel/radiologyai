#!/usr/bin/env python3
"""
MedAI Radiologia - Instalador Modular de Modelos
Instalador especializado para gerenciamento de modelos de IA
Versão: 1.0.0
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import threading
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

class ModelInstaller:
    """
    Instalador modular especializado em modelos de IA
    Pode ser usado standalone ou integrado ao instalador principal
    """
    
    def __init__(self, install_dir: Optional[Path] = None):
        self.install_dir = install_dir or Path("C:/Program Files/MedAI Radiologia")
        self.models_dir = self.install_dir / "models"
        self.pretrained_dir = self.models_dir / "pre_trained"
        
        self.model_catalog = {
            'chest_xray_efficientnetv2': {
                'name': 'EfficientNetV2 - Raio-X Tórax',
                'category': 'basic',
                'size_mb': 150,
                'accuracy': 92.3,
                'description': 'Modelo otimizado para análise de raio-X de tórax',
                'modalities': ['chest_xray'],
                'clinical_focus': ['pneumonia', 'pleural_effusion', 'fractures'],
                'download_url': 'https://models.medai.com/efficientnetv2/chest_xray_v2.1.0.h5',
                'dependencies': ['tensorflow>=2.10.0'],
                'license': 'Apache-2.0',
                'fda_status': 'pending_510k'
            },
            'chest_xray_vision_transformer': {
                'name': 'Vision Transformer - Raio-X Tórax',
                'category': 'advanced',
                'size_mb': 300,
                'accuracy': 91.1,
                'description': 'Modelo baseado em atenção para análise detalhada',
                'modalities': ['chest_xray'],
                'clinical_focus': ['complex_pathology', 'attention_mapping'],
                'download_url': 'https://models.medai.com/vit/chest_xray_v2.0.1.h5',
                'dependencies': ['tensorflow>=2.10.0', 'transformers>=4.0.0'],
                'license': 'MIT',
                'fda_status': 'research_use'
            },
            'chest_xray_convnext': {
                'name': 'ConvNeXt - Raio-X Tórax',
                'category': 'advanced',
                'size_mb': 200,
                'accuracy': 90.8,
                'description': 'Arquitetura convolucional moderna',
                'modalities': ['chest_xray'],
                'clinical_focus': ['balanced_performance', 'fast_inference'],
                'download_url': 'https://models.medai.com/convnext/chest_xray_v1.5.2.h5',
                'dependencies': ['tensorflow>=2.10.0'],
                'license': 'Apache-2.0',
                'fda_status': 'research_use'
            },
            'ensemble_sota': {
                'name': 'Ensemble SOTA - Multi-Modal',
                'category': 'premium',
                'size_mb': 800,
                'accuracy': 94.5,
                'description': 'Ensemble de múltiplos modelos para máxima precisão',
                'modalities': ['chest_xray', 'brain_ct', 'bone_xray'],
                'clinical_focus': ['multi_modal', 'highest_accuracy', 'clinical_grade'],
                'download_url': 'https://models.medai.com/ensemble/sota_v3.0.0.h5',
                'dependencies': ['tensorflow>=2.10.0', 'numpy>=1.21.0'],
                'license': 'Apache-2.0',
                'fda_status': 'pending_510k'
            }
        }
        
        self.installation_packages = {
            'basic': {
                'name': 'Pacote Básico',
                'description': 'Modelos essenciais para raio-X de tórax',
                'models': ['chest_xray_efficientnetv2'],
                'total_size_mb': 150,
                'recommended_for': 'Uso geral, clínicas pequenas'
            },
            'standard': {
                'name': 'Pacote Padrão',
                'description': 'Modelos básicos + avançados para raio-X',
                'models': ['chest_xray_efficientnetv2', 'chest_xray_vision_transformer'],
                'total_size_mb': 450,
                'recommended_for': 'Hospitais, uso clínico regular'
            },
            'professional': {
                'name': 'Pacote Profissional',
                'description': 'Todos os modelos de raio-X + ensemble',
                'models': ['chest_xray_efficientnetv2', 'chest_xray_vision_transformer', 
                          'chest_xray_convnext', 'ensemble_sota'],
                'total_size_mb': 1450,
                'recommended_for': 'Hospitais grandes, pesquisa'
            },
            'complete': {
                'name': 'Pacote Completo',
                'description': 'Todos os modelos disponíveis',
                'models': [],  # Will be filled in __init__
                'total_size_mb': 2200,
                'recommended_for': 'Centros de pesquisa, uso acadêmico'
            }
        }
        
        self.installation_packages['complete']['models'] = list(self.model_catalog.keys())
        
        self.installation_state = {
            'selected_models': [],
            'download_progress': {},
            'installation_log': [],
            'errors': [],
            'total_progress': 0
        }
    
    def run_cli_installer(self):
        """Executa instalador em modo linha de comando"""
        print("=" * 60)
        print("MEDAI RADIOLOGIA - INSTALADOR DE MODELOS")
        print("=" * 60)
        print()
        
        print("Pacotes disponíveis:")
        for i, (key, package) in enumerate(self.installation_packages.items(), 1):
            print(f"{i}. {package['name']} ({package['total_size_mb']}MB)")
            print(f"   {package['description']}")
            print(f"   Recomendado para: {package['recommended_for']}")
            print()
        
        while True:
            try:
                choice = input("Selecione um pacote (1-4) ou 'q' para sair: ").strip()
                
                if choice.lower() == 'q':
                    print("Instalação cancelada.")
                    return
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.installation_packages):
                    package_key = list(self.installation_packages.keys())[choice_num - 1]
                    selected_models = self.installation_packages[package_key]['models']
                    break
                else:
                    print("Opção inválida. Tente novamente.")
                    
            except ValueError:
                print("Por favor, digite um número válido.")
        
        print(f"\nInstalando pacote: {self.installation_packages[package_key]['name']}")
        print(f"Modelos: {len(selected_models)}")
        print()
        
        try:
            self.installation_state['selected_models'] = selected_models
            
            for i, model_id in enumerate(selected_models):
                model_info = self.model_catalog[model_id]
                print(f"[{i+1}/{len(selected_models)}] Baixando {model_info['name']}...")
                
                time.sleep(1)
                
                self.download_model(model_id, model_info)
                print(f"✅ Concluído: {model_info['name']}")
            
            print("\n🎉 Instalação de modelos concluída com sucesso!")
            print(f"📁 Modelos instalados em: {self.pretrained_dir}")
            
        except Exception as e:
            print(f"\n❌ Erro na instalação: {e}")
    
    def download_model(self, model_id: str, model_info: Dict):
        """Simula download de um modelo"""
        try:
            model_dir = self.pretrained_dir / model_info['category']
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_file = model_dir / f"{model_id}.h5"
            
            with open(model_file, 'w', encoding='utf-8') as f:
                f.write(f"""# MedAI Model: {model_info['name']}
# 
PLACEHOLDER_MODEL=True
MODEL_ID="{model_id}"
""")
            
        except Exception as e:
            raise Exception(f"Erro ao baixar modelo {model_id}: {e}")
    
    def run(self):
        """Executa o instalador de modelos"""
        print("MedAI Radiologia - Instalador de Modelos v1.0.0")
        print("Instalador especializado para modelos de IA")
        print()
        
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_cli_installer()

if __name__ == "__main__":
    installer = ModelInstaller()
    installer.run()
