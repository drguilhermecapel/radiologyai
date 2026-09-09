#!/usr/bin/env python3
"""
Test script to validate ensemble model configuration
Ensures proper initialization and functionality of the ensemble system
"""

import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from medai_sota_models import StateOfTheArtModels

def test_ensemble_configuration():
    """Test ensemble model configuration and initialization"""
    print("🔗 Testando configuração do modelo ensemble...")
    
    config_path = "models/advanced_ensemble_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Arquivo de configuração não encontrado: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Configuração carregada: {config['ensemble_type']}")
    
    try:
        input_shape = tuple(config['input_shape'])
        num_classes = config['num_classes']
        
        sota_models = StateOfTheArtModels(
            input_shape=input_shape,
            num_classes=num_classes
        )
        print(f"✅ StateOfTheArtModels inicializado com sucesso")
        print(f"   Input shape: {input_shape}")
        print(f"   Número de classes: {num_classes}")
        
    except Exception as e:
        print(f"❌ Erro na inicialização do StateOfTheArtModels: {e}")
        return False
    
    required_keys = [
        'ensemble_type', 'input_shape', 'num_classes', 'class_names',
        'models', 'fusion_strategy', 'clinical_validation'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Chave obrigatória ausente na configuração: {key}")
            return False
    
    print("✅ Estrutura da configuração validada")
    
    models_config = config['models']
    expected_models = ['EfficientNetV2', 'VisionTransformer', 'ConvNeXt']
    
    for model_name in expected_models:
        if model_name not in models_config:
            print(f"❌ Modelo ausente na configuração: {model_name}")
            return False
        
        model_config = models_config[model_name]
        required_model_keys = ['weight', 'clinical_score', 'model_path', 'architecture', 'specialization']
        
        for key in required_model_keys:
            if key not in model_config:
                print(f"❌ Chave ausente na configuração do modelo {model_name}: {key}")
                return False
    
    print("✅ Configuração dos modelos validada")
    
    total_weight = sum(models_config[model]['weight'] for model in models_config)
    if abs(total_weight - 1.0) > 0.01:
        print(f"❌ Soma dos pesos não é 1.0: {total_weight}")
        return False
    
    print(f"✅ Soma dos pesos validada: {total_weight:.3f}")
    
    clinical_validation = config['clinical_validation']
    required_thresholds = ['minimum_sensitivity', 'minimum_specificity', 'minimum_accuracy', 'minimum_auc']
    
    for threshold in required_thresholds:
        if threshold not in clinical_validation:
            print(f"❌ Threshold clínico ausente: {threshold}")
            return False
        
        value = clinical_validation[threshold]
        if not (0.0 <= value <= 1.0):
            print(f"❌ Threshold clínico inválido {threshold}: {value}")
            return False
    
    print("✅ Thresholds clínicos validados")
    
    return True

def test_ensemble_model_creation():
    """Test ensemble model creation functionality"""
    print("\n🏗️ Testando criação do modelo ensemble...")
    
    try:
        sota_models = StateOfTheArtModels(
            input_shape=(512, 512, 3),
            num_classes=5
        )
        
        ensemble_model = sota_models.build_ensemble_model()
        
        if ensemble_model is not None:
            print("✅ Modelo ensemble criado com sucesso")
            print(f"   Tipo: {type(ensemble_model)}")
            
            try:
                if hasattr(ensemble_model, 'summary'):
                    print("✅ Modelo possui método summary")
                else:
                    print("ℹ️ Modelo não possui método summary (normal para alguns tipos)")
            except Exception as e:
                print(f"⚠️ Erro ao acessar summary do modelo: {e}")
            
            return True
        else:
            print("❌ Modelo ensemble retornou None")
            return False
            
    except Exception as e:
        print(f"❌ Erro na criação do modelo ensemble: {e}")
        return False

if __name__ == "__main__":
    print("🔗 MedAI - Teste de Configuração do Ensemble")
    print("=" * 60)
    
    config_valid = test_ensemble_configuration()
    
    model_valid = test_ensemble_model_creation()
    
    print(f"\n{'='*60}")
    print("🎯 RESULTADO FINAL DOS TESTES")
    print(f"{'='*60}")
    
    if config_valid and model_valid:
        print("✅ CONFIGURAÇÃO DO ENSEMBLE VALIDADA COM SUCESSO")
        print("✅ Ensemble pronto para treinamento e uso clínico")
        sys.exit(0)
    else:
        print("❌ FALHA NA VALIDAÇÃO DO ENSEMBLE")
        if not config_valid:
            print("❌ Problemas na configuração")
        if not model_valid:
            print("❌ Problemas na criação do modelo")
        sys.exit(1)
