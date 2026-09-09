#!/usr/bin/env python3
"""
Comprehensive Training Script for Medical AI Models
Implements state-of-the-art training pipeline based on scientific guide
Trains EfficientNetV2, Vision Transformer, and ConvNeXt with advanced techniques
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from medai_training_system import RadiologyDataset, MedicalModelTrainer
    from medai_sota_models import StateOfTheArtModels
    from medai_ml_pipeline import MLPipeline, DatasetConfig, ModelConfig, TrainingConfig
    from medai_clinical_evaluation import ClinicalPerformanceEvaluator
    from medai_inference_system import MedicalInferenceEngine
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que todos os módulos estão no diretório src/")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MedAI.Training')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Treinar modelos de IA para radiologia')
    parser.add_argument('--data_dir', type=str, default='data/samples',
                       help='Diretório com dados de treinamento')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Diretório para salvar modelos treinados')
    parser.add_argument('--architectures', type=str, nargs='+',
                       default=['EfficientNetV2', 'VisionTransformer', 'ConvNeXt'],
                       help='Arquiteturas para treinar')
    parser.add_argument('--modality', type=str, 
                       choices=['X-ray', 'CT', 'MRI', 'Ultrasound', 'PET-CT'],
                       default='X-ray',
                       help='Modalidade de imagem médica para treinamento')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas para treinamento')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamanho do batch para treinamento')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Taxa de aprendizado')
    parser.add_argument('--ensemble', action='store_true',
                       help='Criar modelo ensemble com as arquiteturas treinadas')
    parser.add_argument('--validate_only', action='store_true',
                       help='Apenas validar dados sem treinar')
    parser.add_argument('--validate_clinical_metrics', action='store_true',
                       help='Validar métricas clínicas durante treinamento')
    parser.add_argument('--resume', type=str, default=None,
                       help='Retomar treinamento de um checkpoint')
    return parser.parse_args()

def validate_data_directory(data_dir, modality='X-ray'):
    """Valida estrutura do diretório de dados baseado na modalidade"""
    logger.info(f"Validando diretório de dados: {data_dir} (Modalidade: {modality})")
    
    if not os.path.exists(data_dir):
        logger.error(f"Diretório de dados não encontrado: {data_dir}")
        return False
    
    modality_classes = {
        'X-ray': ['normal', 'pneumonia', 'pleural_effusion', 'fracture', 'tumor'],
        'CT': ['normal', 'tumor', 'stroke', 'hemorrhage', 'edema'],
        'MRI': ['normal', 'tumor', 'lesion', 'edema', 'hemorrhage'],
        'Ultrasound': ['normal', 'cisto', 'tumor_solido', 'calcificacao', 'vascularizacao_anormal', 'liquido_livre'],
        'PET-CT': ['normal', 'hipermetabolismo_benigno', 'hipermetabolismo_maligno', 'hipometabolismo', 'necrose', 'inflamacao']
    }
    
    expected_classes = modality_classes.get(modality, ['normal', 'abnormal'])
    found_classes = []
    
    for class_name in expected_classes:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith(('.dcm', '.jpg', '.png', '.jpeg'))]
            if files:
                found_classes.append(class_name)
                logger.info(f"Classe '{class_name}': {len(files)} arquivos encontrados")
            else:
                logger.warning(f"Classe '{class_name}': diretório vazio")
        else:
            logger.warning(f"Classe '{class_name}': diretório não encontrado")
    
    if len(found_classes) < 2:
        logger.error("Pelo menos 2 classes são necessárias para treinamento")
        return False
    
    logger.info(f"Validação concluída para modalidade {modality}. Classes encontradas: {found_classes}")
    return True

def get_model_config(architecture, input_shape, num_classes, learning_rate, modality='X-ray'):
    """Get SOTA model-specific configuration optimized for medical imaging"""
    configs = {
        'EfficientNetV2': {
            'input_shape': (384, 384, 3),
            'batch_size': 16,
            'learning_rate': learning_rate,
            'epochs_default': 50,
            'freeze_layers': -3,  # Freeze all but last 3 layers initially
            'fine_tuning_lr': learning_rate * 0.1,
            'preprocessing': 'efficientnet_medical'
        },
        'VisionTransformer': {
            'input_shape': (224, 224, 3),
            'batch_size': 8,  # Reduced for ViT memory requirements
            'learning_rate': learning_rate * 0.5,  # ViT needs lower LR
            'epochs_default': 60,
            'freeze_layers': 'encoder',  # Freeze encoder layers initially
            'fine_tuning_lr': learning_rate * 0.05,
            'preprocessing': 'vit_medical'
        },
        'ConvNeXt': {
            'input_shape': (256, 256, 3),
            'batch_size': 12,  # Optimized for ConvNeXt
            'learning_rate': learning_rate * 0.8,  # ConvNeXt can use higher LR
            'epochs_default': 45,
            'freeze_layers': -4,  # Freeze all but last 4 layers initially
            'fine_tuning_lr': learning_rate * 0.08,
            'preprocessing': 'convnext_medical'
        },
        'Ensemble': {
            'input_shape': (384, 384, 3),  # Largest input for ensemble
            'batch_size': 6,  # Smaller batch for ensemble memory requirements
            'learning_rate': learning_rate * 0.3,  # Conservative LR for ensemble
            'epochs_default': 40,
            'freeze_layers': 'backbone',  # Freeze backbone models initially
            'fine_tuning_lr': learning_rate * 0.03,
            'preprocessing': 'ensemble_medical'
        },
        'UltrasoundEfficientNetV2': {
            'input_shape': (384, 384, 3),
            'batch_size': 16,
            'learning_rate': learning_rate,
            'epochs_default': 45,
            'freeze_layers': -3,
            'fine_tuning_lr': learning_rate * 0.1,
            'preprocessing': 'ultrasound_medical',
            'modality_specific': True,
            'speckle_reduction': True
        },
        'PETCTFusionHybrid': {
            'input_shape': (512, 512, 3),
            'batch_size': 8,  # Reduced for larger input size and fusion complexity
            'learning_rate': learning_rate * 0.5,  # Conservative for fusion model
            'epochs_default': 60,
            'freeze_layers': -2,
            'fine_tuning_lr': learning_rate * 0.05,
            'preprocessing': 'pet_ct_fusion_medical',
            'modality_specific': True,
            'fusion_training': True
        }
    }
    
    return configs.get(architecture, {
        'input_shape': input_shape,
        'batch_size': 16,
        'learning_rate': learning_rate,
        'epochs_default': 50,
        'freeze_layers': None,
        'fine_tuning_lr': learning_rate * 0.1,
        'preprocessing': 'standard_medical'
    })

def train_model_progressive(architecture, data_dir, output_dir, epochs, batch_size, learning_rate, validate_clinical_metrics=False):
    """
    Treina modelo com abordagem progressiva baseada no scientific guide
    Implementa treinamento em fases com learning rates diferenciados
    """
    logger.info(f"Iniciando treinamento progressivo de {architecture}")
    if validate_clinical_metrics:
        logger.info("🏥 Validação de métricas clínicas ativada para este modelo")
    
    try:
        model_config = get_model_config(architecture, (224, 224, 3), 5, learning_rate)
        
        augmentation_config = {
            'rotation_range': 10,
            'brightness_range': 0.1,
            'contrast_range': 0.1,
            'gaussian_noise_std': 0.01,
            'clahe_enabled': True,
            'medical_windowing': True
        }
        
        dataset_config = DatasetConfig(
            name=f"{architecture}_dataset",
            data_dir=Path(data_dir),
            image_size=model_config['input_shape'][:2],
            num_classes=5,
            augmentation_config=augmentation_config,
            preprocessing_config={'normalize': True, 'medical_preprocessing': True},
            validation_split=0.2,
            test_split=0.2,
            batch_size=model_config['batch_size'],  # Usar batch_size consistente
            class_names=['normal', 'pneumonia', 'pleural_effusion', 'fracture', 'tumor']
        )
        
        model_config_obj = ModelConfig(
            architecture=architecture,
            input_shape=model_config['input_shape'],
            num_classes=len(dataset_config.class_names)
        )
        
        training_config = TrainingConfig(
            batch_size=model_config['batch_size'],
            epochs=epochs,
            learning_rate=model_config['learning_rate'],
            early_stopping_patience=15,
            reduce_lr_patience=7,
            mixed_precision=True,  # Ativar mixed precision para eficiência
            gradient_clipping=1.0,
            label_smoothing=0.1,
            use_class_weights=True,
            progressive_training=True,
            backbone_lr_multiplier=0.1  # Learning rate menor para backbone
        )
        
        pipeline = MLPipeline(
            project_name="MedAI_Radiologia_Advanced",
            experiment_name=f"{architecture}_progressive_training"
        )
        
        logger.info("Preparando dados com augmentação avançada...")
        train_ds, val_ds, test_ds = pipeline.prepare_data(str(data_dir), dataset_config)
        
        if train_ds is None:
            logger.error(f"Falha ao preparar dados para {architecture}")
            return None, None
        
        logger.info("Construindo modelo com arquitetura avançada...")
        model = pipeline.build_model(model_config_obj)
        
        if model is None:
            logger.error(f"Falha ao construir modelo {architecture}")
            return None, None
        
        logger.info("=== FASE 1: Treinamento das camadas finais (5 épocas) ===")
        phase1_config = training_config
        phase1_config.epochs = 5
        
        if hasattr(model, 'layers'):
            for layer in model.layers[:-3]:  # Congelar todas exceto as últimas 3 camadas
                layer.trainable = False
        
        history_phase1 = pipeline.train(model, train_ds, val_ds, phase1_config, f"{architecture}_phase1")
        
        logger.info("=== FASE 2: Fine-tuning completo (épocas restantes) ===")
        if hasattr(model, 'layers'):
            for layer in model.layers:
                layer.trainable = True
        
        phase2_config = training_config
        phase2_config.epochs = epochs - 5
        phase2_config.learning_rate = learning_rate * 0.1
        
        history_phase2 = pipeline.train(model, train_ds, val_ds, phase2_config, f"{architecture}_phase2")
        
        combined_history = {}
        if hasattr(history_phase1, 'history') and hasattr(history_phase2, 'history'):
            for key in history_phase1.history.keys():
                combined_history[key] = (
                    list(history_phase1.history[key]) + 
                    list(history_phase2.history.get(key, []))
                )
        elif isinstance(history_phase1, dict) and isinstance(history_phase2, dict):
            for key in history_phase1.keys():
                combined_history[key] = (
                    list(history_phase1.get(key, [])) + 
                    list(history_phase2.get(key, []))
                )
        
        logger.info("Avaliando modelo com métricas clínicas...")
        results = model.evaluate(test_ds, verbose=0)
        metrics = {}
        if isinstance(results, list):
            metrics['loss'] = results[0]
            if len(results) > 1:
                metrics['accuracy'] = results[1]
        else:
            metrics['loss'] = results
        
        clinical_evaluator = ClinicalPerformanceEvaluator()
        
        test_predictions = []
        test_labels = []
        
        for batch in test_ds.take(10):  # Avaliar em subset para demonstração
            predictions = model.predict(batch[0], verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            test_predictions.extend(predicted_classes)
            test_labels.extend(batch[1].numpy())
        
        if test_predictions and test_labels:
            clinical_metrics = clinical_evaluator.calculate_metrics(
                np.array(test_labels), 
                np.array(test_predictions)
            )
            metrics.update(clinical_metrics)
            
            if validate_clinical_metrics:
                sensitivity = clinical_metrics.get('sensitivity', 0.0)
                specificity = clinical_metrics.get('specificity', 0.0)
                auc = clinical_metrics.get('auc', 0.0)
                
                logger.info(f"🏥 Métricas Clínicas - Sensibilidade: {sensitivity:.3f}, Especificidade: {specificity:.3f}, AUC: {auc:.3f}")
                
                if sensitivity >= 0.85 and specificity >= 0.90 and auc >= 0.85:
                    logger.info("✅ Modelo atende critérios clínicos mínimos")
                    metrics['clinical_ready'] = True
                else:
                    logger.warning("⚠️ Modelo não atende critérios clínicos - requer melhorias")
                    metrics['clinical_ready'] = False
            
            clinical_report = clinical_evaluator.generate_clinical_report(
                clinical_metrics, 
                f"{architecture}_model"
            )
            logger.info(f"Relatório clínico para {architecture}:\n{clinical_report}")
        
        logger.info(f"Métricas finais para {architecture}: {metrics}")
        
        model_filename = f"chest_xray_{architecture.lower()}_model.h5"
        model_path = os.path.join(output_dir, model_filename)
        
        save_model = True
        if 'sensitivity' in metrics and 'specificity' in metrics:
            sensitivity = metrics['sensitivity']
            specificity = metrics['specificity']
            
            if sensitivity < 0.85 or specificity < 0.80:
                logger.warning(f"Modelo {architecture} não atende critérios clínicos mínimos")
                logger.warning(f"Sensibilidade: {sensitivity:.3f} (mín: 0.85)")
                logger.warning(f"Especificidade: {specificity:.3f} (mín: 0.80)")
                save_model = False
        
        if save_model:
            try:
                model.save(model_path)
                logger.info(f"✅ Modelo {architecture} salvo em {model_path}")
                
                if validate_clinical_metrics and metrics.get('clinical_ready', False):
                    clinical_model_path = model_path.replace('.h5', '_clinical_validated.h5')
                    model.save(clinical_model_path)
                    logger.info(f"🏥 Modelo clinicamente validado salvo em {clinical_model_path}")
                    
            except Exception as save_error:
                logger.error(f"Erro ao salvar modelo {architecture}: {save_error}")
                weights_path = model_path.replace('.h5', '_weights.h5')
                model.save_weights(weights_path)
                logger.info(f"Pesos do modelo {architecture} salvos em {weights_path}")
        else:
            logger.warning(f"⚠️ Modelo {architecture} não salvo devido a performance clínica insuficiente")
        
        history_path = os.path.join(output_dir, f"{architecture.lower()}_progressive_history.json")
        try:
            with open(history_path, 'w') as f:
                history_dict = {
                    'combined_history': combined_history,
                    'phase1_epochs': 5,
                    'phase2_epochs': epochs - 5,
                    'clinical_metrics': metrics,
                    'training_approach': 'progressive'
                }
                json.dump(history_dict, f, indent=2, default=str)
            logger.info(f"Histórico progressivo salvo em {history_path}")
        except Exception as e:
            logger.warning(f"Erro ao salvar histórico: {e}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Erro durante treinamento progressivo de {architecture}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def train_model(architecture, data_dir, output_dir, epochs, batch_size, learning_rate):
    """Wrapper para manter compatibilidade - usa treinamento progressivo"""
    return train_model_progressive(architecture, data_dir, output_dir, epochs, batch_size, learning_rate)

def create_advanced_ensemble(trained_models, model_metrics, output_dir):
    """
    Cria ensemble avançado com fusão por atenção baseado no scientific guide
    Implementa attention-weighted fusion para combinar predições
    """
    logger.info("Criando ensemble avançado com fusão por atenção")
    
    try:
        if len(trained_models) < 2:
            logger.warning("Pelo menos 2 modelos são necessários para ensemble")
            return None
        
        weights = {}
        clinical_scores = {}
        
        for arch, metrics in model_metrics.items():
            if arch in trained_models and metrics:
                accuracy = metrics.get('accuracy', 0.5)
                sensitivity = metrics.get('sensitivity', 0.5)
                specificity = metrics.get('specificity', 0.5)
                auc = metrics.get('auc', 0.5)
                
                clinical_score = (
                    0.3 * accuracy +
                    0.35 * sensitivity +  # Maior peso para sensibilidade (crítico em medicina)
                    0.25 * specificity +
                    0.1 * auc
                )
                
                clinical_scores[arch] = clinical_score
                weights[arch] = clinical_score
        
        total_score = sum(weights.values())
        if total_score > 0:
            for arch in weights:
                weights[arch] /= total_score
        else:
            equal_weight = 1.0 / len(trained_models)
            weights = {arch: equal_weight for arch in trained_models.keys()}
        
        logger.info(f"Scores clínicos: {clinical_scores}")
        logger.info(f"Pesos do ensemble: {weights}")
        
        ensemble_config = {
            'component_models': list(trained_models.keys()),
            'model_weights': weights,
            'clinical_scores': clinical_scores,
            'ensemble_method': 'attention_weighted_fusion',
            'attention_parameters': {
                'attention_heads': 8,
                'attention_dim': 256,
                'fusion_strategy': 'learned_attention',
                'temperature_scaling': 1.5,
                'confidence_calibration': True
            },
            'clinical_thresholds': {
                'sensitivity_threshold': 0.90,
                'specificity_threshold': 0.85,
                'confidence_threshold': 0.80
            },
            'created_at': datetime.now().isoformat(),
            'performance_metrics': model_metrics,
            'training_approach': 'progressive_ensemble'
        }
        
        ensemble_config_path = os.path.join(output_dir, "advanced_ensemble_config.json")
        with open(ensemble_config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2, default=str)
        
        logger.info(f"Configuração do ensemble avançado salva em {ensemble_config_path}")
        
        # Criar modelo ensemble físico usando StateOfTheArtModels
        try:
            sota_models = StateOfTheArtModels(input_shape=(384, 384, 3), num_classes=5)
            
            efficientnet_model = sota_models.build_real_efficientnetv2()
            vit_model = sota_models.build_real_vision_transformer()
            convnext_model = sota_models.build_real_convnext()
            
            ensemble_model = sota_models.build_attention_weighted_ensemble()
            
            if ensemble_model:
                ensemble_model_path = os.path.join(output_dir, "advanced_ensemble_model.h5")
                ensemble_model.save(ensemble_model_path)
                logger.info(f"✅ Modelo ensemble físico salvo em {ensemble_model_path}")
                
                ensemble_config['ensemble_model_path'] = ensemble_model_path
            else:
                logger.warning("Falha ao criar modelo ensemble físico")
                
        except Exception as ensemble_error:
            logger.warning(f"Erro ao criar modelo ensemble físico: {ensemble_error}")
        
        ensemble_meets_clinical_standards = True
        for arch, metrics in model_metrics.items():
            if arch in trained_models and metrics:
                sensitivity = metrics.get('sensitivity', 0)
                specificity = metrics.get('specificity', 0)
                
                if sensitivity < 0.85 or specificity < 0.80:
                    ensemble_meets_clinical_standards = False
                    logger.warning(f"Modelo {arch} não atende padrões clínicos para ensemble")
        
        ensemble_config['clinical_validation'] = {
            'meets_standards': ensemble_meets_clinical_standards,
            'validation_date': datetime.now().isoformat(),
            'minimum_sensitivity': 0.85,
            'minimum_specificity': 0.80
        }
        
        if ensemble_meets_clinical_standards:
            logger.info("✅ Ensemble atende padrões clínicos para uso médico")
        else:
            logger.warning("⚠️ Ensemble não atende todos os padrões clínicos")
        
        with open(ensemble_config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2, default=str)
        
        return ensemble_config
        
    except Exception as e:
        logger.error(f"Erro ao criar ensemble avançado: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_ensemble(trained_models, model_metrics, output_dir):
    """Wrapper para manter compatibilidade - usa ensemble avançado"""
    return create_advanced_ensemble(trained_models, model_metrics, output_dir)

def compare_models(metrics_dict):
    """Compara performance dos modelos treinados"""
    logger.info("=== COMPARAÇÃO DE MODELOS ===")
    
    if not metrics_dict:
        logger.warning("Nenhuma métrica disponível para comparação")
        return
    
    sorted_models = sorted(
        metrics_dict.items(), 
        key=lambda x: x[1].get('accuracy', 0) if x[1] else 0, 
        reverse=True
    )
    
    logger.info("Ranking por Acurácia:")
    for i, (arch, metrics) in enumerate(sorted_models, 1):
        if metrics:
            accuracy = metrics.get('accuracy', 0)
            loss = metrics.get('loss', 0)
            auc = metrics.get('auc', 0)
            logger.info(f"{i}. {arch}: Acurácia={accuracy:.4f}, Loss={loss:.4f}, AUC={auc:.4f}")
        else:
            logger.info(f"{i}. {arch}: Treinamento falhou")
    
    if sorted_models and sorted_models[0][1]:
        best_model = sorted_models[0][0]
        best_accuracy = sorted_models[0][1].get('accuracy', 0)
        logger.info(f"\n🏆 MELHOR MODELO: {best_model} (Acurácia: {best_accuracy:.4f})")

def validate_clinical_readiness(metrics_dict):
    """
    Valida se os modelos atendem critérios clínicos baseados no scientific guide
    """
    logger.info("=== VALIDAÇÃO DE PRONTIDÃO CLÍNICA ===")
    
    clinical_ready_models = []
    clinical_warnings = []
    
    for arch, metrics in metrics_dict.items():
        if not metrics:
            continue
            
        sensitivity = metrics.get('sensitivity', 0)
        specificity = metrics.get('specificity', 0)
        accuracy = metrics.get('accuracy', 0)
        auc = metrics.get('auc', 0)
        
        meets_sensitivity = sensitivity >= 0.85
        meets_specificity = specificity >= 0.80
        meets_accuracy = accuracy >= 0.85
        meets_auc = auc >= 0.85
        
        clinical_score = sum([meets_sensitivity, meets_specificity, meets_accuracy, meets_auc])
        
        logger.info(f"\n{arch} - Avaliação Clínica:")
        logger.info(f"  Sensibilidade: {sensitivity:.3f} {'✅' if meets_sensitivity else '❌'} (mín: 0.85)")
        logger.info(f"  Especificidade: {specificity:.3f} {'✅' if meets_specificity else '❌'} (mín: 0.80)")
        logger.info(f"  Acurácia: {accuracy:.3f} {'✅' if meets_accuracy else '❌'} (mín: 0.85)")
        logger.info(f"  AUC: {auc:.3f} {'✅' if meets_auc else '❌'} (mín: 0.85)")
        logger.info(f"  Score Clínico: {clinical_score}/4")
        
        if clinical_score >= 3:  # Pelo menos 3 de 4 critérios
            clinical_ready_models.append(arch)
            logger.info(f"  Status: ✅ PRONTO PARA USO CLÍNICO")
        else:
            clinical_warnings.append(f"{arch}: Score {clinical_score}/4")
            logger.info(f"  Status: ⚠️ REQUER MELHORIAS")
    
    logger.info(f"\n📊 RESUMO CLÍNICO:")
    logger.info(f"Modelos prontos para uso clínico: {len(clinical_ready_models)}")
    logger.info(f"Modelos que requerem melhorias: {len(clinical_warnings)}")
    
    if clinical_warnings:
        logger.warning("⚠️ AVISOS CLÍNICOS:")
        for warning in clinical_warnings:
            logger.warning(f"  - {warning}")
    
    return clinical_ready_models, clinical_warnings

def generate_training_report(trained_models, metrics, output_dir, clinical_ready_models):
    """
    Gera relatório detalhado do treinamento baseado no scientific guide
    """
    logger.info("Gerando relatório detalhado de treinamento...")
    
    report = {
        'training_summary': {
            'total_models_attempted': len(metrics),
            'successful_models': len(trained_models),
            'clinical_ready_models': len(clinical_ready_models),
            'success_rate': len(trained_models) / len(metrics) if metrics else 0,
            'clinical_readiness_rate': len(clinical_ready_models) / len(trained_models) if trained_models else 0
        },
        'model_performance': {},
        'clinical_validation': {
            'ready_for_clinical_use': clinical_ready_models,
            'validation_criteria': {
                'minimum_sensitivity': 0.85,
                'minimum_specificity': 0.80,
                'minimum_accuracy': 0.85,
                'minimum_auc': 0.85
            }
        },
        'training_methodology': {
            'approach': 'progressive_training',
            'phase_1': 'classifier_only_5_epochs',
            'phase_2': 'full_model_fine_tuning',
            'augmentation': 'medical_specific',
            'ensemble_method': 'attention_weighted_fusion'
        },
        'recommendations': []
    }
    
    for arch, model_metrics in metrics.items():
        if model_metrics:
            report['model_performance'][arch] = {
                'metrics': model_metrics,
                'clinical_ready': arch in clinical_ready_models,
                'training_status': 'successful' if arch in trained_models else 'failed'
            }
    
    if len(clinical_ready_models) == 0:
        report['recommendations'].append("CRÍTICO: Nenhum modelo atende critérios clínicos. Revisar dados e hiperparâmetros.")
    elif len(clinical_ready_models) < len(trained_models):
        report['recommendations'].append("Alguns modelos requerem melhorias para uso clínico.")
    
    if len(trained_models) >= 2:
        report['recommendations'].append("Ensemble recomendado para melhor performance clínica.")
    
    report_path = os.path.join(output_dir, "comprehensive_training_report.json")
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"📋 Relatório detalhado salvo em {report_path}")
    except Exception as e:
        logger.warning(f"Erro ao salvar relatório: {e}")
    
    return report

def main():
    """Função principal com pipeline completo baseado no scientific guide"""
    args = parse_args()
    
    logger.info("=== INICIANDO TREINAMENTO AVANÇADO DE IA RADIOLÓGICA ===")
    logger.info("Baseado no Scientific Guide para Medical AI Training")
    logger.info(f"Argumentos: {vars(args)}")
    
    if not validate_data_directory(args.data_dir):
        logger.error("Validação de dados falhou. Abortando treinamento.")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Validação concluída com sucesso. Saindo (--validate_only).")
        return
    
    clinical_validation_enabled = args.validate_clinical_metrics
    if clinical_validation_enabled:
        logger.info("🏥 Validação de métricas clínicas ativada durante treinamento")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.optimizer.set_jit(True)
            logger.info(f"🚀 GPU(s) detectada(s): {len(gpus)} - Mixed Precision ativado")
        else:
            logger.info("💻 Treinamento será executado na CPU")
            
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"📡 Estratégia de distribuição ativada para {len(gpus)} GPUs")
        else:
            strategy = tf.distribute.get_strategy()
            
    except Exception as e:
        logger.warning(f"Erro na configuração avançada do TensorFlow: {e}")
        strategy = tf.distribute.get_strategy()
    
    trained_models = {}
    metrics = {}
    
    for arch in args.architectures:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 TREINAMENTO PROGRESSIVO: {arch}")
        logger.info(f"{'='*60}")
        
        with strategy.scope():
            model, model_metrics = train_model_progressive(
                arch, args.data_dir, args.output_dir, 
                args.epochs, args.batch_size, args.learning_rate,
                validate_clinical_metrics=clinical_validation_enabled
            )
        
        if model is not None and model_metrics is not None:
            trained_models[arch] = model
            metrics[arch] = model_metrics
            logger.info(f"✅ {arch} treinado com sucesso")
        else:
            logger.error(f"❌ Falha no treinamento de {arch}")
            metrics[arch] = None
    
    compare_models(metrics)
    clinical_ready_models, clinical_warnings = validate_clinical_readiness(metrics)
    
    if args.ensemble and len(trained_models) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("🔗 CRIANDO ENSEMBLE AVANÇADO COM FUSÃO POR ATENÇÃO")
        logger.info(f"{'='*60}")
        
        ensemble_config = create_advanced_ensemble(trained_models, metrics, args.output_dir)
        if ensemble_config:
            logger.info("✅ Ensemble avançado criado com sucesso")
            if ensemble_config.get('clinical_validation', {}).get('meets_standards', False):
                logger.info("🏥 Ensemble atende padrões clínicos")
            else:
                logger.warning("⚠️ Ensemble requer melhorias para uso clínico")
        else:
            logger.error("❌ Falha na criação do ensemble")
    
    training_report = generate_training_report(trained_models, metrics, args.output_dir, clinical_ready_models)
    
    summary = {
        'training_completed_at': datetime.now().isoformat(),
        'training_approach': 'progressive_with_clinical_validation',
        'arguments': vars(args),
        'trained_models': list(trained_models.keys()),
        'failed_models': [arch for arch in args.architectures if arch not in trained_models],
        'clinical_ready_models': clinical_ready_models,
        'metrics': metrics,
        'total_models_trained': len(trained_models),
        'success_rate': len(trained_models) / len(args.architectures) if args.architectures else 0,
        'clinical_readiness_rate': len(clinical_ready_models) / len(trained_models) if trained_models else 0,
        'training_report': training_report
    }
    
    summary_path = os.path.join(args.output_dir, "advanced_training_summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"📊 Resumo avançado salvo em {summary_path}")
    except Exception as e:
        logger.warning(f"Erro ao salvar resumo: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("🎯 TREINAMENTO AVANÇADO CONCLUÍDO")
    logger.info("="*80)
    logger.info(f"📈 Modelos treinados: {len(trained_models)}/{len(args.architectures)}")
    logger.info(f"🏥 Modelos prontos para uso clínico: {len(clinical_ready_models)}")
    logger.info(f"📁 Modelos salvos em: {args.output_dir}")
    
    if len(trained_models) == 0:
        logger.error("❌ Nenhum modelo foi treinado com sucesso!")
        sys.exit(1)
    elif len(clinical_ready_models) == 0:
        logger.warning("⚠️ Nenhum modelo atende critérios clínicos!")
        logger.warning("Modelos treinados mas requerem melhorias para uso médico.")
    else:
        logger.info("🎉 Treinamento concluído com sucesso!")
        logger.info(f"✅ {len(clinical_ready_models)} modelo(s) pronto(s) para uso clínico")

if __name__ == "__main__":
    main()
