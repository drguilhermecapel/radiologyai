#!/usr/bin/env python3
"""
Script de Treinamento para Modelos de IA Radiológica
Treina modelos EfficientNetV2, Vision Transformer e ConvNeXt
com dados de imagens médicas reais
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

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from medai_training_system import RadiologyDataset, MedicalModelTrainer
    from medai_sota_models import StateOfTheArtModels
    from medai_ml_pipeline import MLPipeline, DatasetConfig, ModelConfig, TrainingConfig
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
    parser.add_argument('--resume', type=str, default=None,
                       help='Retomar treinamento de um checkpoint')
    return parser.parse_args()

def validate_data_directory(data_dir):
    """Valida estrutura do diretório de dados"""
    logger.info(f"Validando diretório de dados: {data_dir}")
    
    if not os.path.exists(data_dir):
        logger.error(f"Diretório de dados não encontrado: {data_dir}")
        return False
    
    expected_classes = ['normal', 'pneumonia', 'pleural_effusion', 'fracture', 'tumor']
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
    
    logger.info(f"Validação concluída. Classes encontradas: {found_classes}")
    return True

def get_model_config(architecture, input_shape, num_classes, learning_rate):
    """Obter configuração específica do modelo"""
    configs = {
        'EfficientNetV2': {
            'input_shape': (384, 384, 3),
            'batch_size': 16,
            'learning_rate': learning_rate,
            'epochs_default': 50
        },
        'VisionTransformer': {
            'input_shape': (224, 224, 3),
            'batch_size': 12,
            'learning_rate': learning_rate * 0.5,  # ViT geralmente precisa de LR menor
            'epochs_default': 60
        },
        'ConvNeXt': {
            'input_shape': (256, 256, 3),
            'batch_size': 14,
            'learning_rate': learning_rate * 2,  # ConvNeXt pode usar LR maior
            'epochs_default': 45
        }
    }
    
    return configs.get(architecture, {
        'input_shape': input_shape,
        'batch_size': 16,
        'learning_rate': learning_rate,
        'epochs_default': 50
    })

def train_model(architecture, data_dir, output_dir, epochs, batch_size, learning_rate):
    """Treina um modelo específico"""
    logger.info(f"Iniciando treinamento de {architecture}")
    
    try:
        model_config = get_model_config(architecture, (224, 224, 3), 5, learning_rate)
        
        dataset_config = DatasetConfig(
            image_size=model_config['input_shape'][:2],
            batch_size=model_config['batch_size'],
            validation_split=0.2,
            test_split=0.1,
            class_names=['normal', 'pneumonia', 'pleural_effusion', 'fracture', 'tumor']
        )
        
        model_config_obj = ModelConfig(
            architecture=architecture,
            input_shape=model_config['input_shape'],
            num_classes=len(dataset_config.class_names),
            learning_rate=model_config['learning_rate']
        )
        
        training_config = TrainingConfig(
            epochs=epochs,
            early_stopping=True,
            patience=10,
            reduce_lr=True,
            save_best_only=True
        )
        
        pipeline = MLPipeline(model_config_obj, dataset_config, training_config)
        
        logger.info("Preparando dados...")
        train_ds, val_ds, test_ds = pipeline.prepare_data(data_dir, dataset_config)
        
        if train_ds is None:
            logger.error(f"Falha ao preparar dados para {architecture}")
            return None, None
        
        logger.info("Construindo modelo...")
        model = pipeline.build_model()
        
        if model is None:
            logger.error(f"Falha ao construir modelo {architecture}")
            return None, None
        
        logger.info("Iniciando treinamento...")
        history = pipeline.train(train_ds, val_ds)
        
        logger.info("Avaliando modelo...")
        metrics = pipeline.evaluate(test_ds)
        logger.info(f"Métricas de avaliação para {architecture}: {metrics}")
        
        model_filename = f"chest_xray_{architecture.lower()}_model.h5"
        model_path = os.path.join(output_dir, model_filename)
        
        try:
            model.save(model_path)
            logger.info(f"Modelo {architecture} salvo em {model_path}")
        except Exception as save_error:
            logger.error(f"Erro ao salvar modelo {architecture}: {save_error}")
            weights_path = model_path.replace('.h5', '_weights.h5')
            model.save_weights(weights_path)
            logger.info(f"Pesos do modelo {architecture} salvos em {weights_path}")
        
        history_path = os.path.join(output_dir, f"{architecture.lower()}_history.json")
        try:
            with open(history_path, 'w') as f:
                history_dict = {}
                if hasattr(history, 'history'):
                    for key, values in history.history.items():
                        history_dict[key] = [float(v) for v in values]
                json.dump(history_dict, f, indent=2)
            logger.info(f"Histórico de treinamento salvo em {history_path}")
        except Exception as e:
            logger.warning(f"Erro ao salvar histórico: {e}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Erro durante treinamento de {architecture}: {e}")
        return None, None

def create_ensemble(trained_models, model_metrics, output_dir):
    """Cria modelo ensemble a partir dos modelos treinados"""
    logger.info("Criando modelo ensemble")
    
    try:
        if len(trained_models) < 2:
            logger.warning("Pelo menos 2 modelos são necessários para ensemble")
            return None
        
        weights = {}
        total_accuracy = 0
        
        for arch, metrics in model_metrics.items():
            if arch in trained_models and metrics:
                accuracy = metrics.get('accuracy', 0.5)
                weights[arch] = accuracy
                total_accuracy += accuracy
        
        if total_accuracy > 0:
            for arch in weights:
                weights[arch] /= total_accuracy
        else:
            equal_weight = 1.0 / len(trained_models)
            weights = {arch: equal_weight for arch in trained_models.keys()}
        
        logger.info(f"Pesos do ensemble: {weights}")
        
        ensemble_config = {
            'component_models': list(trained_models.keys()),
            'model_weights': weights,
            'ensemble_method': 'weighted_average',
            'created_at': datetime.now().isoformat(),
            'performance_metrics': model_metrics
        }
        
        ensemble_config_path = os.path.join(output_dir, "ensemble_config.json")
        with open(ensemble_config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        logger.info(f"Configuração do ensemble salva em {ensemble_config_path}")
        
        
        return ensemble_config
        
    except Exception as e:
        logger.error(f"Erro ao criar ensemble: {e}")
        return None

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

def main():
    """Função principal"""
    args = parse_args()
    
    logger.info("=== INICIANDO TREINAMENTO DE MODELOS DE IA RADIOLÓGICA ===")
    logger.info(f"Argumentos: {vars(args)}")
    
    if not validate_data_directory(args.data_dir):
        logger.error("Validação de dados falhou. Abortando treinamento.")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Validação concluída com sucesso. Saindo (--validate_only).")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU(s) detectada(s): {len(gpus)}")
        else:
            logger.info("Treinamento será executado na CPU")
    except Exception as e:
        logger.warning(f"Erro na configuração do TensorFlow: {e}")
    
    trained_models = {}
    metrics = {}
    
    for arch in args.architectures:
        logger.info(f"\n{'='*50}")
        logger.info(f"TREINANDO: {arch}")
        logger.info(f"{'='*50}")
        
        model, model_metrics = train_model(
            arch, args.data_dir, args.output_dir, 
            args.epochs, args.batch_size, args.learning_rate
        )
        
        if model is not None:
            trained_models[arch] = model
            metrics[arch] = model_metrics
            logger.info(f"✅ {arch} treinado com sucesso")
        else:
            logger.error(f"❌ Falha no treinamento de {arch}")
            metrics[arch] = None
    
    compare_models(metrics)
    
    if args.ensemble and len(trained_models) > 1:
        logger.info(f"\n{'='*50}")
        logger.info("CRIANDO ENSEMBLE")
        logger.info(f"{'='*50}")
        
        ensemble_config = create_ensemble(trained_models, metrics, args.output_dir)
        if ensemble_config:
            logger.info("✅ Ensemble criado com sucesso")
        else:
            logger.error("❌ Falha na criação do ensemble")
    
    summary = {
        'training_completed_at': datetime.now().isoformat(),
        'arguments': vars(args),
        'trained_models': list(trained_models.keys()),
        'failed_models': [arch for arch in args.architectures if arch not in trained_models],
        'metrics': metrics,
        'total_models_trained': len(trained_models),
        'success_rate': len(trained_models) / len(args.architectures) if args.architectures else 0
    }
    
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Resumo do treinamento salvo em {summary_path}")
    except Exception as e:
        logger.warning(f"Erro ao salvar resumo: {e}")
    
    logger.info("\n=== TREINAMENTO CONCLUÍDO ===")
    logger.info(f"Modelos treinados: {len(trained_models)}/{len(args.architectures)}")
    logger.info(f"Modelos salvos em: {args.output_dir}")
    
    if len(trained_models) == 0:
        logger.error("Nenhum modelo foi treinado com sucesso!")
        sys.exit(1)
    else:
        logger.info("Treinamento concluído com sucesso! 🎉")

if __name__ == "__main__":
    main()
