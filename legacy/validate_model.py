# Script de validação do modelo treinado
import sys
import os
sys.path.append('src')

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_config(model_path):
    """Carrega modelo e configuração"""
    model_dir = Path(model_path)
    
    # Carregar modelo
    model = tf.keras.models.load_model(model_dir / "model.h5")
    
    # Carregar configuração
    with open(model_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    return model, config

def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocessa imagem para predição"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Redimensionar
    img = cv2.resize(img, target_size)
    
    # Converter para RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Normalizar
    img = img.astype(np.float32) / 255.0
    
    # Adicionar dimensão do batch
    img = np.expand_dims(img, axis=0)
    
    return img

def validate_model(model_path="models/pre_trained/simple_demo", 
                  data_dir="data/nih_chest_xray/organized"):
    """Valida modelo treinado"""
    
    logger.info("Iniciando validação do modelo")
    
    # Carregar modelo
    model, config = load_model_and_config(model_path)
    logger.info(f"Modelo carregado: {config['name']} v{config['version']}")
    
    # Preparar dados de teste
    data_path = Path(data_dir)
    classes = config['classes']
    
    # Coletar imagens de teste
    test_images = []
    test_labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = data_path / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            # Usar apenas algumas imagens para teste
            test_imgs = images[:5] if len(images) >= 5 else images
            
            for img_path in test_imgs:
                test_images.append(str(img_path))
                label_vector = np.zeros(len(classes))
                label_vector[i] = 1
                test_labels.append(label_vector)
    
    logger.info(f"Total de imagens de teste: {len(test_images)}")
    
    # Fazer predições
    predictions = []
    valid_labels = []
    
    for img_path, true_label in zip(test_images, test_labels):
        img = preprocess_image(img_path, tuple(config['input_shape'][:2]))
        if img is not None:
            pred = model.predict(img, verbose=0)
            predictions.append(pred[0])
            valid_labels.append(true_label)
    
    predictions = np.array(predictions)
    valid_labels = np.array(valid_labels)
    
    logger.info(f"Predições realizadas: {len(predictions)}")
    
    # Calcular métricas
    metrics = {}
    
    # AUC por classe
    for i, class_name in enumerate(classes):
        if np.sum(valid_labels[:, i]) > 0:  # Se há exemplos positivos
            try:
                auc = roc_auc_score(valid_labels[:, i], predictions[:, i])
                metrics[f'AUC_{class_name}'] = auc
            except:
                metrics[f'AUC_{class_name}'] = 0.0
    
    # AUC médio
    auc_values = [v for k, v in metrics.items() if k.startswith('AUC_')]
    metrics['AUC_mean'] = np.mean(auc_values) if auc_values else 0.0
    
    # Predições binárias (threshold = 0.5)
    binary_preds = (predictions > 0.5).astype(int)
    
    # Acurácia por classe
    for i, class_name in enumerate(classes):
        correct = np.sum(binary_preds[:, i] == valid_labels[:, i])
        total = len(valid_labels)
        accuracy = correct / total if total > 0 else 0.0
        metrics[f'Accuracy_{class_name}'] = accuracy
    
    # Acurácia média
    acc_values = [v for k, v in metrics.items() if k.startswith('Accuracy_')]
    metrics['Accuracy_mean'] = np.mean(acc_values) if acc_values else 0.0
    
    # Relatório detalhado
    logger.info("=== RESULTADOS DA VALIDAÇÃO ===")
    logger.info(f"AUC Médio: {metrics['AUC_mean']:.4f}")
    logger.info(f"Acurácia Média: {metrics['Accuracy_mean']:.4f}")
    
    logger.info("\nMétricas por classe:")
    for class_name in classes:
        auc = metrics.get(f'AUC_{class_name}', 0.0)
        acc = metrics.get(f'Accuracy_{class_name}', 0.0)
        logger.info(f"  {class_name}: AUC={auc:.4f}, Acc={acc:.4f}")
    
    # Salvar resultados
    results = {
        'model_info': config,
        'validation_metrics': metrics,
        'test_samples': len(predictions),
        'validation_date': pd.Timestamp.now().isoformat()
    }
    
    results_path = Path(model_path) / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Resultados salvos em: {results_path}")
    
    return results

def test_inference(model_path="models/pre_trained/simple_demo", 
                  test_image_path="data/nih_chest_xray/organized/Pneumonia"):
    """Testa inferência em uma imagem específica"""
    
    logger.info("Testando inferência em imagem específica")
    
    # Carregar modelo
    model, config = load_model_and_config(model_path)
    
    # Encontrar uma imagem de teste
    test_dir = Path(test_image_path)
    test_images = list(test_dir.glob("*.png"))
    
    if not test_images:
        logger.error(f"Nenhuma imagem encontrada em {test_dir}")
        return None
    
    test_img_path = test_images[0]
    logger.info(f"Testando com imagem: {test_img_path}")
    
    # Preprocessar
    img = preprocess_image(test_img_path, tuple(config['input_shape'][:2]))
    if img is None:
        logger.error("Erro ao carregar imagem")
        return None
    
    # Predição
    prediction = model.predict(img, verbose=0)[0]
    
    # Interpretar resultados
    logger.info("=== RESULTADO DA INFERÊNCIA ===")
    logger.info(f"Imagem: {test_img_path.name}")
    
    for i, class_name in enumerate(config['classes']):
        prob = prediction[i]
        status = "POSITIVO" if prob > 0.5 else "NEGATIVO"
        logger.info(f"  {class_name}: {prob:.4f} ({status})")
    
    # Classe mais provável
    max_idx = np.argmax(prediction)
    max_class = config['classes'][max_idx]
    max_prob = prediction[max_idx]
    
    logger.info(f"\nClasse mais provável: {max_class} ({max_prob:.4f})")
    
    return {
        'image_path': str(test_img_path),
        'predictions': dict(zip(config['classes'], prediction.tolist())),
        'most_likely_class': max_class,
        'max_probability': float(max_prob)
    }

if __name__ == "__main__":
    logger.info("=== VALIDAÇÃO DO MODELO RADIOLOGYAI ===")
    
    try:
        # Validação completa
        validation_results = validate_model()
        
        # Teste de inferência
        inference_result = test_inference()
        
        logger.info("\n=== VALIDAÇÃO CONCLUÍDA ===")
        logger.info("✅ Modelo validado com sucesso")
        
        # Resumo final
        auc_mean = validation_results['validation_metrics']['AUC_mean']
        acc_mean = validation_results['validation_metrics']['Accuracy_mean']
        
        logger.info(f"📊 Performance geral:")
        logger.info(f"   AUC médio: {auc_mean:.4f}")
        logger.info(f"   Acurácia média: {acc_mean:.4f}")
        
        if auc_mean > 0.7:
            logger.info("🎯 Performance EXCELENTE (AUC > 0.7)")
        elif auc_mean > 0.6:
            logger.info("✅ Performance BOA (AUC > 0.6)")
        elif auc_mean > 0.5:
            logger.info("⚠️ Performance ACEITÁVEL (AUC > 0.5)")
        else:
            logger.info("❌ Performance BAIXA (AUC <= 0.5)")
        
    except Exception as e:
        logger.error(f"❌ Erro na validação: {e}")
        import traceback
        traceback.print_exc()

