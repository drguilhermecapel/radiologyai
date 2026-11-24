# test_trained_model.py
"""
Script para testar o modelo treinado com novas imagens
Permite fazer inferências em imagens individuais ou em lote
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 60)
print("Teste de Modelo Treinado - NIH ChestX-ray14")
print("=" * 60)

# Configurações
DATASET_ROOT = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", "/mnt/data/NIH_CHEST_XRAY")
MODEL_DIR = Path(DATASET_ROOT) / "models_trained"
CONFIG_FILE = Path(DATASET_ROOT) / "models_trained" / "config_*.json" # Pode ser necessário ajustar para o nome exato do arquivo de configuração salvo

def load_model_and_config():
    """Carrega modelo treinado e configuração"""
    print("\nCarregando modelo...")
    
    # Verificar diretório de modelos
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        print("ERRO: Diretório de modelos não encontrado!")
        return None, None
    
    # Procurar modelo
    model_files = list(model_dir.glob("*.h5"))
    if not model_files:
        print("ERRO: Nenhum modelo .h5 encontrado!")
        return None, None
    
    # Usar o modelo mais recente
    model_path = max(model_files, key=os.path.getctime)
    print(f"   Modelo encontrado: {model_path.name}")
    
    # Carregar modelo
    try:
        model = tf.keras.models.load_model(model_path)
        print("   OK: Modelo carregado com sucesso!")
    except Exception as e:
        print(f"   ERRO: Erro ao carregar modelo: {e}")
        return None, None
    
    # Carregar configuração
    config_files = list(model_dir.glob("config_*.json"))
    if config_files:
        config_file_path = max(config_files, key=os.path.getctime)
        with open(config_file_path, 'r') as f:
            config = json.load(f)
    else:
        # Configuração padrão
        config = {
            "image_size": [256, 256],
            "selected_classes": [
                "No Finding", "Infiltration", "Effusion",
                "Atelectasis", "Nodule", "Pneumothorax"
            ]
        }
    
    return model, config

def preprocess_image(image_path, size):
    """Pré-processa imagem para inferência"""
    # Carregar imagem
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERRO: Não foi possível carregar: {image_path}")
        return None
    
    # Redimensionar
    img = cv2.resize(img, tuple(size))
    
    # Normalizar
    img = img.astype(np.float32) / 255.0
    
    # Converter para RGB (3 canais)
    img = np.stack([img] * 3, axis=-1)
    
    # Adicionar dimensão do batch
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_single_image(model, config, image_path):
    """Faz predição em uma única imagem"""
    print(f"\nAnalisando: {Path(image_path).name}")
    
    # Pré-processar imagem
    img = preprocess_image(image_path, config["image_size"])
    if img is None:
        return None
    
    # Fazer predição
    predictions = model.predict(img, verbose=0)[0]
    
    # Criar dicionário de resultados
    results = {}
    for i, class_name in enumerate(config["selected_classes"]):
        results[class_name] = float(predictions[i])
    
    # Ordenar por probabilidade
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return results

def visualize_prediction(image_path, results, threshold=0.3):
    """Visualiza imagem com predições"""
    # Carregar imagem original
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Imagem
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=\'gray\')
    plt.title(f"Raio-X: {Path(image_path).name}")
    plt.axis(\'off\')
    
    # Subplot 2: Predições
    plt.subplot(1, 2, 2)
    
    # Filtrar predições acima do threshold
    filtered_results = {k: v for k, v in results.items() if v > threshold}
    if not filtered_results:
        filtered_results = {list(results.keys())[0]: list(results.values())[0]}
    
    classes = list(filtered_results.keys())
    probabilities = list(filtered_results.values())
    
    # Criar gráfico de barras
    bars = plt.barh(classes, probabilities)
    
    # Colorir barras
    for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
        if class_name == "No Finding":
            bars[i].set_color(\'green\')
        elif prob > 0.7:
            bars[i].set_color(\'red\')
        elif prob > 0.5:
            bars[i].set_color(\'orange\')
        else:
            bars[i].set_color(\'yellow\')
    
    plt.xlabel(\'Probabilidade\')
    plt.title(\'Predições do Modelo\')
    plt.xlim(0, 1)
    
    # Adicionar valores nas barras
    for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
        plt.text(prob + 0.01, i, f\'{prob:.1%}\
', va=\'center\')
    
    plt.tight_layout()
    
    # Salvar visualização
    output_path = Path("prediction_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches=\'tight\')
    print(f"\nVisualização salva em: {output_path}")
    
    plt.show()

def test_batch_images(model, config, image_folder):
    """Testa múltiplas imagens de uma pasta"""
    image_folder = Path(image_folder)
    
    # Encontrar imagens
    image_files = list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg"))
    
    if not image_files:
        print(f"ERRO: Nenhuma imagem encontrada em: {image_folder}")
        return
    
    print(f"\nTestando {len(image_files)} imagens...")
    
    all_results = []
    
    for image_path in image_files[:10]:  # Limitar a 10 para demonstração
        results = predict_single_image(model, config, image_path)
        if results:
            all_results.append({
                "image": image_path.name,
                "predictions": results
            })
    
    # Salvar resultados
    output_file = "batch_predictions.json"
    with open(output_file, \'w\') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResultados salvos em: {output_file}")
    
    # Mostrar resumo
    print("\nResumo das Predições:")
    for result in all_results[:5]:  # Mostrar apenas 5 primeiros
        print(f"\n{result[\'image\']}:")
        top_pred = list(result[\'predictions\'].items())[0]
        print(f"   Principal: {top_pred[0]} ({top_pred[1]:.1%})")

def interactive_test(model, config):
    """Modo interativo para testar imagens"""
    while True:
        print("\n" + "=" * 60)
        print("MODO DE TESTE INTERATIVO")
        print("=" * 60)
        print("\nOpções:")
        print("1. Testar imagem única")
        print("2. Testar pasta de imagens") 
        print("3. Testar imagem de exemplo do dataset")
        print("4. Sair")
        
        choice = input("\nEscolha uma opção (1-4): ")
        
        if choice == "1":
            image_path = input("\nCaminho da imagem: ").strip(\'"\')
            if Path(image_path).exists():
                results = predict_single_image(model, config, image_path)
                if results:
                    print("\nResultados:")
                    for class_name, prob in results.items():
                        print(f"   {class_name}: {prob:.1%}")
                    
                    visualize_prediction(image_path, results)
            else:
                print("ERRO: Arquivo não encontrado!")
                
        elif choice == "2":
            folder_path = input("\nCaminho da pasta: ").strip(\'"\')
            if Path(folder_path).exists():
                test_batch_images(model, config, folder_path)
            else:
                print("ERRO: Pasta não encontrada!")
                
        elif choice == "3":
            # Usar imagem de exemplo do dataset
            example_dir = Path(DATASET_ROOT) / "images"
            if example_dir.exists():
                example_images = list(example_dir.glob("*.png"))[:5]
                if example_images:
                    print("\nImagens de exemplo disponíveis:")
                    for i, img in enumerate(example_images):
                        print(f"{i+1}. {img.name}")
                    
                    idx = int(input("\nEscolha o número da imagem: ")) - 1
                    if 0 <= idx < len(example_images):
                        results = predict_single_image(model, config, example_images[idx])
                        if results:
                            print("\nResultados:")
                            for class_name, prob in results.items():
                                print(f"   {class_name}: {prob:.1%}")
                            
                            visualize_prediction(example_images[idx], results)
                else:
                    print("ERRO: Nenhuma imagem de exemplo encontrada!")
            else:
                print("ERRO: Diretório de imagens não encontrado!")
                
        elif choice == "4":
            print("\nEncerrando...")
            break
        else:
            print("ERRO: Opção inválida!")

def main():
    """Função principal"""
    # Carregar modelo
    model, config = load_model_and_config()
    
    if model is None:
        print("\nERRO: Não foi possível carregar o modelo.")
        print("   Certifique-se de que o treinamento foi concluído.")
        input("\nPressione ENTER para sair...")
        return
    
    print(f"\nInformações do Modelo:")
    print(f"   Classes: {\\', \\'.join(config[\'selected_classes\'])}")
    print(f"   Tamanho de entrada: {config[\'image_size\']}")
    
    # Iniciar modo interativo
    interactive_test(model, config)

if __name__ == "__main__":
    main()


