# config_training.py
"""
Arquivo de configuração para treinamento com NIH ChestX-ray14
Ajuste os caminhos e parâmetros conforme necessário
"""

import os
from pathlib import Path

# Configuração customizada para seu dataset
# Tenta ler o caminho do dataset de uma variável de ambiente, caso contrário, usa um padrão
DATASET_ROOT = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", r"/home/ubuntu/radiologyai_project/radiologyai/NIH_CHEST_XRAY")

CONFIG = {
    # Caminhos do dataset - ajuste para sua localização
    'data_dir': DATASET_ROOT,
    'image_dir': os.path.join(DATASET_ROOT, 'images'),
    'csv_file': os.path.join(DATASET_ROOT, 'Data_Entry_2017_v2020.csv'),
    
    # Diretório de saída para modelos treinados
    'output_dir': os.path.join(DATASET_ROOT, 'models_trained'),
    
    # Parâmetros de treinamento
    'batch_size': 16,  # Reduzir para 8 ou 4 se tiver pouca memória RAM
    'image_size': (320, 320),  # Pode reduzir para (224, 224) se necessário
    'epochs': 50,  # Número de épocas de treinamento
    'learning_rate': 1e-4,  # Taxa de aprendizado inicial
    'validation_split': 0.15,  # 15% dos dados para validação
    'test_split': 0.15,  # 15% dos dados para teste
    
    # Classes de patologias para treinar
    # Você pode adicionar ou remover classes conforme necessário
    'selected_classes': [
        'No Finding',      # Sem achados patológicos
        'Pneumonia',       # Pneumonia
        'Effusion',        # Derrame pleural
        'Atelectasis',     # Atelectasia
        'Infiltration',    # Infiltração
        'Mass',            # Massa
        'Nodule',          # Nódulo
        'Consolidation',   # Consolidação
        'Pneumothorax'     # Pneumotórax
    ],
    
    # Configurações avançadas
    'random_seed': 42,  # Para reprodutibilidade
    'num_workers': 4,   # Threads para carregamento de dados
    'prefetch_buffer': 2,  # Buffer de pré-carregamento
    
    # Thresholds clínicos (ajuste conforme necessário)
    'clinical_thresholds': {
        'default': 0.5,
        'high_sensitivity': 0.3,  # Para screening
        'high_specificity': 0.7   # Para confirmação
    }
}

# Verificar se os caminhos existem
def verify_paths():
    """Verifica se os caminhos configurados existem"""
    paths_to_check = {
        'data_dir': CONFIG['data_dir'],
        'image_dir': CONFIG['image_dir'],
        'csv_file': CONFIG['csv_file']
    }
    
    all_exist = True
    for name, path in paths_to_check.items():
        if not Path(path).exists():
            print(f"AVISO: {name} não encontrado: {path}")
            all_exist = False
    
    return all_exist

# Criar diretório de saída se não existir
def create_output_dir():
    """Cria diretório de saída se não existir"""
    output_path = Path(CONFIG['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Diretório de saída: {output_path}")

if __name__ == "__main__":
    print("Configuração de Treinamento NIH ChestX-ray14")
    print("=" * 50)
    print("\nCaminhos configurados:")
    for key, value in CONFIG.items():
        if 'dir' in key or 'file' in key:
            print(f"  {key}: {value}")
    
    print("\nClasses selecionadas:")
    for i, cls in enumerate(CONFIG['selected_classes'], 1):
        print(f"  {i}. {cls}")
    
    print(f"\nParâmetros de treinamento:")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Image size: {CONFIG['image_size']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    
    if verify_paths():
        print("\nTodos os caminhos verificados com sucesso!")
        create_output_dir()
    else:
        print("\nATENÇÃO: Verifique os caminhos acima antes de iniciar o treinamento!")


