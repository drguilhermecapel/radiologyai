# verify_environment.py
"""
Script para verificar se o ambiente está pronto para treinamento
Execute este script primeiro para garantir que tudo está configurado
"""

import sys
import os
from pathlib import Path
import importlib

print("=" * 60)
print("Verificação de Ambiente - NIH ChestX-ray14")
print("=" * 60)

# Configuração do dataset
DATASET_PATH = os.getenv("NIH_CHEST_XRAY_DATASET_ROOT", "/home/ubuntu/radiologyai_project/radiologyai/NIH_CHEST_XRAY")

def check_python_version():
    """Verifica versão do Python"""
    print("\n1. Verificando Python:")
    version = sys.version_info
    print(f"   Versão: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   Status: OK")
        return True
    else:
        print("   Status: AVISO - Recomendado Python 3.8+")
        return False

def check_packages():
    """Verifica pacotes necessários"""
    print("\n2. Verificando pacotes:")
    
    required_packages = {
        "tensorflow": "2.15.0",
        "numpy": None,
        "pandas": None,
        "cv2": "opencv-python-headless",
        "sklearn": "scikit-learn",
        "matplotlib": None,
        "PIL": "pillow"
    }
    
    missing = []
    installed = []
    
    for package, install_name in required_packages.items():
        try:
            importlib.import_module(package)
            installed.append(package)
            print(f"   {package}: OK")
        except ImportError:
            missing.append(install_name or package)
            print(f"   {package}: NAO ENCONTRADO")
    
    return missing, installed

def check_dataset():
    """Verifica estrutura do dataset"""
    print(f"\n3. Verificando dataset em {DATASET_PATH}:")
    
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        print(f"   Pasta principal: NAO ENCONTRADA")
        return False
    else:
        print(f"   Pasta principal: OK")
    
    csv_files = list(dataset_path.glob("*.csv"))
    if csv_files:
        print(f"   Arquivo CSV: {csv_files[0].name}")
    else:
        print("   Arquivo CSV: NAO ENCONTRADO")
        return False
    
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        print("   Pasta images: NAO ENCONTRADA")
        return False
    else:
        png_count = len(list(images_dir.glob("*.png")))
        print(f"   Pasta images: OK ({png_count} arquivos PNG)")
    
    return True

def check_disk_space():
    """Verifica espaço em disco"""
    print("\n4. Verificando espaço em disco:")
    
    try:
        import shutil
        # Adapta para Windows ou Linux
        drive_letter = DATASET_PATH.split(":/")[0]
        if sys.platform == "win32":
            path_to_check = drive_letter + ":/"
        else:
            path_to_check = DATASET_PATH # Em Linux, o caminho já é absoluto

        total, used, free = shutil.disk_usage(path_to_check)
        free_gb = free / (1024**3)
        print(f"   Espaço livre em {drive_letter}: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print("   Status: AVISO - Recomendado pelo menos 5GB livres")
        else:
            print("   Status: OK")
    except Exception as e:
        print(f"   Não foi possível verificar espaço em disco: {e}")

def check_memory():
    """Verifica memória RAM"""
    print("\n5. Verificando memória RAM:")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Disponível: {available_gb:.1f} GB")
        
        if total_gb < 8:
            print("   Status: AVISO - Recomendado pelo menos 8GB RAM")
        else:
            print("   Status: OK")
    except ImportError:
        print("   psutil não instalado - não foi possível verificar")

def check_gpu():
    """Verifica disponibilidade de GPU"""
    print("\n6. Verificando GPU:")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        
        if gpus:
            print(f"   GPU encontrada: {len(gpus)} dispositivo(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("   Nenhuma GPU encontrada (treinamento será mais lento)")
    except Exception as e:
        print(f"   Não foi possível verificar GPU: {e}")

def main():
    """Executa todas as verificações"""
    
    python_ok = check_python_version()
    missing, installed = check_packages()
    dataset_ok = check_dataset()
    check_disk_space()
    check_memory()
    check_gpu()
    
    print("\n" + "=" * 60)
    print("RESUMO DA VERIFICACAO")
    print("=" * 60)
    
    all_ok = True
    
    if not python_ok:
        print("- Python: Atualizar para versão 3.8+")
        all_ok = False
    
    if missing:
        print("- Pacotes faltando: " + ", ".join(missing))
        print("  Execute: pip install " + " ".join(missing))
        all_ok = False
    
    if not dataset_ok:
        print(f"- Dataset: Verificar estrutura em {DATASET_PATH}")
        all_ok = False
    
    if all_ok:
        print("\nTUDO OK! Ambiente pronto para treinamento.")
        print("\nPróximo passo: Execute \'python quick_start_nih_training.py\' ou defina a variável de ambiente NIH_CHEST_XRAY_DATASET_ROOT")
    else:
        print("\nCORRIJA OS PROBLEMAS ACIMA antes de iniciar o treinamento.")
    
    input("\nPressione ENTER para sair...")

if __name__ == "__main__":
    main()


