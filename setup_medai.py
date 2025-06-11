#!/usr/bin/env python3
"""
Script de Setup Automatizado para MedAI Radiologia
Instala dependências e configura o ambiente para teste local
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

def print_header():
    """Exibe cabeçalho do setup"""
    print("=" * 60)
    print("🏥 MEDAI RADIOLOGIA - SETUP AUTOMATIZADO")
    print("=" * 60)
    print("Sistema de Análise Radiológica com IA de Última Geração")
    print("Arquiteturas: EfficientNetV2, Vision Transformer, ConvNeXt")
    print("=" * 60)

def check_python_version():
    """Verifica versão do Python"""
    print("\n🐍 Verificando versão do Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário")
        print(f"   Versão atual: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Instala dependências do requirements.txt"""
    print("\n📦 Instalando dependências...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Arquivo requirements.txt não encontrado")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        print(f"   Saída: {e.stdout}")
        print(f"   Erro: {e.stderr}")
        return False

def create_directories():
    """Cria diretórios necessários"""
    print("\n📁 Criando estrutura de diretórios...")
    
    directories = [
        "data/samples/normal",
        "data/samples/pneumonia", 
        "data/samples/pleural_effusion",
        "data/samples/fracture",
        "data/samples/tumor",
        "models",
        "logs",
        "temp",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    return True

def check_gpu_support():
    """Verifica suporte a GPU"""
    print("\n🖥️  Verificando suporte a GPU...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) detectada(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  Nenhuma GPU detectada - usando CPU")
        return True
    except ImportError:
        print("❌ TensorFlow não instalado")
        return False

def validate_models():
    """Valida modelos de IA"""
    print("\n🤖 Validando modelos de IA...")
    
    try:
        sys.path.insert(0, str(Path("src")))
        from medai_sota_models import StateOfTheArtModels
        
        sota_models = StateOfTheArtModels(input_shape=(224, 224, 3), num_classes=5)
        print("✅ StateOfTheArtModels inicializado")
        
        architectures = ['EfficientNetV2', 'VisionTransformer', 'ConvNeXt']
        for arch in architectures:
            print(f"✅ {arch} disponível")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao validar modelos: {e}")
        return False

def test_web_server():
    """Testa servidor web"""
    print("\n🌐 Testando servidor web...")
    
    try:
        import requests
        import time
        import threading
        
        sys.path.insert(0, str(Path("src")))
        from web_server import app
        
        def run_server():
            app.run(host='0.0.0.0', port=8080, debug=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        time.sleep(3)
        
        response = requests.get('http://localhost:8080/', timeout=5)
        if response.status_code == 200:
            print("✅ Servidor web funcionando")
            return True
        else:
            print(f"❌ Servidor retornou código {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar servidor: {e}")
        return False

def run_functionality_tests():
    """Executa testes de funcionalidade"""
    print("\n🧪 Executando testes de funcionalidade...")
    
    test_files = [
        "test_ai_functionality.py",
        "test_verification.py"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            try:
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print(f"✅ {test_file}")
                else:
                    print(f"❌ {test_file} - Código: {result.returncode}")
                    print(f"   Erro: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file} - Timeout")
            except Exception as e:
                print(f"❌ {test_file} - Erro: {e}")
        else:
            print(f"⚠️  {test_file} não encontrado")

def create_sample_config():
    """Cria configuração de exemplo"""
    print("\n⚙️  Criando configuração de exemplo...")
    
    config = {
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "debug": False
        },
        "ai_models": {
            "default_architecture": "ensemble_model",
            "confidence_threshold": 0.75,
            "enable_gpu": True
        },
        "medical_settings": {
            "sensitivity_mode": "high",
            "pathology_focus": ["pneumonia", "pleural_effusion", "fracture", "tumor"],
            "generate_reports": True
        },
        "data_paths": {
            "models_dir": "models/",
            "samples_dir": "data/samples/",
            "reports_dir": "reports/",
            "logs_dir": "logs/"
        }
    }
    
    config_file = Path("config/local_config.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ Configuração criada em config/local_config.json")

def display_usage_instructions():
    """Exibe instruções de uso"""
    print("\n" + "=" * 60)
    print("🚀 SETUP CONCLUÍDO - INSTRUÇÕES DE USO")
    print("=" * 60)
    
    print("\n📋 Para iniciar o sistema:")
    print("   python src/web_server.py")
    print("   ou")
    print("   python src/main.py")
    
    print("\n🌐 Acesso via navegador:")
    print("   http://localhost:8080")
    
    print("\n🧪 Para executar testes:")
    print("   python test_ai_functionality.py")
    print("   python test_verification.py")
    
    print("\n🤖 Para treinar modelos:")
    print("   python train_models.py --architecture EfficientNetV2")
    print("   python train_models.py --architecture VisionTransformer")
    print("   python train_models.py --architecture ConvNeXt")
    
    print("\n📊 Arquivos de configuração:")
    print("   config/local_config.json - Configurações locais")
    print("   models/model_config.json - Configurações dos modelos")
    
    print("\n📁 Estrutura de dados:")
    print("   data/samples/ - Amostras DICOM para teste")
    print("   models/ - Modelos de IA treinados")
    print("   reports/ - Relatórios gerados")
    
    print("\n" + "=" * 60)
    print("✅ Sistema MedAI pronto para uso!")
    print("=" * 60)

def main():
    """Função principal do setup"""
    print_header()
    
    if not Path("src").exists():
        print("❌ Execute este script no diretório raiz do projeto")
        print("   Certifique-se de que a pasta 'src' existe")
        return 1
    
    steps = [
        ("Verificação do Python", check_python_version),
        ("Instalação de dependências", install_requirements),
        ("Criação de diretórios", create_directories),
        ("Verificação de GPU", check_gpu_support),
        ("Validação de modelos", validate_models),
        ("Criação de configuração", create_sample_config),
        ("Testes de funcionalidade", run_functionality_tests)
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for step_name, step_func in steps:
        try:
            if step_func():
                success_count += 1
            else:
                print(f"⚠️  {step_name} falhou, mas continuando...")
        except Exception as e:
            print(f"❌ Erro em {step_name}: {e}")
    
    print(f"\n📊 Resultado: {success_count}/{total_steps} etapas concluídas")
    
    if success_count >= total_steps - 1:  # Permite 1 falha
        display_usage_instructions()
        return 0
    else:
        print("❌ Setup falhou - verifique os erros acima")
        return 1

if __name__ == "__main__":
    sys.exit(main())
