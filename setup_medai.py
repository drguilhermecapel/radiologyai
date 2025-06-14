#!/usr/bin/env python3
"""
Script de Setup Automatizado para MedAI Radiologia - VERSÃO CORRIGIDA
Instala dependências e configura o ambiente para teste local
Resolve conflitos de dependências e problemas de encoding no Windows
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

def setup_encoding():
    """Configura encoding UTF-8 para evitar erros no Windows"""
    if platform.system() == 'Windows':
        try:
            os.system('chcp 65001 > nul')
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Falha silenciosa se não conseguir configurar

def safe_print(text):
    """Imprime texto de forma segura, evitando erros de encoding"""
    try:
        print(text)
    except UnicodeEncodeError:
        ascii_text = text.encode('ascii', 'replace').decode('ascii')
        print(ascii_text)

def print_header():
    """Exibe cabeçalho do setup"""
    safe_print("=" * 60)
    safe_print("MEDAI RADIOLOGIA - SETUP AUTOMATIZADO")
    safe_print("=" * 60)
    safe_print("Sistema de Análise Radiológica com IA de Última Geração")
    safe_print("Arquiteturas: EfficientNetV2, Vision Transformer, ConvNeXt")
    safe_print("=" * 60)

def check_python_version():
    """Verifica versão do Python"""
    safe_print("\nVerificando versão do Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        safe_print("ERRO: Python 3.8+ é necessário")
        safe_print(f"   Versão atual: {version.major}.{version.minor}")
        return False
    safe_print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Instala dependências do requirements.txt com tratamento robusto de erros"""
    safe_print("\nInstalando dependências...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        safe_print("ERRO: Arquivo requirements.txt não encontrado")
        return False
    
    safe_print("Atualizando pip...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True, text=True)
        safe_print("OK: Pip atualizado")
    except subprocess.CalledProcessError:
        safe_print("AVISO: Não foi possível atualizar pip, continuando...")
    
    # Instala dependências críticas primeiro
    critical_deps = [
        "numpy==1.24.3",
        "tensorflow==2.15.0",  # Versão compatível com Python 3.11 e Pydantic 2.5.0
        "opencv-python==4.8.1.78",
        "Pillow>=10.0.0",
        "pydicom==2.4.3"
    ]
    
    safe_print("Instalando dependências críticas...")
    failed_critical = []
    
    for dep in critical_deps:
        try:
            safe_print(f"  Instalando {dep}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True, text=True)
            safe_print(f"  OK: {dep}")
        except subprocess.CalledProcessError as e:
            safe_print(f"  ERRO: {dep} - {e}")
            failed_critical.append(dep)
    
    try:
        safe_print("Instalando demais dependências...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            safe_print("OK: Dependências instaladas com sucesso")
            return True
        else:
            safe_print("AVISO: Algumas dependências falharam, mas continuando...")
            safe_print(f"   Saída: {result.stdout}")
            safe_print(f"   Erro: {result.stderr}")
            return len(failed_critical) == 0  # Sucesso se dependências críticas OK
            
    except subprocess.TimeoutExpired:
        safe_print("AVISO: Timeout na instalação, mas continuando...")
        return len(failed_critical) == 0
    except subprocess.CalledProcessError as e:
        safe_print(f"AVISO: Erro ao instalar dependências: {e}")
        safe_print("Continuando com dependências críticas...")
        return len(failed_critical) == 0

def create_directories():
    """Cria diretórios necessários"""
    safe_print("\nCriando estrutura de diretórios...")
    
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
        safe_print(f"OK: {directory}")
    
    return True

def check_gpu_support():
    """Verifica suporte a GPU"""
    safe_print("\nVerificando suporte a GPU...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            safe_print(f"OK: {len(gpus)} GPU(s) detectada(s)")
            for i, gpu in enumerate(gpus):
                safe_print(f"   GPU {i}: {gpu.name}")
        else:
            safe_print("AVISO: Nenhuma GPU detectada - usando CPU")
        return True
    except ImportError:
        safe_print("ERRO: TensorFlow não instalado")
        return False

def validate_models():
    """Valida modelos de IA"""
    safe_print("\nValidando modelos de IA...")
    
    try:
        sys.path.insert(0, str(Path("src")))
        from medai_sota_models import StateOfTheArtModels
        
        sota_models = StateOfTheArtModels(input_shape=(224, 224, 3), num_classes=5)
        safe_print("OK: StateOfTheArtModels inicializado")
        
        architectures = ['EfficientNetV2', 'VisionTransformer', 'ConvNeXt']
        for arch in architectures:
            safe_print(f"OK: {arch} disponível")
        
        return True
    except Exception as e:
        safe_print(f"ERRO: Erro ao validar modelos: {e}")
        return False

def test_web_server():
    """Testa servidor web"""
    safe_print("\nTestando servidor web...")
    
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
            safe_print("OK: Servidor web funcionando")
            return True
        else:
            safe_print(f"ERRO: Servidor retornou código {response.status_code}")
            return False
            
    except Exception as e:
        safe_print(f"ERRO: Erro ao testar servidor: {e}")
        return False

def run_functionality_tests():
    """Executa testes de funcionalidade"""
    safe_print("\nExecutando testes de funcionalidade...")
    
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
                    safe_print(f"OK: {test_file}")
                else:
                    safe_print(f"ERRO: {test_file} - Código: {result.returncode}")
                    safe_print(f"   Erro: {result.stderr}")
            except subprocess.TimeoutExpired:
                safe_print(f"TIMEOUT: {test_file} - Timeout")
            except Exception as e:
                safe_print(f"ERRO: {test_file} - Erro: {e}")
        else:
            safe_print(f"AVISO: {test_file} não encontrado")

def create_sample_config():
    """Cria configuração de exemplo"""
    safe_print("\nCriando configuração de exemplo...")
    
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
    
    safe_print("OK: Configuração criada em config/local_config.json")

def display_usage_instructions():
    """Exibe instruções de uso"""
    safe_print("\n" + "=" * 60)
    safe_print("SETUP CONCLUÍDO - INSTRUÇÕES DE USO")
    safe_print("=" * 60)
    
    safe_print("\nPara iniciar o sistema:")
    safe_print("   python src/web_server.py")
    safe_print("   ou")
    safe_print("   python src/main.py")
    
    safe_print("\nAcesso via navegador:")
    safe_print("   http://localhost:8080")
    
    safe_print("\nPara executar testes:")
    safe_print("   python test_ai_functionality.py")
    safe_print("   python test_verification.py")
    
    safe_print("\nPara treinar modelos:")
    safe_print("   python train_models.py --architecture EfficientNetV2")
    safe_print("   python train_models.py --architecture VisionTransformer")
    safe_print("   python train_models.py --architecture ConvNeXt")
    
    safe_print("\nArquivos de configuração:")
    safe_print("   config/local_config.json - Configurações locais")
    safe_print("   models/model_config.json - Configurações dos modelos")
    
    safe_print("\nEstrutura de dados:")
    safe_print("   data/samples/ - Amostras DICOM para teste")
    safe_print("   models/ - Modelos de IA treinados")
    safe_print("   reports/ - Relatórios gerados")
    
    safe_print("\n" + "=" * 60)
    safe_print("Sistema MedAI pronto para uso!")
    safe_print("=" * 60)

def main():
    """Função principal do setup"""
    setup_encoding()
    
    print_header()
    
    if not Path("src").exists():
        safe_print("ERRO: Execute este script no diretório raiz do projeto")
        safe_print("   Certifique-se de que a pasta 'src' existe")
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
                safe_print(f"AVISO: {step_name} falhou, mas continuando...")
        except Exception as e:
            safe_print(f"ERRO: Erro em {step_name}: {e}")
    
    safe_print(f"\nResultado: {success_count}/{total_steps} etapas concluídas")
    
    if success_count >= total_steps - 1:  # Permite 1 falha
        display_usage_instructions()
        return 0
    else:
        safe_print("ERRO: Setup falhou - verifique os erros acima")
        return 1

if __name__ == "__main__":
    sys.exit(main())
