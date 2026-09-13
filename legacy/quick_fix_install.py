#!/usr/bin/env python3
"""
Script de Instalação Rápida MedAI com Todas as Correções
Resolve problemas de dependências e encoding automaticamente
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def setup_encoding():
    """Configura encoding para evitar erros"""
    if platform.system() == 'Windows':
        os.system('chcp 65001 > nul')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

def install_compatible_deps():
    """Instala dependências compatíveis uma por uma"""
    print("=" * 60)
    print("INSTALAÇÃO RÁPIDA MEDAI - CORREÇÕES AUTOMÁTICAS")
    print("=" * 60)
    
    print("\n[1/4] Atualizando pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                   capture_output=True)
    print("[OK] Pip atualizado")
    
    essential_deps = [
        ("numpy", "1.24.3"),
        ("tensorflow", "2.15.0"),  # Versão compatível com Python 3.11
        ("opencv-python", "4.8.1.78"),
        ("Pillow", None),  # Última versão
        ("pydicom", "2.4.3"),
        ("matplotlib", None),
        ("scikit-learn", None),
        ("pandas", None),
        ("flask", None),
        ("PyQt5", None),
    ]
    
    print("\n[2/4] Instalando dependências essenciais...")
    failed = []
    
    for package, version in essential_deps:
        try:
            if version:
                pkg_spec = f"{package}=={version}"
            else:
                pkg_spec = package
            
            print(f"   Instalando {pkg_spec}...", end='', flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg_spec],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(" [OK]")
            else:
                print(" [FALHOU]")
                failed.append(package)
                
        except Exception as e:
            print(f" [ERRO: {e}]")
            failed.append(package)
    
    if failed:
        print(f"\n[AVISO] Falha ao instalar: {', '.join(failed)}")
        print("         O sistema pode funcionar parcialmente")
    
    print("\n[3/4] Criando estrutura de diretórios...")
    dirs = [
        "data/samples/normal",
        "data/samples/pneumonia",
        "data/samples/pleural_effusion", 
        "data/samples/fracture",
        "data/samples/tumor",
        "models",
        "logs",
        "temp",
        "reports",
        "config"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("[OK] Diretórios criados")
    
    print("\n[4/4] Criando arquivo de configuração...")
    config = {
        "app_name": "MedAI Radiologia",
        "version": "2.0.0",
        "server": {
            "host": "0.0.0.0",
            "port": 8080
        },
        "ai_models": {
            "default": "EfficientNetV2",
            "available": ["EfficientNetV2", "VisionTransformer", "ConvNeXt"]
        }
    }
    
    import json
    with open("config/local_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("[OK] Configuração criada")
    
    print("\n" + "=" * 60)
    print("INSTALAÇÃO CONCLUÍDA!")
    print("=" * 60)
    print("\nPara iniciar o sistema:")
    print("  python src/web_server.py")
    print("\nOu execute o setup completo:")
    print("  python setup_medai.py")
    print("\nAcesse: http://localhost:8080")
    print("=" * 60)

if __name__ == "__main__":
    setup_encoding()
    install_compatible_deps()
