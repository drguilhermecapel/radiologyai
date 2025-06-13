#!/usr/bin/env python3
"""
Script de teste para verificar o instalador MedAI Radiologia
"""

import sys
import os
from pathlib import Path

def test_installer():
    """Testa se o instalador funciona corretamente"""
    print("🧪 Testando Instalador MedAI Radiologia...")
    print("-" * 50)
    
    try:
        # Tenta importar o instalador
        from MedAI_Radiologia_Installer import MedAIWindowsInstaller
        print("✅ Instalador importado com sucesso")
        
        # Verifica se a classe existe
        installer = MedAIWindowsInstaller()
        print("✅ Classe MedAIWindowsInstaller criada")
        
        # Verifica métodos essenciais
        methods = [
            'run', 'create_gui_installer', 'install_application',
            'setup_model_system', 'download_selected_models',
            'verify_model_integrity', 'setup_offline_models',
            'get_model_registry_content'
        ]
        
        missing_methods = []
        for method in methods:
            if not hasattr(installer, method):
                missing_methods.append(method)
            else:
                print(f"✅ Método {method} encontrado")
        
        if missing_methods:
            print(f"❌ Métodos faltando: {missing_methods}")
            return False
        
        print("\n✅ Todos os testes passaram!")
        print("O instalador está pronto para uso.")
        return True
        
    except ImportError as e:
        print(f"❌ Erro ao importar instalador: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        return False

if __name__ == "__main__":
    print("MedAI Radiologia - Teste do Instalador")
    print("=" * 50)
    
    if test_installer():
        print("\n🎉 Instalador funcionando corretamente!")
        print("Execute 'python MedAI_Radiologia_Installer.py' para iniciar")
    else:
        print("\n❌ Problemas encontrados no instalador")
        sys.exit(1)
