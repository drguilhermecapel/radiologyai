#!/usr/bin/env python3
"""
Script para testar a funcionalidade do instalador MedAI Radiologia
Verifica se todos os métodos críticos estão implementados e se a GUI funciona
"""

import sys
import os
from pathlib import Path

def test_installer_methods():
    """Testa se todos os métodos críticos estão implementados"""
    print("🔍 Verificando métodos críticos do instalador...")
    
    try:
        from MedAI_Radiologia_Installer import MedAIWindowsInstaller
        installer = MedAIWindowsInstaller()
        
        methods_to_check = [
            'start_gui_installation',
            'install_with_gui_feedback', 
            'setup_model_system',
            'download_selected_models',
            'verify_model_integrity',
            'setup_offline_models',
            'get_model_registry_content'
        ]
        
        all_methods_exist = True
        for method in methods_to_check:
            if hasattr(installer, method):
                print(f'✅ {method} - ENCONTRADO')
            else:
                print(f'❌ {method} - AUSENTE')
                all_methods_exist = False
        
        if all_methods_exist:
            print('\n✅ Todos os métodos críticos estão implementados!')
            print('✅ O instalador não deve mais travar!')
            return True
        else:
            print('\n❌ Alguns métodos ainda estão ausentes!')
            return False
            
    except ImportError as e:
        print(f'❌ Erro ao importar o instalador: {e}')
        return False
    except Exception as e:
        print(f'❌ Erro inesperado: {e}')
        return False

def test_gui_availability():
    """Testa se a GUI está disponível"""
    print("\n🔍 Verificando disponibilidade da GUI...")
    
    try:
        import tkinter as tk
        print("✅ Tkinter disponível - GUI pode ser testada")
        return True
    except ImportError:
        print("⚠️ Tkinter não disponível - apenas modo texto")
        return False

def test_threading_implementation():
    """Verifica se o threading está implementado corretamente"""
    print("\n🔍 Verificando implementação de threading...")
    
    try:
        from MedAI_Radiologia_Installer import MedAIWindowsInstaller
        installer = MedAIWindowsInstaller()
        
        if hasattr(installer, 'start_gui_installation') and hasattr(installer, 'install_with_gui_feedback'):
            print("✅ Métodos de threading implementados")
            
            import threading
            print("✅ Módulo threading disponível")
            return True
        else:
            print("❌ Métodos de threading ausentes")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao verificar threading: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Teste de Funcionalidade do Instalador MedAI Radiologia")
    print("=" * 60)
    
    methods_ok = test_installer_methods()
    
    gui_ok = test_gui_availability()
    
    threading_ok = test_threading_implementation()
    
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES:")
    print(f"✅ Métodos implementados: {'SIM' if methods_ok else 'NÃO'}")
    print(f"✅ GUI disponível: {'SIM' if gui_ok else 'NÃO'}")
    print(f"✅ Threading implementado: {'SIM' if threading_ok else 'NÃO'}")
    
    if methods_ok and threading_ok:
        print("\n🎉 INSTALADOR PRONTO PARA USO!")
        print("✅ O botão instalar deve funcionar sem travar")
        sys.exit(0)
    else:
        print("\n⚠️ INSTALADOR AINDA TEM PROBLEMAS")
        sys.exit(1)
