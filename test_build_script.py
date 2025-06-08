#!/usr/bin/env python3
"""
Test script to verify build script functionality
"""

import subprocess
import os
import sys

def test_build_script():
    """Test the build script components"""
    print("🔧 TESTANDO FUNCIONALIDADE DO BUILD SCRIPT")
    print("=" * 50)
    
    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Python disponível: {result.stdout.strip()}")
        else:
            print("❌ Python não encontrado")
            return False
    except Exception as e:
        print(f"❌ Erro ao verificar Python: {e}")
        return False
    
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        if 'pyinstaller' in result.stdout.lower():
            print("✅ PyInstaller disponível")
        else:
            print("⚠️  PyInstaller não instalado (será instalado pelo script)")
    except Exception as e:
        print(f"⚠️  Erro ao verificar PyInstaller: {e}")
    
    required_files = [
        'MedAI_Radiologia_Installer.py',
        'MedAI_Installer.spec', 
        'build_final_installer.py',
        'build_installer_windows.bat'
    ]
    
    print("\n📁 VERIFICANDO ARQUIVOS NECESSÁRIOS")
    print("-" * 40)
    all_files_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024
            print(f"✅ {file_path} ({size:.1f} KB)")
        else:
            print(f"❌ {file_path} - AUSENTE")
            all_files_present = False
    
    print("\n🔍 VERIFICANDO CONTEÚDO DO BUILD SCRIPT")
    print("-" * 40)
    try:
        with open('build_installer_windows.bat', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('python build_final_installer.py', 'Geração do arquivo .spec'),
            ('pyinstaller MedAI_Installer.spec', 'Comando PyInstaller'),
            ('if not exist "MedAI_Installer.spec"', 'Verificação do arquivo .spec'),
            ('goto :end', 'Controle de fluxo')
        ]
        
        for check, description in checks:
            if check in content:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - AUSENTE")
                all_files_present = False
                
        nsis_refs = ['nsis', 'NSIS', 'makensis']
        nsis_found = any(ref in content for ref in nsis_refs)
        if not nsis_found:
            print("✅ Nenhuma referência NSIS encontrada")
        else:
            print("❌ Referências NSIS ainda presentes")
            all_files_present = False
            
    except Exception as e:
        print(f"❌ Erro ao verificar build script: {e}")
        all_files_present = False
    
    print("\n" + "=" * 50)
    if all_files_present:
        print("🎉 TODOS OS COMPONENTES DO BUILD ESTÃO PRONTOS!")
        print("✅ Build script atualizado e funcional")
        print("✅ Instalador Python autônomo disponível")
        print("✅ Nenhuma dependência NSIS")
        print("✅ Pronto para construção em Windows")
        return True
    else:
        print("❌ Alguns componentes estão ausentes ou incorretos")
        return False

if __name__ == "__main__":
    success = test_build_script()
    sys.exit(0 if success else 1)
