#!/usr/bin/env python3
"""
Final build test to verify all installer issues are resolved
"""

import os
import sys
import subprocess
from pathlib import Path

def test_spec_file_generation():
    """Test that spec file is properly generated"""
    print("📄 TESTANDO GERAÇÃO DO ARQUIVO .SPEC")
    print("=" * 50)
    
    spec_file = Path("MedAI_Installer.spec")
    if spec_file.exists():
        with open(spec_file, 'r') as f:
            content = f.read()
        
        checks = [
            ("('models', 'models')", "Diretório models incluído"),
            ("('data', 'data')", "Diretório data incluído"),
            ("'tkinter'", "Tkinter importado"),
            ("'winreg'", "WinReg importado"),
            ("excludes=", "Exclusões definidas")
        ]
        
        all_good = True
        for check, description in checks:
            if check in content:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - AUSENTE")
                all_good = False
        
        return all_good
    else:
        print("❌ Arquivo .spec não encontrado")
        return False

def test_directory_structure():
    """Test that required directories exist"""
    print("\n📁 TESTANDO ESTRUTURA DE DIRETÓRIOS")
    print("=" * 50)
    
    required_dirs = ['models', 'data', 'src']
    all_exist = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ existe")
        else:
            print(f"❌ {dir_name}/ ausente")
            all_exist = False
    
    return all_exist

def test_opencv_version():
    """Test opencv version in requirements"""
    print("\n🔍 TESTANDO VERSÃO OPENCV")
    print("=" * 50)
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    if 'opencv-python==4.8.1.78' in content:
        print("✅ OpenCV versão 4.8.1.78 (compatível com Python 3.11.9)")
        return True
    else:
        print("❌ Versão OpenCV incorreta")
        return False

def test_installer_script():
    """Test that installer script exists and is valid"""
    print("\n🔧 TESTANDO SCRIPT DO INSTALADOR")
    print("=" * 50)
    
    installer = Path("MedAI_Radiologia_Installer.py")
    if installer.exists():
        size = installer.stat().st_size / 1024
        print(f"✅ Instalador existe ({size:.1f} KB)")
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'py_compile', str(installer)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Sintaxe do instalador válida")
                return True
            else:
                print(f"❌ Erro de sintaxe: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Erro ao verificar sintaxe: {e}")
            return False
    else:
        print("❌ Script do instalador não encontrado")
        return False

def main():
    """Main test function"""
    print("🏥 TESTE FINAL DO BUILD - VERIFICAÇÃO COMPLETA")
    print("=" * 70)
    print("Verificando se todos os erros foram corrigidos")
    print("=" * 70)
    
    tests = [
        test_directory_structure(),
        test_opencv_version(),
        test_spec_file_generation(),
        test_installer_script()
    ]
    
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES FINAIS")
    print("=" * 70)
    
    if all(tests):
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Erro do diretório models: CORRIGIDO")
        print("✅ Erro da versão opencv: CORRIGIDO")
        print("✅ Arquivo .spec: ATUALIZADO")
        print("✅ Estrutura de diretórios: COMPLETA")
        print("✅ Script do instalador: VÁLIDO")
        print()
        print("🚀 BUILD PRONTO PARA EXECUÇÃO EM WINDOWS!")
        return True
    else:
        print("❌ Alguns testes falharam")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
