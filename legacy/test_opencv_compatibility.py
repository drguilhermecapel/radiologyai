#!/usr/bin/env python3
"""
Test OpenCV compatibility with Python 3.11.9
"""

import sys
import subprocess

def test_opencv_compatibility():
    """Test if opencv-python==4.8.1.78 is compatible with current Python version"""
    print("🔍 TESTANDO COMPATIBILIDADE OPENCV")
    print("=" * 50)
    
    python_version = sys.version
    print(f"✅ Versão Python: {python_version}")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--dry-run', 'opencv-python==4.8.1.78'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ opencv-python==4.8.1.78 é compatível com Python 3.11.9")
            return True
        else:
            print("❌ opencv-python==4.8.1.78 não é compatível")
            print(f"Erro: {result.stderr}")
            
            print("\n🔍 Procurando versão compatível...")
            result2 = subprocess.run([
                sys.executable, '-m', 'pip', 'index', 'versions', 'opencv-python'
            ], capture_output=True, text=True)
            
            if result2.returncode == 0:
                print("Versões disponíveis:")
                print(result2.stdout)
            
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar compatibilidade: {e}")
        return False

def test_requirements_compatibility():
    """Test if all requirements are compatible"""
    print("\n📋 TESTANDO COMPATIBILIDADE DE REQUIREMENTS")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--dry-run', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Todos os requirements são compatíveis")
            return True
        else:
            print("❌ Alguns requirements têm problemas de compatibilidade")
            print(f"Erro: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar requirements: {e}")
        return False

if __name__ == "__main__":
    opencv_ok = test_opencv_compatibility()
    requirements_ok = test_requirements_compatibility()
    
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    if opencv_ok and requirements_ok:
        print("🎉 TODOS OS TESTES DE COMPATIBILIDADE PASSARAM!")
        sys.exit(0)
    else:
        print("❌ Alguns testes falharam - verificar compatibilidade")
        sys.exit(1)
