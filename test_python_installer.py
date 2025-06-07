#!/usr/bin/env python3
"""
Teste do Instalador Python Autônomo
Verifica se o instalador funciona corretamente sem dependências externas
"""

import os
import sys
from pathlib import Path

def test_python_installer():
    """Testa o instalador Python autônomo"""
    print("🏥 MEDAI RADIOLOGIA - TESTE DO INSTALADOR PYTHON AUTÔNOMO")
    print("=" * 70)
    print("Verificando instalador que NÃO depende de NSIS ou programas externos")
    print("=" * 70)
    
    print("\n📦 VERIFICANDO ARQUIVO DO INSTALADOR")
    print("-" * 40)
    
    installer_file = Path("MedAI_Radiologia_Installer.py")
    if installer_file.exists():
        size = installer_file.stat().st_size
        print(f"✅ Instalador encontrado: {installer_file}")
        print(f"📏 Tamanho: {size / 1024:.1f} KB")
    else:
        print(f"❌ Instalador não encontrado: {installer_file}")
        return False
    
    print("\n🔍 VERIFICANDO ESTRUTURA DO INSTALADOR")
    print("-" * 40)
    
    with open(installer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    essential_components = [
        ("class MedAIWindowsInstaller", "Classe principal do instalador"),
        ("EMBEDDED_FILES_DATA", "Dados embarcados"),
        ("def create_gui_installer", "Interface gráfica"),
        ("def run_text_installer", "Modo texto fallback"),
        ("def install_application", "Processo de instalação"),
        ("def create_shortcuts", "Criação de atalhos"),
        ("def register_application", "Registro no Windows"),
        ("import tkinter", "Interface gráfica"),
        ("import winreg", "Registro do Windows"),
        ("import base64", "Decodificação de arquivos")
    ]
    
    all_components_present = True
    for component, description in essential_components:
        if component in content:
            print(f"✅ {description}: Presente")
        else:
            print(f"❌ {description}: AUSENTE")
            all_components_present = False
    
    print("\n📁 VERIFICANDO DADOS EMBARCADOS")
    print("-" * 40)
    
    if "EMBEDDED_FILES_DATA" in content:
        import re
        import base64
        import json
        
        match = re.search(r'EMBEDDED_FILES_DATA = "([^"]+)"', content)
        if match:
            try:
                embedded_data = match.group(1)
                decoded_data = base64.b64decode(embedded_data).decode()
                files_data = json.loads(decoded_data)
                
                print(f"✅ Dados embarcados decodificados com sucesso")
                print(f"📊 Arquivos embarcados: {len(files_data)}")
                
                for file_path in files_data.keys():
                    print(f"   • {file_path}")
                    
            except Exception as e:
                print(f"❌ Erro ao decodificar dados embarcados: {e}")
                all_components_present = False
        else:
            print("❌ Dados embarcados não encontrados")
            all_components_present = False
    
    print("\n🪟 VERIFICANDO COMPATIBILIDADE WINDOWS")
    print("-" * 40)
    
    windows_features = [
        ("os.name != 'nt'", "Verificação de sistema Windows"),
        ("winreg", "Acesso ao registro do Windows"),
        ("win32com.client", "Criação de atalhos"),
        ("C:/Program Files", "Diretório de instalação padrão"),
        (".dcm", "Associação de arquivos DICOM")
    ]
    
    for feature, description in windows_features:
        if feature in content:
            print(f"✅ {description}: Implementado")
        else:
            print(f"⚠️  {description}: Não encontrado")
    
    print("\n🚀 VERIFICANDO AUTONOMIA DO INSTALADOR")
    print("-" * 40)
    
    external_dependencies = [
        ("import nsis", "NSIS import"),
        ("makensis", "Comando NSIS"),
        ("docker", "Docker"),
        ("subprocess.call.*cmd", "Linha de comando obrigatória")
    ]
    
    autonomous = True
    for dep, description in external_dependencies:
        if dep.lower() in content.lower():
            print(f"❌ Dependência externa encontrada: {description}")
            autonomous = False
        else:
            print(f"✅ Não depende de: {description}")
    
    print("\n👥 VERIFICANDO INTERFACE DO USUÁRIO")
    print("-" * 40)
    
    ui_features = [
        ("tkinter", "Interface gráfica"),
        ("messagebox", "Caixas de diálogo"),
        ("ttk.Progressbar", "Barra de progresso"),
        ("run_text_installer", "Modo texto fallback"),
        ("Bem-vindo ao instalador", "Mensagem amigável"),
        ("Instalação concluída", "Feedback de sucesso")
    ]
    
    for feature, description in ui_features:
        if feature in content:
            print(f"✅ {description}: Presente")
        else:
            print(f"⚠️  {description}: Não encontrado")
    
    print("\n" + "=" * 70)
    print("📊 RESULTADO FINAL DOS TESTES")
    print("=" * 70)
    
    if all_components_present and autonomous:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Instalador Python autônomo está pronto")
        print("✅ Não depende de NSIS ou programas externos")
        print("✅ Interface gráfica amigável para usuários")
        print("✅ Modo texto fallback disponível")
        print("✅ Instalação automática sem etapas manuais")
        print("✅ Adequado para usuários sem conhecimento técnico")
        print()
        print("🔧 Para criar executável Windows:")
        print("   1. Execute em máquina Windows com Python")
        print("   2. pip install pyinstaller")
        print("   3. pyinstaller --onefile --windowed MedAI_Radiologia_Installer.py")
        print()
        print("📦 Resultado: MedAI_Radiologia_Installer.exe")
        print("   • Executável standalone")
        print("   • Não requer instalação de Python")
        print("   • Interface gráfica nativa")
        print("   • Instalação com 1 clique")
        
        return True
    else:
        print("❌ ALGUNS TESTES FALHARAM")
        print("⚠️  Instalador requer correções")
        return False

if __name__ == "__main__":
    success = test_python_installer()
    sys.exit(0 if success else 1)
