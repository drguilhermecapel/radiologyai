#!/usr/bin/env python3
"""
Script para corrigir problemas de encoding no Windows
Execute este script antes do setup_medai.py se tiver problemas com caracteres Unicode
"""

import os
import sys
import subprocess
import platform

def fix_windows_console():
    """Configura o console do Windows para UTF-8"""
    if platform.system() != 'Windows':
        print("Este script é específico para Windows")
        return
    
    print("Configurando console do Windows para UTF-8...")
    
    os.system('chcp 65001')
    
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        print("✅ Teste de emoji Unicode")
        print("🏥 MedAI Radiologia")
        print("Configuração bem-sucedida!")
    except UnicodeEncodeError:
        print("[OK] Teste de caracteres especiais")
        print("MedAI Radiologia")
        print("Configuração aplicada, mas emojis podem não aparecer corretamente")

def set_system_locale():
    """Configura locale do sistema para UTF-8"""
    try:
        import locale
        
        locales_to_try = [
            ('pt_BR', 'UTF-8'),
            ('en_US', 'UTF-8'),
            ('', 'UTF-8'),
            ('C', 'UTF-8')
        ]
        
        for loc, encoding in locales_to_try:
            try:
                if loc:
                    locale.setlocale(locale.LC_ALL, f'{loc}.{encoding}')
                else:
                    locale.setlocale(locale.LC_ALL, '')
                print(f"Locale configurado: {locale.getlocale()}")
                break
            except locale.Error:
                continue
    except Exception as e:
        print(f"Aviso: Não foi possível configurar locale: {e}")

def create_windows_setup_batch():
    """Cria um arquivo batch para executar o setup com encoding correto"""
    batch_content = """@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo ========================================
echo MedAI Radiologia - Setup Windows
echo ========================================
echo.

python setup_medai.py

pause
"""
    
    with open('setup_windows.bat', 'w', encoding='utf-8') as f:
        f.write(batch_content)
    
    print("\nArquivo 'setup_windows.bat' criado.")
    print("Execute este arquivo para rodar o setup com encoding UTF-8")

def main():
    """Função principal"""
    print("=" * 50)
    print("Correção de Encoding para Windows")
    print("=" * 50)
    
    if platform.system() != 'Windows':
        print("Sistema operacional detectado:", platform.system())
        print("Este script é necessário apenas no Windows")
        return
    
    fix_windows_console()
    set_system_locale()
    create_windows_setup_batch()
    
    print("\n" + "=" * 50)
    print("Correções aplicadas!")
    print("=" * 50)
    print("\nPróximos passos:")
    print("1. Execute 'setup_windows.bat' (recomendado)")
    print("   OU")
    print("2. Execute 'python setup_medai.py' diretamente")

if __name__ == "__main__":
    main()
