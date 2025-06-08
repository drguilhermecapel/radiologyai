#!/usr/bin/env python3
"""
MedAI Radiologia - Construtor Final do Instalador
Cria o instalador executável final para distribuição
"""

import os
import sys
import subprocess
from pathlib import Path

def create_pyinstaller_spec():
    """Cria arquivo .spec otimizado para o instalador"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['MedAI_Radiologia_Installer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),
        ('data', 'data'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk', 
        'tkinter.messagebox',
        'threading',
        'json',
        'base64',
        'winreg',
        'win32com.client'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'opencv-python',
        'vtk',
        'SimpleITK',
        'nibabel',
        'scikit-image',
        'transformers',
        'timm'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyd = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyd,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MedAI_Radiologia_Installer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None
)
'''
    
    with open("MedAI_Installer.spec", "w") as f:
        f.write(spec_content)
    
    print("✅ Arquivo .spec criado para PyInstaller")

def build_installer():
    """Constrói o instalador final"""
    print("🏥 MEDAI RADIOLOGIA - CONSTRUTOR FINAL DO INSTALADOR")
    print("=" * 70)
    print("Criando instalador executável final para distribuição")
    print("=" * 70)
    
    installer_script = Path("MedAI_Radiologia_Installer.py")
    if not installer_script.exists():
        print(f"❌ Script do instalador não encontrado: {installer_script}")
        return False
    
    print(f"✅ Script do instalador encontrado: {installer_script}")
    print(f"📏 Tamanho: {installer_script.stat().st_size / 1024:.1f} KB")
    
    create_pyinstaller_spec()
    
    print("\n🔧 INSTRUÇÕES PARA CONSTRUÇÃO NO WINDOWS")
    print("-" * 50)
    print("Para criar o executável final, execute os seguintes comandos")
    print("em uma máquina Windows com Python instalado:")
    print()
    print("1. Instalar PyInstaller:")
    print("   pip install pyinstaller")
    print()
    print("2. Construir o executável:")
    print("   pyinstaller MedAI_Installer.spec")
    print()
    print("3. O executável será criado em:")
    print("   dist/MedAI_Radiologia_Installer.exe")
    print()
    
    build_script = '''@echo off
echo ========================================
echo MedAI Radiologia - Build do Instalador
echo ========================================
echo.

REM Verificar Python
python --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Python nao encontrado!
    echo Instale Python 3.8+ de: https://python.org
    pause
    exit /b 1
)

REM Instalar PyInstaller
echo Instalando PyInstaller...
pip install pyinstaller

REM Construir executável
echo.
echo Construindo instalador executável...
pyinstaller MedAI_Installer.spec

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCESSO! Instalador executável criado!
    echo ========================================
    echo.
    echo Arquivo: dist\\MedAI_Radiologia_Installer.exe
    echo.
    echo Este instalador:
    echo - E um executavel standalone
    echo - NAO requer Python instalado
    echo - Tem interface grafica nativa
    echo - Instala com 1 clique
    echo - Funciona em qualquer Windows
    echo.
    echo Pronto para distribuicao!
) else (
    echo.
    echo ERRO: Falha na criacao do executavel
    echo Verifique os erros acima
)

echo.
pause
'''
    
    with open("build_installer_windows.bat", "w") as f:
        f.write(build_script)
    
    print("✅ Script de build criado: build_installer_windows.bat")
    
    print("\n" + "=" * 70)
    print("📦 RESUMO DOS ARQUIVOS CRIADOS")
    print("=" * 70)
    print("✅ MedAI_Radiologia_Installer.py - Script Python do instalador")
    print("✅ MedAI_Installer.spec - Configuração PyInstaller")
    print("✅ build_installer_windows.bat - Script de build Windows")
    print("✅ test_python_installer.py - Testes do instalador")
    print()
    print("🎯 CARACTERÍSTICAS DO INSTALADOR FINAL:")
    print("   • Executável standalone (.exe)")
    print("   • Não requer Python instalado no sistema")
    print("   • Interface gráfica nativa do Windows")
    print("   • Instalação com 1 clique")
    print("   • Não depende de NSIS ou programas externos")
    print("   • Adequado para usuários sem conhecimento técnico")
    print("   • Criação automática de atalhos")
    print("   • Associação de arquivos DICOM")
    print("   • Registro no Windows")
    print()
    print("🚀 PRÓXIMOS PASSOS:")
    print("   1. Transferir arquivos para máquina Windows")
    print("   2. Executar build_installer_windows.bat")
    print("   3. Distribuir MedAI_Radiologia_Installer.exe")
    
    return True

if __name__ == "__main__":
    success = build_installer()
    sys.exit(0 if success else 1)
