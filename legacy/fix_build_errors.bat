@echo off
setlocal enabledelayedexpansion
cls

echo ========================================
echo Corrigindo erros de build MedAI
echo ========================================
echo.

REM Navegar para o diretório do projeto
cd /d "%~dp0"

REM Criar diretório models se não existir
if not exist "models" (
    echo [1] Criando diretorio models...
    mkdir models
    echo. > models\.gitkeep
    echo [OK] Diretorio models criado
) else (
    echo [OK] Diretorio models ja existe
)

REM Verificar e corrigir requirements.txt
echo.
echo [2] Verificando requirements.txt...

REM Fazer backup do requirements.txt original
copy requirements.txt requirements.txt.backup >nul 2>&1

REM Criar novo requirements.txt corrigido
(
    echo numpy==1.24.3
    echo tensorflow==2.13.0
    echo pydicom==2.4.3
    echo opencv-python==4.8.1.78
    echo Pillow^>=9.0.0
    echo matplotlib^>=3.5.0
    echo scikit-learn^>=1.1.0
    echo pyinstaller^>=6.0.0
) > requirements_fixed.txt

REM Substituir o arquivo original
move /y requirements_fixed.txt requirements.txt >nul 2>&1
echo [OK] requirements.txt atualizado

REM Atualizar pip
echo.
echo [3] Atualizando pip...
python -m pip install --upgrade pip

REM Limpar cache do pip
echo.
echo [4] Limpando cache do pip...
python -m pip cache purge

REM Instalar dependências
echo.
echo [5] Instalando dependencias corrigidas...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERRO] Falha ao instalar dependencias
    echo Tentando instalacao individual...
    
    python -m pip install numpy==1.24.3
    python -m pip install tensorflow==2.13.0
    python -m pip install pydicom==2.4.3
    python -m pip install opencv-python==4.8.1.78
    python -m pip install Pillow matplotlib scikit-learn
)

echo.
echo [6] Preparando para build...

REM Verificar se existe main.py ou app.py
if exist "src\main.py" (
    set MAIN_FILE=src\main.py
) else if exist "main.py" (
    set MAIN_FILE=main.py
) else if exist "src\app.py" (
    set MAIN_FILE=src\app.py
) else (
    echo [ERRO] Arquivo principal nao encontrado!
    pause
    exit /b 1
)

echo Arquivo principal: !MAIN_FILE!

REM Criar arquivo .spec customizado
echo.
echo [7] Criando arquivo spec customizado...

(
echo # -*- mode: python ; coding: utf-8 -*-
echo.
echo block_cipher = None
echo.
echo a = Analysis(
echo     ['!MAIN_FILE!'],
echo     pathex=[],
echo     binaries=[],
echo     datas=[
echo         ^('models', 'models'^),
echo     ],
echo     hiddenimports=['tensorflow', 'cv2', 'pydicom', 'PIL'],
echo     hookspath=[],
echo     hooksconfig={},
echo     runtime_hooks=[],
echo     excludes=[],
echo     win_no_prefer_redirects=False,
echo     win_private_assemblies=False,
echo     cipher=block_cipher,
echo     noarchive=False,
echo ^)
echo.
echo pyz = PYZ^(a.pure, a.zipped_data, cipher=block_cipher^)
echo.
echo exe = EXE(
echo     pyz,
echo     a.scripts,
echo     a.binaries,
echo     a.zipfiles,
echo     a.datas,
echo     [],
echo     name='MedAI_Radiologia',
echo     debug=False,
echo     bootloader_ignore_signals=False,
echo     strip=False,
echo     upx=True,
echo     upx_exclude=[],
echo     runtime_tmpdir=None,
echo     console=True,
echo     disable_windowed_traceback=False,
echo     argv_emulation=False,
echo     target_arch=None,
echo     codesign_identity=None,
echo     entitlements_file=None,
echo ^)
) > medai_radiologia.spec

echo [OK] Arquivo spec criado

REM Executar PyInstaller com o novo spec
echo.
echo [8] Executando PyInstaller...
echo.

REM Executar sem privilégios de administrador se possível
pyinstaller medai_radiologia.spec --clean --noconfirm

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Build concluido com sucesso!
    echo O executavel esta em: dist\MedAI_Radiologia.exe
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Erro durante o build
    echo Verifique os logs acima
    echo ========================================
)

pause
endlocal
