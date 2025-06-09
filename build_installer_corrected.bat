@echo off
cls
echo ========================================
echo MedAI Radiologia - Build Installer (Corrigido)
echo ========================================
echo.

echo [INFO] Este script usa apenas Python - NSIS nao e necessario!
echo.

REM Verificar se Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado!
    echo Por favor, instale Python de: https://python.org/downloads
    pause
    exit /b 1
)

echo [1] Python encontrado - OK
echo.

REM Verificar se PyInstaller esta instalado
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [2] Instalando PyInstaller...
    pip install pyinstaller
) else (
    echo [2] PyInstaller ja instalado - OK
)

echo.
echo [3] Construindo instalador unificado...
echo.

REM Usar o instalador CLI que nao depende de NSIS
if exist "MedAI_CLI_Installer.py" (
    echo Usando MedAI_CLI_Installer.py...
    pyinstaller --onefile --name="MedAI_Radiologia_Installer" MedAI_CLI_Installer.py
) else if exist "MedAI_Radiologia_Installer.py" (
    echo Usando MedAI_Radiologia_Installer.py...
    pyinstaller --onefile --name="MedAI_Radiologia_Installer" MedAI_Radiologia_Installer.py
) else (
    echo [ERRO] Arquivo do instalador nao encontrado!
    echo Procurando por: MedAI_CLI_Installer.py ou MedAI_Radiologia_Installer.py
    pause
    exit /b 1
)

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✅ BUILD CONCLUIDO COM SUCESSO!
    echo ========================================
    echo.
    echo O instalador esta em: dist\MedAI_Radiologia_Installer.exe
    echo.
    echo ✅ SEM DEPENDENCIA DE NSIS!
    echo ✅ Instalador Python autonomo
    echo ✅ Duplo clique para instalar
    echo.
) else (
    echo.
    echo ========================================
    echo ❌ ERRO DURANTE O BUILD
    echo ========================================
    echo Verifique os logs acima
)

pause
