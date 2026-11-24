@echo off
REM =============================================================
REM Script de Build do Instalador MedAI Radiologia
REM Cria executável standalone do instalador
REM =============================================================

echo.
echo ========================================
echo   MedAI Radiologia - Build Instalador
echo ========================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado!
    echo Por favor, instale Python 3.7 ou superior
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python encontrado
echo.

REM Instalar PyInstaller se necessário
echo Verificando PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Instalando PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo [ERRO] Falha ao instalar PyInstaller
        pause
        exit /b 1
    )
)
echo [OK] PyInstaller disponivel
echo.

REM Limpar builds anteriores
echo Limpando builds anteriores...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "*.spec" del /q "*.spec"
echo [OK] Limpeza concluida
echo.

REM Criar o executável
echo Construindo instalador...
echo.

REM Opção 1: Com console (recomendado para debug)
REM pyinstaller --onefile --name "MedAI_Installer" --icon="medai.ico" MedAI_Radiologia_Installer.py

REM Opção 2: Sem console (versão final)
pyinstaller --onefile --windowed --name "MedAI_Installer" MedAI_Radiologia_Installer.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na construcao do instalador
    echo Verifique os erros acima
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Build concluido com sucesso!
echo ========================================
echo.
echo Instalador criado em: dist\MedAI_Installer.exe
echo.
echo Instrucoes de uso:
echo 1. Copie dist\MedAI_Installer.exe para distribuicao
echo 2. Usuario executa com duplo clique
echo 3. Instalacao automatica sem dependencias
echo.

REM Abrir pasta de saída
start "" "dist"

pause
