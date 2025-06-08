@echo off
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

REM Gerar arquivo spec primeiro
echo.
echo Gerando arquivo de configuração...
python build_final_installer.py

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
    echo Arquivo: dist\MedAI_Radiologia_Installer.exe
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
