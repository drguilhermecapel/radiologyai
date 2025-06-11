@echo off
REM Script de Setup para Windows - MedAI Radiologia
echo ===============================================
echo MEDAI RADIOLOGIA - SETUP AUTOMATIZADO WINDOWS
echo ===============================================

REM Verifica se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado
    echo Instale Python 3.8+ de https://python.org
    pause
    exit /b 1
)

echo Python encontrado
echo.

REM Executa setup Python
echo Executando setup automatizado...
python setup_medai.py

if errorlevel 1 (
    echo.
    echo ERRO: Setup falhou
    pause
    exit /b 1
)

echo.
echo ===============================================
echo SETUP CONCLUIDO COM SUCESSO!
echo ===============================================
echo.
echo Para iniciar o sistema:
echo   python src/web_server.py
echo.
echo Acesse: http://localhost:8080
echo.
pause
