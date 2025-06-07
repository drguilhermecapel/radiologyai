@echo off
echo ========================================
echo MedAI Radiologia - Build Installer
echo ========================================
echo.

REM Verificar se NSIS está instalado
where makensis >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: NSIS não encontrado!
    echo Por favor, instale o NSIS de: https://nsis.sourceforge.io/Download
    pause
    exit /b 1
)

REM Criar diretório de build se não existir
if not exist "dist" mkdir dist

echo Passo 1: Instalando dependências Python...
pip install -r ..\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao instalar dependências
    pause
    exit /b 1
)

echo.
echo Passo 2: Construindo executável com PyInstaller...
pyinstaller --clean MedAI_Radiologia.spec
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao construir executável
    pause
    exit /b 1
)

echo.
echo Passo 3: Criando arquivos de configuração padrão...
echo { > medai_config_default.json
echo   "app_name": "MedAI Radiologia", >> medai_config_default.json
echo   "version": "1.0.0", >> medai_config_default.json
echo   "ai_models_path": "models/", >> medai_config_default.json
echo   "data_path": "data/", >> medai_config_default.json
echo   "reports_path": "reports/", >> medai_config_default.json
echo   "temp_path": "temp/", >> medai_config_default.json
echo   "language": "pt-BR", >> medai_config_default.json
echo   "auto_save": true, >> medai_config_default.json
echo   "max_image_size": "10MB", >> medai_config_default.json
echo   "supported_formats": [".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".nii"] >> medai_config_default.json
echo } >> medai_config_default.json

echo.
echo Passo 4: Criando licença...
echo MedAI Radiologia - Sistema de Análise Radiológica > LICENSE.txt
echo Copyright (c) 2024 MedAI Systems >> LICENSE.txt
echo. >> LICENSE.txt
echo Este software é fornecido "como está", sem garantias. >> LICENSE.txt
echo Destinado apenas para uso educacional e de pesquisa. >> LICENSE.txt
echo Para uso clínico, consulte as regulamentações locais. >> LICENSE.txt

echo.
echo Passo 5: Construindo instalador NSIS...
makensis medai_installer.nsi
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao construir instalador
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCESSO! Instalador criado com sucesso!
echo ========================================
echo.
echo Arquivo gerado: MedAI_Radiologia_Installer.exe
echo.
echo O instalador inclui:
echo - Executável principal (MedAI_Radiologia.exe)
echo - Atalhos no menu iniciar e área de trabalho
echo - Associação de arquivos DICOM
echo - Desinstalador automático
echo - Configuração padrão do sistema
echo.
echo Para distribuir: Envie apenas o arquivo MedAI_Radiologia_Installer.exe
echo O usuário final só precisa executar este arquivo como administrador.
echo.
pause
