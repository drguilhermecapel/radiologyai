@echo off
echo Construindo MedAI Radiologia para Windows...

REM Verificar se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo Erro: Python não encontrado. Instale Python 3.8+ primeiro.
    pause
    exit /b 1
)

REM Verificar se PyInstaller está instalado
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Instalando PyInstaller...
    pip install pyinstaller
)

REM Navegar para diretório do projeto
cd /d "%~dp0\.."

REM Instalar dependências
echo Instalando dependências...
pip install -r requirements.txt

REM Construir executável
echo Construindo executável...
pyinstaller build/MedAI_Radiologia.spec

REM Verificar se build foi bem-sucedido
if exist "dist\MedAI_Radiologia.exe" (
    echo.
    echo Build concluído com sucesso!
    echo Executável criado em: dist\MedAI_Radiologia.exe
    echo.
    pause
) else (
    echo.
    echo Erro no build. Verifique os logs acima.
    echo.
    pause
    exit /b 1
)
