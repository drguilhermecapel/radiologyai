@echo off
cls
echo ========================================
echo Build Limpo - MedAI Radiologia
echo ========================================

REM Criar ambiente virtual
echo [1] Criando ambiente virtual...
python -m venv venv

REM Ativar ambiente virtual
echo [2] Ativando ambiente virtual...
call venv\Scripts\activate.bat

REM Atualizar pip
echo [3] Atualizando pip...
python -m pip install --upgrade pip

REM Instalar dependências uma por uma
echo [4] Instalando dependencias...
pip install numpy==1.24.3
pip install tensorflow==2.13.0
pip install pydicom==2.4.3
pip install opencv-python==4.8.1.78
pip install Pillow
pip install matplotlib
pip install scikit-learn
pip install pyinstaller

REM Criar diretório models
echo [5] Criando estrutura de diretorios...
if not exist models mkdir models

REM Build
echo [6] Construindo executavel...
pyinstaller --onefile --name="MedAI_Radiologia" --add-data="models;models" src\main.py

echo.
echo Build concluido!
pause
