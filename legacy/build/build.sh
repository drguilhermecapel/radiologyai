#!/bin/bash

echo "Construindo MedAI Radiologia..."

if ! command -v python3 &> /dev/null; then
    echo "Erro: Python3 não encontrado. Instale Python 3.8+ primeiro."
    exit 1
fi

cd "$(dirname "$0")/.."

if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Instalando dependências..."
pip install -r requirements.txt

if ! pip show pyinstaller &> /dev/null; then
    echo "Instalando PyInstaller..."
    pip install pyinstaller
fi

echo "Construindo executável..."
pyinstaller build/MedAI_Radiologia.spec

if [ -f "dist/MedAI_Radiologia" ]; then
    echo ""
    echo "Build concluído com sucesso!"
    echo "Executável criado em: dist/MedAI_Radiologia"
    echo ""
else
    echo ""
    echo "Erro no build. Verifique os logs acima."
    echo ""
    exit 1
fi
