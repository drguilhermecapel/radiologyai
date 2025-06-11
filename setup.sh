#!/bin/bash

echo "==============================================="
echo "MEDAI RADIOLOGIA - SETUP AUTOMATIZADO LINUX"
echo "==============================================="

if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python3 não encontrado"
    echo "Instale Python 3.8+ usando seu gerenciador de pacotes"
    exit 1
fi

echo "Python3 encontrado"
echo

echo "Executando setup automatizado..."
python3 setup_medai.py

if [ $? -ne 0 ]; then
    echo
    echo "ERRO: Setup falhou"
    exit 1
fi

echo
echo "==============================================="
echo "SETUP CONCLUÍDO COM SUCESSO!"
echo "==============================================="
echo
echo "Para iniciar o sistema:"
echo "  python3 src/web_server.py"
echo
echo "Acesse: http://localhost:8080"
echo

chmod +x setup.sh
