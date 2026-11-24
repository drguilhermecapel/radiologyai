#!/bin/bash
# Script para baixar dataset NIH ChestX-ray14

echo "=== Download do Dataset NIH ChestX-ray14 ==="
echo "Este script baixa o dataset completo (~42GB) ou um subconjunto para teste"

# Criar diretórios
mkdir -p data/nih_chest_xray
cd data/nih_chest_xray

# URLs dos arquivos
BASE_URL="https://nihcc.app.box.com/shared/static"
LABELS_URL="https://nihcc.box.com/shared/static/7jjl4p7h7m9xqjb04z7s5m3x8p7qv3yx.csv"

# Baixar arquivo de labels (obrigatório)
echo "Baixando arquivo de labels..."
wget -O Data_Entry_2017_v2020.csv "$LABELS_URL"

if [ $? -eq 0 ]; then
    echo "✅ Arquivo de labels baixado com sucesso"
else
    echo "❌ Erro ao baixar arquivo de labels"
    exit 1
fi

# Baixar listas de treino/validação/teste
echo "Baixando listas de divisão dos dados..."
wget -O train_val_list.txt "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.txt"
wget -O test_list.txt "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.txt"

# Perguntar ao usuário sobre download completo ou parcial
echo ""
echo "Opções de download:"
echo "1) Download completo (42GB - todos os 12 arquivos)"
echo "2) Download parcial (3.6GB - apenas images_001.tar.gz para teste)"
echo "3) Pular download de imagens (usar apenas para configuração)"
echo ""
read -p "Escolha uma opção (1-3): " choice

case $choice in
    1)
        echo "Iniciando download completo..."
        # URLs dos arquivos de imagem (exemplo - URLs reais precisam ser obtidas do Box)
        for i in {001..012}; do
            echo "Baixando images_$i.tar.gz..."
            # Nota: URLs específicas do Box precisam ser obtidas manualmente
            echo "⚠️ Para download completo, acesse: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737"
            echo "   e baixe manualmente os arquivos images_001.tar.gz até images_012.tar.gz"
            break
        done
        ;;
    2)
        echo "Iniciando download parcial (apenas images_001.tar.gz)..."
        echo "⚠️ Para download, acesse: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737"
        echo "   e baixe manualmente o arquivo images_001.tar.gz"
        ;;
    3)
        echo "Pulando download de imagens..."
        ;;
    *)
        echo "Opção inválida"
        exit 1
        ;;
esac

echo ""
echo "=== Instruções para extração ==="
echo "Após baixar os arquivos .tar.gz, execute:"
echo "  cd data/nih_chest_xray"
echo "  for file in images_*.tar.gz; do tar -xzf \"\$file\"; done"
echo ""
echo "Para organizar os dados por patologia, execute:"
echo "  python organize_data.py"
echo ""
echo "=== Estrutura final esperada ==="
echo "data/nih_chest_xray/"
echo "├── Data_Entry_2017_v2020.csv"
echo "├── train_val_list.txt"
echo "├── test_list.txt"
echo "├── images/"
echo "│   ├── 00000001_000.png"
echo "│   ├── 00000001_001.png"
echo "│   └── ..."
echo "└── organized/"
echo "    ├── Pneumonia/"
echo "    ├── Effusion/"
echo "    ├── No Finding/"
echo "    └── ..."

