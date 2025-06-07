# Instalação do MedAI Radiologia

## Requisitos do Sistema

### Windows
- Windows 10 ou superior (64-bit)
- 8 GB RAM mínimo (16 GB recomendado)
- 5 GB espaço livre em disco
- Placa gráfica compatível com DirectX 11 (GPU opcional para aceleração)

### Para Desenvolvimento
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

## Instalação Rápida (Usuário Final)

1. Baixe o arquivo `MedAI_Radiologia.exe` da seção Releases
2. Execute o arquivo baixado
3. O programa iniciará automaticamente

## Instalação para Desenvolvimento

### 1. Clone o Repositório
```bash
git clone https://github.com/drguilhermecapel/radiologyai.git
cd radiologyai
```

### 2. Instale Dependências
```bash
pip install -r requirements.txt
```

### 3. Execute o Programa
```bash
python src/main.py
```

## Construindo o Executável

### Windows
```bash
cd build
build.bat
```

### Linux/macOS
```bash
cd build
chmod +x build.sh
./build.sh
```

## Solução de Problemas

### Erro: "Módulo não encontrado"
- Verifique se todas as dependências foram instaladas: `pip install -r requirements.txt`

### Erro: "GPU não detectada"
- O programa funcionará em modo CPU
- Para usar GPU, instale drivers NVIDIA atualizados

### Erro: "Arquivo DICOM não suportado"
- Verifique se o arquivo não está corrompido
- Tente converter para formato padrão (PNG, JPEG)

## Suporte

Para suporte técnico, abra uma issue no repositório GitHub ou entre em contato com drguilhermecapel@gmail.com
