# Instalação do MedAI Radiologia

## Sobre o Sistema

O MedAI Radiologia utiliza **inteligência artificial de última geração** para análise de imagens médicas, incluindo:
- **EfficientNetV2L**: Arquitetura mais avançada para máxima precisão
- **Vision Transformer (ViT)**: Modelo baseado em atenção para análise detalhada
- **ConvNeXt XLarge**: CNN moderna que compete com Transformers
- **Modelos Híbridos**: Combinação de múltiplas arquiteturas para confiabilidade máxima

## Requisitos do Sistema

### Windows
- Windows 10 ou superior (64-bit)
- 8 GB RAM mínimo (16 GB recomendado para modelos avançados)
- 8 GB espaço livre em disco (modelos de IA requerem mais espaço)
- Placa gráfica compatível com DirectX 11 (GPU recomendada para aceleração dos modelos SOTA)

### Para Desenvolvimento
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

## Instalação Rápida (Usuário Final)

### 🎯 Instalador Autônomo (RECOMENDADO - Sem Dependências Externas)
**✨ Novo! Instalador que NÃO requer NSIS ou outros programas**

1. **Download**: Baixe `MedAI_Radiologia_Installer.exe` da seção de releases
2. **Instalação Ultra-Simples**: 
   - ✅ **Duplo clique** no arquivo baixado
   - ✅ **Interface gráfica** guia todo o processo
   - ✅ **Sem programas externos** necessários
   - ✅ **Sem linha de comando** - tudo automático
   - ✅ **Aceite permissões** quando solicitado
   - ✅ **Aguarde conclusão** (1-2 minutos)
3. **Uso Imediato**: 
   - 🖱️ Duplo clique no atalho da área de trabalho "MedAI Radiologia"
   - 📋 Ou acesse pelo Menu Iniciar → MedAI Radiologia
   - 📁 Arquivos .dcm abrem automaticamente no programa
4. **Desinstalação**: 
   - 🗑️ Painel de Controle → Programas e Recursos → MedAI Radiologia → Desinstalar

**🎉 Características do Instalador Autônomo:**
- 🚀 **Zero dependências**: Não precisa de NSIS, Visual Studio, ou outros programas
- 🖱️ **1 clique**: Instalação completamente automática
- 👥 **Para não-técnicos**: Interface amigável para qualquer usuário
- 📦 **Tudo incluído**: Python, bibliotecas e IA embarcados
- 🔧 **Auto-configuração**: Sistema pronto para uso imediato
- 🏥 **Uso profissional**: Pronto para ambiente hospitalar

### Opção 2: Executável Standalone (Para Usuários Técnicos)
1. Baixe o arquivo `MedAI_Radiologia.exe` da seção Releases
2. Execute o arquivo baixado
3. O programa iniciará automaticamente

## Instalação para Desenvolvimento

### 1. Clone o Repositório
```bash
git clone https://github.com/drguilhermecapel/radiologyai.git
cd radiologyai
```

### 2. Instale Dependências (OBRIGATÓRIO)
```bash
pip install -r requirements.txt
```
**IMPORTANTE**: Este passo é obrigatório antes de executar o programa ou construir o executável. O arquivo requirements.txt contém todas as dependências necessárias, incluindo os modelos de IA de última geração (TensorFlow, transformers, timm).

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
**Nota**: O script build.bat automaticamente instala todas as dependências via `pip install -r requirements.txt` antes de construir o executável.

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
