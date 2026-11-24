# MedAI Radiologia - Instalador Windows

## 📦 Instalador User-Friendly para Windows

Este diretório contém os arquivos necessários para criar um instalador Windows profissional e fácil de usar para o sistema MedAI Radiologia.

## 🎯 Características do Instalador

### ✅ **Para Usuários Finais (Não Técnicos)**
- **Instalação com 1 clique**: Apenas executar o arquivo `.exe`
- **Interface gráfica**: Assistente de instalação visual
- **Sem linha de comando**: Não requer CMD, PowerShell ou Docker
- **Atalhos automáticos**: Menu Iniciar e Área de Trabalho
- **Desinstalação fácil**: Através do Painel de Controle do Windows

### 🔧 **Funcionalidades Técnicas**
- **Instalador NSIS**: Padrão da indústria para Windows
- **Verificação de privilégios**: Solicita automaticamente permissões de administrador
- **Associação de arquivos**: Arquivos .dcm abrem automaticamente no MedAI
- **Registro no Windows**: Aparece em "Programas e Recursos"
- **Configuração automática**: Cria diretórios e arquivos necessários

## 🚀 Como Construir o Instalador

### Pré-requisitos
1. **NSIS (Nullsoft Scriptable Install System)**
   - Download: https://nsis.sourceforge.io/Download
   - Instalar versão mais recente

2. **Python 3.8+** com dependências do projeto
3. **PyInstaller** (instalado automaticamente)

### Construção Automática
```batch
# Execute o script de build
cd build
build_installer.bat
```

### Processo Manual (se necessário)
```batch
# 1. Instalar dependências
pip install -r ../requirements.txt

# 2. Construir executável
pyinstaller --clean MedAI_Radiologia.spec

# 3. Construir instalador
makensis medai_installer.nsi
```

## 📁 Arquivos Gerados

### Para Distribuição
- **`MedAI_Radiologia_Installer.exe`** - Instalador final para usuários

### Arquivos de Build
- **`dist/`** - Executável e dependências
- **`build/`** - Arquivos temporários de build

## 🎯 Experiência do Usuário Final

### Instalação
1. **Download**: Usuário baixa `MedAI_Radiologia_Installer.exe`
2. **Execução**: Duplo clique no arquivo
3. **Permissões**: Windows solicita permissões de administrador
4. **Assistente**: Interface gráfica guia a instalação
5. **Conclusão**: Atalhos criados automaticamente

### Uso Diário
1. **Iniciar**: Duplo clique no atalho da área de trabalho
2. **Arquivos DICOM**: Duplo clique abre automaticamente no MedAI
3. **Menu Iniciar**: Acesso através do menu "MedAI Radiologia"

### Desinstalação
1. **Painel de Controle** → Programas e Recursos
2. **Selecionar** "MedAI Radiologia"
3. **Desinstalar** → Assistente remove tudo automaticamente

## 🔒 Segurança e Compatibilidade

### Requisitos do Sistema
- **Windows 7/8/10/11** (32-bit ou 64-bit)
- **4 GB RAM** mínimo (8 GB recomendado)
- **2 GB espaço livre** em disco
- **Permissões de administrador** para instalação

### Segurança
- **Assinatura digital**: Recomendado para distribuição comercial
- **Verificação de integridade**: NSIS inclui checksums automáticos
- **Isolamento**: Instalação em diretório protegido

## 📋 Estrutura de Instalação

```
C:\Program Files\MedAI Radiologia\
├── MedAI_Radiologia.exe          # Executável principal
├── config.json                   # Configuração padrão
├── data/                          # Dados do usuário
├── models/                        # Modelos de IA
├── reports/                       # Relatórios gerados
├── temp/                          # Arquivos temporários
├── docs/                          # Documentação
│   ├── Manual_Usuario.pdf
│   └── Instalacao.pdf
└── uninstall.exe                  # Desinstalador
```

## 🎨 Personalização

### Ícones e Branding
- **Ícone**: `medai_icon.ico` (incluir arquivo de ícone)
- **Logo**: Personalizar no script NSIS
- **Cores**: Modificar tema no `medai_installer.nsi`

### Configurações Avançadas
- **Diretório padrão**: Modificar `InstallDir` no script
- **Associações de arquivo**: Adicionar formatos em `writeRegStr`
- **Atalhos**: Personalizar em `createShortCut`

## 🐛 Solução de Problemas

### Erros Comuns
1. **"NSIS não encontrado"**
   - Instalar NSIS e adicionar ao PATH

2. **"Falha ao construir executável"**
   - Verificar dependências Python
   - Executar `pip install -r requirements.txt`

3. **"Permissões negadas"**
   - Executar como administrador
   - Verificar antivírus

### Logs e Debug
- **Build logs**: Verificar saída do `build_installer.bat`
- **Instalação**: Logs em `%TEMP%\nsis_install.log`
- **Execução**: Logs do aplicativo em `data/logs/`

## 📞 Suporte

Para problemas com o instalador:
1. Verificar requisitos do sistema
2. Executar como administrador
3. Desabilitar temporariamente antivírus
4. Consultar logs de erro
5. Reportar issues no repositório GitHub

---

**🎉 Resultado Final**: Um instalador Windows profissional que permite a qualquer usuário instalar e usar o MedAI Radiologia sem conhecimento técnico!
