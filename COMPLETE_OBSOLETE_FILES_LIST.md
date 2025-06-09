# Lista Completa de Arquivos Obsoletos para Remoção

## 🔍 Análise Baseada nos Logs CMD e Estrutura do Repositório

### ❌ ARQUIVOS OBSOLETOS CONFIRMADOS (REMOVER)

#### 1. Instaladores NSIS Dependentes
- `build_final_installer.py` - Gera instalador que requer NSIS externo
- `build_installer_windows.bat` - Script que falha com "ERRO: NSIS não encontrado!"

#### 2. Scripts de Build Conflitantes  
- `build/build.bat` - Usa spec file obsoleto (`build/MedAI_Radiologia.spec`)
- `build/build_windows.py` - PyInstaller com configuração conflitante
- `build/setup.py` - Setup.py desnecessário para instalador standalone

#### 3. Testes de Sistemas Obsoletos
- `test_final_build.py` - Testa estrutura obsoleta com diretório models
- `test_build_script.py` - Testa scripts NSIS que não funcionam

### ✅ ARQUIVOS NECESSÁRIOS (MANTER)

#### Instalador Unificado Atual
- `MedAI_CLI_Installer.py` - Instalador unificado CLI (PRINCIPAL)
- `MedAI_Installer.spec` - Configuração PyInstaller atualizada
- `MedAI_Radiologia_Installer.py` - Instalador GUI alternativo

#### Testes Válidos
- `test_python_installer.py` - Testa instalador Python autônomo
- `test_unified_installer.py` - Testa instalador unificado

## 🚨 PROBLEMAS ESPECÍFICOS DOS LOGS CMD

### Erro NSIS (build_installer_windows.bat)
```
ERRO: NSIS n├úo encontrado!
Por favor, instale o NSIS de: https://nsis.sourceforge.io/Download
```

### Erro OpenCV (build/build.bat)
```
ERROR: Could not find a version that satisfies the requirement opencv-python==4.8.1
```

### Erro Models Directory (build/build.bat)
```
ERROR: Unable to find 'C:\...\models' when adding binary and data files.
```

## 📋 AÇÕES DE REMOÇÃO NECESSÁRIAS

1. **Remover** `build_final_installer.py` (depende de NSIS)
2. **Remover** `build_installer_windows.bat` (falha sem NSIS)  
3. **Remover** `build/build.bat` (configuração obsoleta)
4. **Remover** `build/build_windows.py` (conflita com instalador atual)
5. **Remover** `build/setup.py` (desnecessário)
6. **Remover** `test_final_build.py` (testa estrutura obsoleta)
7. **Remover** `test_build_script.py` (testa NSIS obsoleto)

## ✅ RESULTADO ESPERADO

Após remoção:
- ✅ Apenas instalador unificado funcional permanece
- ✅ Sem dependências NSIS
- ✅ Sem conflitos de configuração
- ✅ Build limpo e funcional
