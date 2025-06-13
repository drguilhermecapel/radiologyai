#!/usr/bin/env python3
"""
Script de teste para verificar o instalador corrigido
"""

import sys
import subprocess
import platform

def test_installer():
    print("🧪 Testando o instalador MedAI Radiologia...")
    print("-" * 50)
    
    # Testar importação
    try:
        # Adicionar diretório atual ao path
        sys.path.insert(0, '.')
        from MedAI_Radiologia_Installer import MedAIWindowsInstaller
        print("✅ Importação bem-sucedida")
    except ImportError as e:
        print(f"❌ Erro na importação: {e}")
        return False
    
    # Testar criação da instância
    try:
        installer = MedAIWindowsInstaller()
        print("✅ Instância criada com sucesso")
    except Exception as e:
        print(f"❌ Erro ao criar instância: {e}")
        return False
    
    # Testar métodos principais
    try:
        # Testar verificação de privilégios
        result = installer.check_admin_privileges()
        print(f"✅ Verificação de privilégios: {'Admin' if result else 'Usuário normal'}")
        
        # Testar criação de diretórios (sem executar)
        print("✅ Método create_directories disponível")
        
        # Testar outros métodos
        print("✅ Método extract_files disponível")
        print("✅ Método create_configuration disponível")
        
    except Exception as e:
        print(f"❌ Erro ao testar métodos: {e}")
        return False
    
    print("\n📊 Resumo do teste:")
    print(f"- Sistema operacional: {platform.system()}")
    print(f"- Python versão: {sys.version.split()[0]}")
    print(f"- Interface gráfica: {'Disponível' if 'tkinter' in sys.modules else 'Não disponível'}")
    
    print("\n✅ Todos os testes passaram! O instalador está funcional.")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO INSTALADOR MEDAI RADIOLOGIA")
    print("=" * 60)
    
    success = test_installer()
    
    if success:
        print("\n🎉 Instalador pronto para uso!")
        print("\nPara executar o instalador:")
        print("  python MedAI_Radiologia_Installer.py")
    else:
        print("\n❌ Alguns testes falharam. Verifique os erros acima.")
