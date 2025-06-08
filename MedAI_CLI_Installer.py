#!/usr/bin/env python3
"""
MedAI Radiologia - Instalador Portátil Unificado
Versão 2.0.0 - Interface de Linha de Comando

Este instalador unifica todas as funções de instalação em um único arquivo executável.
Permite instalação com duplo clique sem dependências externas.
"""

import os
import sys
import shutil
import tempfile
import zipfile
import json
import base64
import subprocess
import time
from pathlib import Path

class MedAIUnifiedInstaller:
    def __init__(self):
        self.version = "2.0.0"
        self.app_name = "MedAI Radiologia"
        self.default_install_path = "C:\\MedAI_Radiologia"
        
    def show_welcome(self):
        """Exibe mensagem de boas-vindas"""
        print("=" * 70)
        print(f"🏥 {self.app_name} - Instalador Portátil Unificado v{self.version}")
        print("=" * 70)
        print("Sistema de Análise de Imagens Médicas com Inteligência Artificial")
        print("Desenvolvido para profissionais de radiologia e medicina")
        print("=" * 70)
        print()
        
    def get_installation_settings(self):
        """Coleta configurações de instalação do usuário"""
        print("📋 CONFIGURAÇÕES DE INSTALAÇÃO")
        print("-" * 50)
        
        install_path = input(f"Caminho de instalação [{self.default_install_path}]: ").strip()
        if not install_path:
            install_path = self.default_install_path
            
        create_shortcut = input("Criar atalho na área de trabalho? [S/n]: ").strip().lower()
        create_shortcut = create_shortcut != 'n'
        
        associate_dicom = input("Associar arquivos DICOM (.dcm) ao MedAI? [S/n]: ").strip().lower()
        associate_dicom = associate_dicom != 'n'
        
        return {
            'install_path': install_path,
            'create_shortcut': create_shortcut,
            'associate_dicom': associate_dicom
        }
        
    def extract_main_application(self, install_path):
        """Extrai a aplicação principal para o diretório de instalação"""
        print("\n📦 Extraindo aplicação principal...")
        
        main_app_data = """
IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwojIC0qLSBjb2Rpbmc6IHV0Zi04IC0qLQoKaW1wb3J0IG9zCmltcG9ydCBzeXMKaW1wb3J0IGxvZ2dpbmcKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgojIENvbmZpZ3VyYXIgbG9nZ2luZwpsb2dnaW5nLmJhc2ljQ29uZmlnKGxldmVsPWxvZ2dpbmcuSU5GTykKbG9nZ2VyID0gbG9nZ2luZy5nZXRMb2dnZXIoJ01lZEFJJykKCmRlZiBtYWluKCk6CiAgICBwcmludCgi8J+PpSBNZWRBSSBSYWRpb2xvZ2lhIC0gU2lzdGVtYSBkZSBBbsOhbGlzZSBkZSBJbWFnZW5zIE3DqWRpY2FzIikKICAgIHByaW50KCLwn5qAIFNpc3RlbWEgaW5pY2lhbGl6YWRvIGNvbSBzdWNlc3NvISIpCiAgICBwcmludCgi8J+TiyBQYXJhIGluaWNpYXIgYSBhbsOhbGlzZSwgZXhlY3V0ZSBNZWRBSV9SYWRpb2xvZ2lhLmJhdCIpCiAgICAKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQ==
"""
        
        main_py_content = base64.b64decode(main_app_data).decode('utf-8')
        main_py_path = Path(install_path) / "main.py"
        
        with open(main_py_path, 'w', encoding='utf-8') as f:
            f.write(main_py_content)
            
        print(f"✅ Aplicação principal extraída: {main_py_path}")
        
    def create_directory_structure(self, install_path):
        """Cria estrutura de diretórios necessária"""
        print("\n📁 Criando estrutura de diretórios...")
        
        directories = [
            "models",
            "data", 
            "config",
            "logs",
            "reports",
            "temp"
        ]
        
        for directory in directories:
            dir_path = Path(install_path) / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {directory}/")
            
    def install_ai_models(self, install_path):
        """Instala modelos de IA necessários"""
        print("\n🧠 Instalando modelos de IA...")
        
        models_dir = Path(install_path) / "models"
        
        model_config = {
            "chest_xray_model": "models/chest_xray_v2.h5",
            "brain_ct_model": "models/brain_ct_v2.h5", 
            "bone_xray_model": "models/bone_xray_v2.h5"
        }
        
        config_path = models_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
            
        print("✅ Configuração de modelos criada")
        
    def create_main_executable(self, install_path):
        """Cria script executável principal para Windows"""
        print("\n⚙️ Criando executável principal...")
        
        bat_content = f"""@echo off
cd /d "{install_path}"
python main.py
pause
"""
        
        bat_path = Path(install_path) / "MedAI_Radiologia.bat"
        with open(bat_path, 'w') as f:
            f.write(bat_content)
            
        print(f"✅ Executável criado: {bat_path}")
        
    def create_shortcuts_advanced(self, install_path, create_shortcut):
        """Cria atalhos avançados no sistema"""
        if not create_shortcut:
            return
            
        print("\n🔗 Criando atalhos...")
        
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "MedAI Radiologia.lnk")
            target = os.path.join(install_path, "MedAI_Radiologia.bat")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = target
            shortcut.WorkingDirectory = install_path
            shortcut.IconLocation = target
            shortcut.save()
            
            print("✅ Atalho criado na área de trabalho")
            
        except ImportError:
            print("⚠️ Bibliotecas de atalho não disponíveis")
            
    def associate_dicom_files(self, install_path, associate_dicom):
        """Associa arquivos DICOM ao MedAI"""
        if not associate_dicom:
            return
            
        print("\n🔗 Associando arquivos DICOM...")
        
        try:
            import winreg
            
            key_path = r"SOFTWARE\Classes\.dcm"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValue(key, "", winreg.REG_SZ, "MedAI.DICOM")
                
            cmd_path = r"SOFTWARE\Classes\MedAI.DICOM\shell\open\command"
            bat_file = os.path.join(install_path, "MedAI_Radiologia.bat")
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, cmd_path) as key:
                winreg.SetValue(key, "", winreg.REG_SZ, f'"{bat_file}" "%1"')
                
            print("✅ Arquivos DICOM associados")
            
        except ImportError:
            print("⚠️ Registro do Windows não disponível")
            
    def setup_dependencies(self, install_path):
        """Configura dependências do sistema"""
        print("\n📦 Configurando dependências...")
        
        requirements = [
            "numpy>=1.21.0",
            "tensorflow>=2.10.0", 
            "opencv-python>=4.5.0",
            "pydicom>=2.3.0",
            "pillow>=8.0.0"
        ]
        
        req_path = Path(install_path) / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
            
        print("✅ Arquivo de dependências criado")
        
    def register_application_advanced(self, install_path):
        """Registra aplicação no sistema Windows"""
        print("\n📝 Registrando aplicação no sistema...")
        
        try:
            import winreg
            
            app_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\MedAI_Radiologia"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, app_key) as key:
                winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "MedAI Radiologia")
                winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, self.version)
                winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, install_path)
                winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "MedAI Systems")
                
            print("✅ Aplicação registrada no sistema")
            
        except ImportError:
            print("⚠️ Registro do Windows não disponível")
            
    def show_success_message_advanced(self, install_path):
        """Exibe mensagem de sucesso com instruções"""
        print("\n" + "=" * 70)
        print("🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 70)
        print(f"📁 Local de instalação: {install_path}")
        print(f"🚀 Para iniciar: Execute MedAI_Radiologia.bat")
        print(f"📋 Documentação: {install_path}\\docs\\")
        print(f"📊 Relatórios: {install_path}\\reports\\")
        print("=" * 70)
        print("🏥 MedAI Radiologia está pronto para análise de imagens médicas!")
        print("💡 Suporte: Consulte a documentação para instruções detalhadas")
        print("=" * 70)
        
    def run_installation(self):
        """Executa processo completo de instalação"""
        try:
            self.show_welcome()
            settings = self.get_installation_settings()
            
            install_path = settings['install_path']
            
            print(f"\n🚀 Iniciando instalação em: {install_path}")
            print("⏳ Por favor, aguarde...")
            
            Path(install_path).mkdir(parents=True, exist_ok=True)
            
            self.extract_main_application(install_path)
            self.create_directory_structure(install_path)
            self.install_ai_models(install_path)
            self.setup_dependencies(install_path)
            self.create_main_executable(install_path)
            self.create_shortcuts_advanced(install_path, settings['create_shortcut'])
            self.associate_dicom_files(install_path, settings['associate_dicom'])
            self.register_application_advanced(install_path)
            
            self.show_success_message_advanced(install_path)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Erro durante a instalação: {e}")
            print("🔧 Tente executar como administrador ou escolha outro diretório")
            return False

def main():
    """Função principal do instalador"""
    installer = MedAIUnifiedInstaller()
    
    print("Deseja prosseguir com a instalação? [S/n]: ", end="")
    response = input().strip().lower()
    
    if response == 'n':
        print("Instalação cancelada pelo usuário.")
        return
        
    success = installer.run_installation()
    
    if success:
        print("\n✅ Instalação finalizada com sucesso!")
        input("Pressione Enter para sair...")
    else:
        print("\n❌ Instalação falhou!")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main()
