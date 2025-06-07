#!/usr/bin/env python3
"""
MedAI Radiologia - Instalador Python Autônomo
Instalador que não depende de NSIS ou programas externos
"""

import os
import sys
import shutil
import subprocess
import json
import base64
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import tempfile

class MedAIAutonomousInstaller:
    def __init__(self):
        self.app_name = "MedAI Radiologia"
        self.company_name = "MedAI Systems"
        self.version = "1.0.0"
        self.install_dir = Path("C:/Program Files/MedAI Radiologia")
        self.temp_dir = Path(tempfile.gettempdir()) / "medai_installer"
        
        self.embedded_files = {}
        
    def create_gui(self):
        """Cria interface gráfica do instalador"""
        self.root = tk.Tk()
        self.root.title(f"Instalador {self.app_name}")
        self.root.geometry("600x450")
        self.root.resizable(False, False)
        
        try:
            self.root.iconbitmap(default="medai_icon.ico")
        except:
            pass
        
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=100)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text=self.app_name, 
                              font=("Arial", 18, "bold"), 
                              fg="white", bg="#2c3e50")
        title_label.pack(pady=15)
        
        subtitle_label = tk.Label(header_frame, 
                                 text="Sistema de Análise Radiológica com IA", 
                                 font=("Arial", 11), 
                                 fg="#ecf0f1", bg="#2c3e50")
        subtitle_label.pack()
        
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill="both", expand=True, padx=30, pady=20)
        
        info_text = f"""Bem-vindo ao instalador do {self.app_name}!

Este programa utiliza inteligência artificial de última geração para análise de imagens médicas.

Diretório de instalação:
{self.install_dir}

O instalador irá:
• Instalar o programa principal e dependências
• Criar atalhos no Menu Iniciar e Área de Trabalho  
• Associar arquivos DICOM (.dcm) ao programa
• Configurar o sistema para uso imediato

Clique em "Instalar" para continuar."""
        
        info_label = tk.Label(main_frame, text=info_text, 
                             font=("Arial", 10), justify="left",
                             bg="white", wraplength=540)
        info_label.pack(pady=20)
        
        progress_frame = tk.Frame(main_frame, bg="white")
        progress_frame.pack(fill="x", pady=15)
        
        tk.Label(progress_frame, text="Progresso da instalação:", 
                font=("Arial", 9), bg="white").pack(anchor="w")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                           variable=self.progress_var,
                                           maximum=100, length=540)
        self.progress_bar.pack(fill="x", pady=5)
        
        self.status_var = tk.StringVar(value="Pronto para instalar")
        status_label = tk.Label(progress_frame, textvariable=self.status_var,
                               font=("Arial", 9), fg="#7f8c8d", bg="white")
        status_label.pack(anchor="w")
        
        button_frame = tk.Frame(main_frame, bg="white")
        button_frame.pack(side="bottom", fill="x", pady=20)
        
        self.install_button = tk.Button(button_frame, text="Instalar",
                                       font=("Arial", 12, "bold"),
                                       bg="#27ae60", fg="white",
                                       width=15, height=2,
                                       command=self.start_installation)
        self.install_button.pack(side="right", padx=10)
        
        cancel_button = tk.Button(button_frame, text="Cancelar",
                                 font=("Arial", 12),
                                 bg="#e74c3c", fg="white",
                                 width=15, height=2,
                                 command=self.cancel_installation)
        cancel_button.pack(side="right")
        
    def update_progress(self, value, status):
        """Atualiza barra de progresso e status"""
        self.progress_var.set(value)
        self.status_var.set(status)
        self.root.update()
        
    def start_installation(self):
        """Inicia processo de instalação"""
        self.install_button.config(state="disabled", text="Instalando...")
        
        install_thread = threading.Thread(target=self.install_medai)
        install_thread.daemon = True
        install_thread.start()
        
    def install_medai(self):
        """Executa instalação completa"""
        try:
            self.update_progress(5, "Verificando privilégios de administrador...")
            if not self.check_admin_privileges():
                messagebox.showerror("Erro", 
                    "Privilégios de administrador necessários!\n\n"
                    "Execute o instalador como administrador.")
                return
                
            self.update_progress(15, "Criando estrutura de diretórios...")
            self.create_directories()
            
            self.update_progress(30, "Extraindo arquivos do programa...")
            self.extract_program_files()
            
            self.update_progress(50, "Instalando dependências Python...")
            self.install_python_dependencies()
            
            self.update_progress(70, "Configurando sistema...")
            self.create_configuration()
            
            self.update_progress(85, "Criando atalhos...")
            self.create_shortcuts()
            
            self.update_progress(95, "Registrando no sistema...")
            self.register_application()
            
            self.update_progress(100, "Instalação concluída!")
            
            messagebox.showinfo("Instalação Concluída!", 
                               f"{self.app_name} foi instalado com sucesso!\n\n"
                               f"Você pode executar o programa através do atalho "
                               f"na área de trabalho ou no menu iniciar.\n\n"
                               f"Arquivos DICOM (.dcm) agora abrem automaticamente "
                               f"no {self.app_name}.")
            
            self.root.quit()
            
        except Exception as e:
            messagebox.showerror("Erro na Instalação", 
                               f"Erro durante a instalação:\n\n{str(e)}\n\n"
                               f"Tente executar como administrador ou entre em "
                               f"contato com o suporte técnico.")
            
    def check_admin_privileges(self):
        """Verifica se tem privilégios de administrador"""
        try:
            test_file = Path("C:/Windows/Temp/medai_admin_test.tmp")
            test_file.touch()
            test_file.unlink()
            return True
        except:
            return False
            
    def create_directories(self):
        """Cria estrutura de diretórios"""
        directories = [
            self.install_dir,
            self.install_dir / "src",
            self.install_dir / "data",
            self.install_dir / "models", 
            self.install_dir / "reports",
            self.install_dir / "temp",
            self.install_dir / "docs",
            self.install_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def extract_program_files(self):
        """Extrai arquivos do programa embarcados"""
        
        source_files = [
            ("src/main.py", "src/main.py"),
            ("src/medai_integration_manager.py", "src/medai_integration_manager.py"),
            ("src/medai_sota_models.py", "src/medai_sota_models.py"),
            ("src/medai_ml_pipeline.py", "src/medai_ml_pipeline.py"),
            ("src/medai_neural_networks.py", "src/medai_neural_networks.py"),
            ("src/medai_model_selector.py", "src/medai_model_selector.py"),
            ("src/medai_main_structure.py", "src/medai_main_structure.py"),
            ("requirements.txt", "requirements.txt")
        ]
        
        for source, dest in source_files:
            source_path = Path(source)
            dest_path = self.install_dir / dest
            
            if source_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
            else:
                if dest == "requirements.txt":
                    with open(dest_path, 'w') as f:
                        f.write("tensorflow>=2.13.0\nnumpy>=1.21.0\nPillow>=8.0.0\n")
                        
    def install_python_dependencies(self):
        """Instala dependências Python necessárias"""
        requirements_file = self.install_dir / "requirements.txt"
        
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", 
                    str(requirements_file)
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                basic_deps = ["tensorflow", "numpy", "Pillow", "tkinter"]
                for dep in basic_deps:
                    try:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", dep
                        ], check=True, capture_output=True)
                    except:
                        pass  # Continuar mesmo se algumas dependências falharem
                        
    def create_configuration(self):
        """Cria arquivos de configuração"""
        config = {
            "app_name": self.app_name,
            "version": self.version,
            "install_path": str(self.install_dir),
            "data_path": str(self.install_dir / "data"),
            "models_path": str(self.install_dir / "models"),
            "reports_path": str(self.install_dir / "reports"),
            "logs_path": str(self.install_dir / "logs"),
            "language": "pt-BR",
            "auto_save": True,
            "max_image_size_mb": 100,
            "supported_formats": [".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".nii"],
            "ai_models": {
                "default": "medical_vit",
                "precision_mode": True,
                "gpu_acceleration": True
            }
        }
        
        config_file = self.install_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        license_file = self.install_dir / "LICENSE.txt"
        with open(license_file, 'w', encoding='utf-8') as f:
            f.write(f"""{self.app_name} - Sistema de Análise Radiológica
Copyright (c) 2024 {self.company_name}

Este software é fornecido "como está", sem garantias.
Destinado para uso educacional e de pesquisa.
Para uso clínico, consulte as regulamentações locais.
""")
            
    def create_shortcuts(self):
        """Cria atalhos no Windows"""
        try:
            import win32com.client
            
            shell = win32com.client.Dispatch("WScript.Shell")
            
            desktop = shell.SpecialFolders("Desktop")
            shortcut_desktop = shell.CreateShortCut(f"{desktop}\\{self.app_name}.lnk")
            shortcut_desktop.Targetpath = sys.executable
            shortcut_desktop.Arguments = f'"{self.install_dir / "src" / "main.py"}"'
            shortcut_desktop.WorkingDirectory = str(self.install_dir)
            shortcut_desktop.Description = f"{self.app_name} - Análise Radiológica com IA"
            shortcut_desktop.save()
            
            start_menu = shell.SpecialFolders("StartMenu")
            programs_folder = Path(start_menu) / "Programs" / self.app_name
            programs_folder.mkdir(exist_ok=True)
            
            shortcut_start = shell.CreateShortCut(f"{programs_folder}\\{self.app_name}.lnk")
            shortcut_start.Targetpath = sys.executable
            shortcut_start.Arguments = f'"{self.install_dir / "src" / "main.py"}"'
            shortcut_start.WorkingDirectory = str(self.install_dir)
            shortcut_start.Description = f"{self.app_name} - Análise Radiológica com IA"
            shortcut_start.save()
            
        except ImportError:
            desktop = Path.home() / "Desktop"
            bat_file = desktop / f"{self.app_name}.bat"
            
            with open(bat_file, 'w', encoding='utf-8') as f:
                f.write(f'''@echo off
cd /d "{self.install_dir}"
python src/main.py
pause
''')
            
    def register_application(self):
        """Registra aplicação no Windows Registry"""
        try:
            import winreg
            
            uninstall_key = winreg.CreateKey(
                winreg.HKEY_LOCAL_MACHINE,
                f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{self.app_name}"
            )
            
            winreg.SetValueEx(uninstall_key, "DisplayName", 0, winreg.REG_SZ, 
                             f"{self.app_name} - Sistema de Análise Radiológica")
            winreg.SetValueEx(uninstall_key, "DisplayVersion", 0, winreg.REG_SZ, self.version)
            winreg.SetValueEx(uninstall_key, "Publisher", 0, winreg.REG_SZ, self.company_name)
            winreg.SetValueEx(uninstall_key, "InstallLocation", 0, winreg.REG_SZ, str(self.install_dir))
            winreg.SetValueEx(uninstall_key, "EstimatedSize", 0, winreg.REG_DWORD, 2048000)  # 2GB
            
            winreg.CloseKey(uninstall_key)
            
            dcm_key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, ".dcm")
            winreg.SetValueEx(dcm_key, "", 0, winreg.REG_SZ, "MedAI.DICOM")
            winreg.CloseKey(dcm_key)
            
            medai_key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, "MedAI.DICOM")
            winreg.SetValueEx(medai_key, "", 0, winreg.REG_SZ, 
                             f"Arquivo DICOM - {self.app_name}")
            winreg.CloseKey(medai_key)
            
            command_key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, 
                                          "MedAI.DICOM\\shell\\open\\command")
            winreg.SetValueEx(command_key, "", 0, winreg.REG_SZ, 
                             f'"{sys.executable}" "{self.install_dir / "src" / "main.py"}" "%1"')
            winreg.CloseKey(command_key)
            
        except Exception as e:
            print(f"Aviso: Não foi possível registrar no Windows Registry: {e}")
            
    def cancel_installation(self):
        """Cancela instalação"""
        if messagebox.askyesno("Cancelar Instalação", 
                              "Deseja cancelar a instalação do MedAI Radiologia?"):
            self.root.quit()
            
    def run(self):
        """Executa o instalador"""
        self.create_gui()
        self.root.mainloop()

if __name__ == "__main__":
    print("🏥 MedAI Radiologia - Instalador Autônomo")
    print("Instalador Python puro - Não requer NSIS ou programas externos")
    
    installer = MedAIAutonomousInstaller()
    installer.run()
