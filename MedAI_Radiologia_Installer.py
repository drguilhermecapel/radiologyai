#!/usr/bin/env python3
"""
MedAI Radiologia - Instalador Autônomo Windows
Instalador Python puro que não depende de NSIS ou programas externos
Versão: 1.0.0 - CORRIGIDA E FUNCIONAL
"""

import os
import sys
import shutil
import subprocess
import json
import base64
from pathlib import Path
import tempfile
import platform

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import threading
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️  Interface gráfica não disponível, usando modo texto")

# Dados embarcados da aplicação principal e requirements
EMBEDDED_FILES_DATA = "eyJzcmMvbWFpbi5weSI6ICIjIS91c3IvYmluL2VudiBweXRob24zXG5cIlwiXCJcbk1lZEFJIFJhZGlvbG9naWEgLSBBcGxpY2HDp8OjbyBQcmluY2lwYWxcblNpc3RlbWEgZGUgQW7DoWxpc2UgUmFkaW9sw7NnaWNhIGNvbSBJbnRlbGlnw6puY2lhIEFydGlmaWNpYWxcblwiXCJcIlxuXG5pbXBvcnQgb3NcbmltcG9ydCBzeXNcbmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aFxuXG4jIEFkaWNpb25hciBkaXJldMOzcmlvIGRvIHByb2pldG8gYW8gcGF0aFxuY3VycmVudF9kaXIgPSBQYXRoKF9fZmlsZV9fKS5wYXJlbnRcbnN5cy5wYXRoLmluc2VydCgwLCBzdHIoY3VycmVudF9kaXIpKVxuXG50cnk6XG4gICAgZnJvbSBtZWRhaV9pbnRlZ3JhdGlvbl9tYW5hZ2VyIGltcG9ydCBNZWRBSUludGVncmF0aW9uTWFuYWdlclxuICAgIGZyb20gbWVkYWlfbWFpbl9zdHJ1Y3R1cmUgaW1wb3J0IENvbmZpZ1xuICAgIFxuICAgIGRlZiBtYWluKCk6XG4gICAgICAgIHByaW50KFwiXHUyNjk1IEluaWNpYW5kbyBNZWRBSSBSYWRpb2xvZ2lhLi4uXCIpXG4gICAgICAgIFxuICAgICAgICAjIEluaWNpYWxpemFyIGNvbmZpZ3VyYcOnw6NvXG4gICAgICAgIGNvbmZpZyA9IENvbmZpZygpXG4gICAgICAgIFxuICAgICAgICAjIEluaWNpYWxpemFyIGdlcmVuY2lhZG9yIGRlIGludGVncmHDp8Ojb1xuICAgICAgICBtYW5hZ2VyID0gTWVkQUlJbnRlZ3JhdGlvbk1hbmFnZXIoY29uZmlnKVxuICAgICAgICBcbiAgICAgICAgIyBJbmljaWFyIGFwbGljYcOnw6NvXG4gICAgICAgIG1hbmFnZXIucnVuKClcbiAgICAgICAgXG5leGNlcHQgSW1wb3J0RXJyb3IgYXMgZTpcbiAgICBwcmludChmXCJcXHUyNzc0IEVycm8gZGUgaW1wb3J0YcOnw6NvOiB7ZX1cIilcbiAgICBwcmludChcIlZlcmlmaXF1ZSBzZSB0b2RhcyBhcyBkZXBlbmTDqm5jaWFzIGVzdMOjbyBpbnN0YWxhZGFzXCIpXG4gICAgaW5wdXQoXCJQcmVzc2lvbmUgRW50ZXIgcGFyYSBzYWlyLi4uXCIpXG4gICAgc3lzLmV4aXQoMSlcbmV4Y2VwdCBFeGNlcHRpb24gYXMgZTpcbiAgICBwcmludChmXCJcXHUyNzc0IEVycm8gaW5lc3BlcmFkbzoge2V9XCIpXG4gICAgaW5wdXQoXCJQcmVzc2lvbmUgRW50ZXIgcGFyYSBzYWlyLi4uXCIpXG4gICAgc3lzLmV4aXQoMSlcblxuaWYgX19uYW1lX18gPT0gXCJfX21haW5fX1wiOlxuICAgIG1haW4oKVxuIiwgInJlcXVpcmVtZW50cy50eHQiOiAidGVuc29yZmxvdz49Mi4xMy4wXG5udW1weT49MS4yMS4wXG5QaWxsb3c+PTguMC4wXG5vcGVuY3YtcHl0aG9uPj00LjUuMFxuc2Npa2l0LWxlYXJuPj0xLjAuMFxubWF0cGxvdGxpYj49My41LjBcbnBhbmRhcz49MS4zLjBcbnB5ZGljb20+PTIuMy4wXG4ifQ=="

class MedAIWindowsInstaller:
    def __init__(self):
        self.app_name = "MedAI Radiologia"
        self.company_name = "MedAI Systems"
        self.version = "1.0.0"
        self.install_dir = Path("C:/Program Files/MedAI Radiologia")
        self.progress_var = None
        self.status_var = None
        self.root = None
        
    def run_text_installer(self):
        """Instalador em modo texto (fallback)"""
        print("=" * 60)
        print("INSTALADOR " + self.app_name.upper())
        print("=" * 60)
        print("Sistema de Análise Radiológica com Inteligência Artificial")
        print()
        print("Diretório de instalação: " + str(self.install_dir))
        print()
        
        response = input("Deseja continuar com a instalação? (S/n): ").strip().lower()
        if response in ['n', 'no', 'não']:
            print("Instalação cancelada.")
            return
            
        try:
            self.install_application()
            print("\n✅ Instalação concluída com sucesso!")
            print("Execute o programa através do atalho na área de trabalho.")
        except Exception as e:
            print("\n❌ Erro na instalação: " + str(e))
            
        input("\nPressione Enter para sair...")
        
    def create_gui_installer(self):
        """Instalador com interface gráfica"""
        self.root = tk.Tk()
        self.root.title("Instalador " + self.app_name)
        self.root.geometry("550x400")
        self.root.resizable(False, False)
        
        # Centralizar janela
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text=self.app_name, 
                font=("Arial", 16, "bold"), 
                fg="white", bg="#2c3e50").pack(pady=20)
        
        # Main content
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        info_text = """Bem-vindo ao instalador do MedAI Radiologia!

Sistema de Análise Radiológica com Inteligência Artificial

Diretório de instalação:
C:/Program Files/MedAI Radiologia

O instalador irá:
• Instalar o programa e dependências Python
• Criar atalhos no Menu Iniciar e Área de Trabalho
• Associar arquivos DICOM (.dcm) ao programa
• Configurar o sistema para uso imediato

Clique em "Instalar" para continuar."""
        
        tk.Label(main_frame, text=info_text, 
                font=("Arial", 9), justify="left",
                bg="white", wraplength=500).pack(pady=15)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, 
                                           variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill="x", pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Pronto para instalar")
        tk.Label(main_frame, textvariable=self.status_var,
                font=("Arial", 8), fg="#666", bg="white").pack()
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg="white")
        button_frame.pack(side="bottom", fill="x", pady=15)
        
        self.install_btn = tk.Button(button_frame, text="Instalar",
                                    font=("Arial", 11, "bold"),
                                    bg="#27ae60", fg="white",
                                    width=12, height=2,
                                    command=self.start_gui_installation)
        self.install_btn.pack(side="right", padx=5)
        
        tk.Button(button_frame, text="Cancelar",
                 font=("Arial", 11),
                 bg="#e74c3c", fg="white", 
                 width=12, height=2,
                 command=self.root.quit).pack(side="right")
        
    def update_progress(self, value, status):
        """Atualiza progresso (GUI ou console)"""
        try:
            if GUI_AVAILABLE and self.progress_var:
                self.progress_var.set(value)
            if GUI_AVAILABLE and self.status_var:
                self.status_var.set(status)
            if GUI_AVAILABLE and self.root:
                self.root.update_idletasks()
        except:
            pass
        print(f"[{value}%] {status}")
            
    def start_gui_installation(self):
        """Inicia instalação via GUI"""
        self.install_btn.config(state="disabled", text="Instalando...")
        threading.Thread(target=self.install_with_gui_feedback, daemon=True).start()
        
    def install_with_gui_feedback(self):
        """Instalação com feedback visual"""
        try:
            self.install_application()
            if GUI_AVAILABLE:
                messagebox.showinfo("Instalação Concluída!", 
                                   self.app_name + " foi instalado com sucesso!\n\n"
                                   "Você pode executar o programa através do atalho "
                                   "na área de trabalho ou no menu iniciar.")
                self.root.quit()
        except Exception as e:
            error_msg = f"Erro durante a instalação:\n\n{str(e)}"
            print(f"Installation error: {e}")
            if GUI_AVAILABLE:
                messagebox.showerror("Erro na Instalação", error_msg)
                self.install_btn.config(state="normal", text="Tentar Novamente")
            
    def check_admin_privileges(self):
        """Verifica privilégios de administrador"""
        try:
            # Tenta criar arquivo em local protegido
            if platform.system() == "Windows":
                test_path = Path("C:/Windows/Temp/medai_admin_test.tmp")
            else:
                test_path = Path("/tmp/medai_admin_test.tmp")
            
            test_path.touch()
            test_path.unlink()
            return True
        except (PermissionError, OSError):
            return False
        except Exception:
            return False
            
    def create_directories(self):
        """Cria estrutura de diretórios"""
        directories = [
            self.install_dir,
            self.install_dir / "src",
            self.install_dir / "data",
            self.install_dir / "models",
            self.install_dir / "reports",
            self.install_dir / "logs",
            self.install_dir / "config",
            self.install_dir / "docs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def extract_files(self, files_data):
        """Extrai arquivos embarcados"""
        for file_path, content in files_data.items():
            dest_file = self.install_dir / file_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
    def install_dependencies(self):
        """Instala dependências Python"""
        requirements_file = self.install_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", 
                    str(requirements_file), "--quiet"
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Aviso: Algumas dependências não puderam ser instaladas: {e}")
                
    def create_configuration(self):
        """Cria arquivo de configuração padrão"""
        config = {
            "version": self.version,
            "install_date": str(Path.ctime(self.install_dir)),
            "models": {
                "chest_xray": "models/chest_xray_model.h5",
                "brain_ct": "models/brain_ct_model.h5",
                "bone_xray": "models/bone_xray_model.h5"
            },
            "settings": {
                "auto_save_reports": True,
                "report_format": "PDF",
                "ai_confidence_threshold": 0.85
            }
        }
        
        config_file = self.install_dir / "config" / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def create_shortcuts(self):
        """Cria atalhos no sistema"""
        if platform.system() != "Windows":
            print("Criação de atalhos não disponível neste sistema")
            return
            
        try:
            # Criar arquivo .bat para executar o programa
            bat_content = f"""@echo off
cd /d "{self.install_dir}"
python src/main.py
pause
"""
            bat_file = self.install_dir / "MedAI_Radiologia.bat"
            with open(bat_file, 'w') as f:
                f.write(bat_content)
                
            # Tentar criar atalho na área de trabalho
            desktop = Path.home() / "Desktop"
            if desktop.exists():
                shortcut = desktop / "MedAI Radiologia.lnk"
                # Aqui normalmente usaríamos win32com para criar o atalho
                # Como pode não estar disponível, criamos um .bat como alternativa
                with open(desktop / "MedAI Radiologia.bat", 'w') as f:
                    f.write(f'@echo off\nstart "" "{bat_file}"\n')
                    
        except Exception as e:
            print(f"Aviso: Não foi possível criar atalhos: {e}")
            
    def register_application(self):
        """Registra aplicação no Windows"""
        if platform.system() != "Windows":
            return
            
        try:
            import winreg
            
            # Registrar no Painel de Controle
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\\" + self.app_name.replace(" ", "_")
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
            
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, self.app_name)
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, self.version)
            winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, self.company_name)
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(self.install_dir))
            winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, 
                            f'"{sys.executable}" "{self.install_dir}/uninstall.py"')
            winreg.CloseKey(key)
            
            # Associar arquivos DICOM
            dcm_key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, ".dcm")
            winreg.SetValueEx(dcm_key, "", 0, winreg.REG_SZ, "MedAI.DICOM")
            winreg.CloseKey(dcm_key)
            
            # Criar handler para arquivos DICOM
            handler_key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, "MedAI.DICOM\\shell\\open\\command")
            winreg.SetValueEx(handler_key, "", 0, winreg.REG_SZ, 
                            f'"{self.install_dir}\\MedAI_Radiologia.bat" "%1"')
            winreg.CloseKey(handler_key)
            
        except Exception as e:
            print(f"Aviso: Registro no Windows falhou: {e}")
            
    def install_application(self):
        """Processo principal de instalação"""
        self.update_progress(10, "Verificando privilégios...")
        if not self.check_admin_privileges():
            if platform.system() == "Windows":
                raise Exception(
                    "Privilégios de administrador necessários!\n\n"
                    "Por favor, execute o instalador como administrador:\n"
                    "1. Clique com botão direito no instalador\n"
                    "2. Selecione 'Executar como administrador'"
                )
            else:
                print("⚠️  Executando sem privilégios elevados")
        
        self.update_progress(20, "Decodificando arquivos...")
        try:
            files_data = json.loads(base64.b64decode(EMBEDDED_FILES_DATA).decode())
        except Exception as e:
            raise Exception(f"Erro ao decodificar arquivos: {e}")
        
        self.update_progress(30, "Criando diretórios...")
        self.create_directories()
        
        self.update_progress(50, "Extraindo arquivos...")
        self.extract_files(files_data)
        
        self.update_progress(70, "Instalando dependências Python...")
        self.install_dependencies()
        
        self.update_progress(85, "Configurando sistema...")
        self.create_configuration()
        self.create_shortcuts()
        self.register_application()
        
        self.update_progress(100, "Instalação concluída!")
        
    def run(self):
        """Executa instalador"""
        print("🏥 MedAI Radiologia - Instalador Autônomo")
        print("Sistema de Análise Radiológica com IA")
        print()
        
        if platform.system() != "Windows":
            print("⚠️  Sistema operacional detectado:", platform.system())
            print("Este instalador foi otimizado para Windows.")
            print("Algumas funcionalidades podem não estar disponíveis.")
            print()
        
        if GUI_AVAILABLE:
            try:
                self.create_gui_installer()
                self.root.mainloop()
            except Exception as e:
                print(f"Erro na interface gráfica: {e}")
                print("Usando modo texto...")
                self.run_text_installer()
        else:
            self.run_text_installer()

if __name__ == "__main__":
    installer = MedAIWindowsInstaller()
    installer.run()
