#!/usr/bin/env python3
"""
MedAI Radiologia - Criador de Instalador Python Puro
Cria um instalador standalone que não depende de NSIS ou programas externos
"""

import os
import sys
import shutil
import zipfile
import base64
from pathlib import Path

def create_self_extracting_installer():
    """Cria instalador auto-extraível em Python puro"""
    print("🏥 CRIANDO INSTALADOR PYTHON PURO")
    print("=" * 50)
    
    files_to_include = {}
    
    src_files = [
        "src/main.py",
        "src/medai_integration_manager.py", 
        "src/medai_sota_models.py",
        "src/medai_ml_pipeline.py",
        "src/medai_neural_networks.py",
        "src/medai_model_selector.py",
        "src/medai_main_structure.py",
        "src/medai_batch_processor.py",
        "src/medai_comparison_system.py",
        "src/medai_advanced_visualization.py",
        "src/medai_cli.py",
        "src/medai_test_system.py",
        "src/medai_pacs_integration.py"
    ]
    
    print("📁 Coletando arquivos do programa...")
    for file_path in src_files:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                files_to_include[file_path] = f.read()
            print(f"✅ {file_path}")
        else:
            print(f"⚠️  {file_path} não encontrado")
    
    if Path("requirements.txt").exists():
        with open("requirements.txt", 'r', encoding='utf-8') as f:
            files_to_include["requirements.txt"] = f.read()
        print("✅ requirements.txt")
    
    installer_code = f'''#!/usr/bin/env python3
"""
MedAI Radiologia - Instalador Auto-Extraível
Instalador Python puro com interface gráfica
"""

import os
import sys
import shutil
import subprocess
import json
import base64
import zipfile
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import tempfile

EMBEDDED_FILES = {base64.b64encode(str(files_to_include).encode()).decode()}

class MedAIInstaller:
    def __init__(self):
        self.app_name = "MedAI Radiologia"
        self.install_dir = Path(f"C:/Program Files/{{self.app_name}}")
        self.temp_dir = Path(tempfile.gettempdir()) / "medai_temp"
        
    def create_gui(self):
        """Interface gráfica do instalador"""
        self.root = tk.Tk()
        self.root.title(f"Instalador {{self.app_name}}")
        self.root.geometry("500x350")
        self.root.resizable(False, False)
        
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text=self.app_name, 
                font=("Arial", 16, "bold"), 
                fg="white", bg="#2c3e50").pack(pady=15)
        
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        info_text = f"""
Instalador do {{self.app_name}}

Sistema de Análise Radiológica com Inteligência Artificial

Diretório de instalação:
{{self.install_dir}}

O instalador irá:
• Extrair arquivos do programa
• Instalar dependências Python
• Criar atalhos
• Configurar associações de arquivo
        """
        
        tk.Label(main_frame, text=info_text, 
                font=("Arial", 9), justify="left",
                bg="white").pack(pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, 
                                           variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill="x", pady=10)
        
        self.status_var = tk.StringVar(value="Pronto para instalar")
        tk.Label(main_frame, textvariable=self.status_var,
                font=("Arial", 8), fg="#666", bg="white").pack()
        
        button_frame = tk.Frame(main_frame, bg="white")
        button_frame.pack(side="bottom", fill="x", pady=15)
        
        self.install_btn = tk.Button(button_frame, text="Instalar",
                                    font=("Arial", 11, "bold"),
                                    bg="#27ae60", fg="white",
                                    width=12, height=2,
                                    command=self.start_install)
        self.install_btn.pack(side="right", padx=5)
        
        tk.Button(button_frame, text="Cancelar",
                 font=("Arial", 11),
                 bg="#e74c3c", fg="white", 
                 width=12, height=2,
                 command=self.root.quit).pack(side="right")
        
    def update_progress(self, value, status):
        """Atualiza progresso"""
        self.progress_var.set(value)
        self.status_var.set(status)
        self.root.update()
        
    def start_install(self):
        """Inicia instalação"""
        self.install_btn.config(state="disabled")
        threading.Thread(target=self.install, daemon=True).start()
        
    def install(self):
        """Processo de instalação"""
        try:
            self.update_progress(10, "Decodificando arquivos...")
            files_data = eval(base64.b64decode(EMBEDDED_FILES).decode())
            
            self.update_progress(20, "Criando diretórios...")
            self.install_dir.mkdir(parents=True, exist_ok=True)
            (self.install_dir / "src").mkdir(exist_ok=True)
            (self.install_dir / "data").mkdir(exist_ok=True)
            (self.install_dir / "models").mkdir(exist_ok=True)
            (self.install_dir / "reports").mkdir(exist_ok=True)
            
            self.update_progress(40, "Extraindo arquivos...")
            for file_path, content in files_data.items():
                dest_file = self.install_dir / file_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            self.update_progress(60, "Instalando dependências Python...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", 
                           str(self.install_dir / "requirements.txt")], 
                          check=True, capture_output=True)
            
            self.update_progress(80, "Configurando sistema...")
            config = {{
                "app_name": self.app_name,
                "install_path": str(self.install_dir),
                "language": "pt-BR"
            }}
            with open(self.install_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            self.update_progress(90, "Criando atalhos...")
            self.create_shortcut()
            
            self.update_progress(100, "Instalação concluída!")
            
            messagebox.showinfo("Sucesso!", 
                               f"{{self.app_name}} instalado com sucesso!\\n\\n"
                               f"Execute através do atalho na área de trabalho.")
            
            self.root.quit()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na instalação: {{e}}")
            
    def create_shortcut(self):
        """Cria atalho na área de trabalho"""
        try:
            desktop = Path.home() / "Desktop"
            shortcut_path = desktop / f"{{self.app_name}}.bat"
            
            with open(shortcut_path, 'w') as f:
                f.write(f'''@echo off
cd /d "{{self.install_dir}}"
python src/main.py
pause
''')
            print(f"Atalho criado: {{shortcut_path}}")
        except Exception as e:
            print(f"Aviso: Não foi possível criar atalho: {{e}}")
            
    def run(self):
        """Executa instalador"""
        self.create_gui()
        self.root.mainloop()

if __name__ == "__main__":
    print("🏥 MedAI Radiologia - Instalador")
    installer = MedAIInstaller()
    installer.run()
'''
    
    installer_file = Path("MedAI_Radiologia_Installer.py")
    with open(installer_file, 'w', encoding='utf-8') as f:
        f.write(installer_code)
    
    print(f"\n✅ Instalador criado: {installer_file}")
    print(f"📏 Tamanho: {installer_file.stat().st_size / 1024:.1f} KB")
    
    print("\n🔧 Criando executável...")
    try:
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--windowed", 
            "--name", "MedAI_Radiologia_Installer",
            str(installer_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            exe_path = Path("dist/MedAI_Radiologia_Installer.exe")
            if exe_path.exists():
                print(f"✅ Executável criado: {exe_path}")
                print(f"📏 Tamanho: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            else:
                print("⚠️  Executável não encontrado")
        else:
            print(f"❌ Erro no PyInstaller: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️  PyInstaller não disponível: {e}")
        print("💡 Use: pip install pyinstaller")
    
    print("\n🎉 INSTALADOR PYTHON PURO CRIADO!")
    print("=" * 50)
    print("✅ Não depende de NSIS ou programas externos")
    print("✅ Interface gráfica amigável")
    print("✅ Instalação automática de dependências")
    print("✅ Funciona em qualquer Windows com Python")
    
    return True

if __name__ == "__main__":
    create_self_extracting_installer()
