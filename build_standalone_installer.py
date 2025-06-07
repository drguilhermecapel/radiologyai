#!/usr/bin/env python3
"""
MedAI Radiologia - Gerador de Instalador Standalone
Cria instalador executável que não depende de NSIS ou programas externos
"""

import os
import sys
import shutil
import base64
import json
from pathlib import Path

def collect_program_files():
    """Coleta todos os arquivos necessários do programa"""
    print("📁 Coletando arquivos do programa...")
    
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
    else:
        basic_requirements = """tensorflow>=2.13.0
numpy>=1.21.0
Pillow>=8.0.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
pydicom>=2.3.0
"""
        files_to_include["requirements.txt"] = basic_requirements
        print("✅ requirements.txt (criado)")
    
    return files_to_include

def create_installer_script(embedded_files):
    """Cria script do instalador com arquivos embarcados"""
    print("🔧 Criando script do instalador...")
    
    embedded_data = base64.b64encode(json.dumps(embedded_files).encode()).decode()
    
    installer_script = f'''#!/usr/bin/env python3
"""
MedAI Radiologia - Instalador Autônomo Windows
Instalador Python puro que não depende de NSIS ou programas externos
Versão: 1.0.0
"""

import os
import sys
import shutil
import subprocess
import json
import base64
from pathlib import Path
import tempfile

if os.name != 'nt':
    print("❌ Este instalador é específico para Windows")
    input("Pressione Enter para sair...")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import threading
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️  Interface gráfica não disponível, usando modo texto")

EMBEDDED_FILES_DATA = "{embedded_data}"

class MedAIWindowsInstaller:
    def __init__(self):
        self.app_name = "MedAI Radiologia"
        self.company_name = "MedAI Systems"
        self.version = "1.0.0"
        self.install_dir = Path("C:/Program Files/MedAI Radiologia")
        
    def run_text_installer(self):
        """Instalador em modo texto (fallback)"""
        print("=" * 60)
        print(f"INSTALADOR {self.app_name.upper()}")
        print("=" * 60)
        print("Sistema de Análise Radiológica com Inteligência Artificial")
        print()
        print(f"Diretório de instalação: {{self.install_dir}}")
        print()
        
        response = input("Deseja continuar com a instalação? (S/n): ").strip().lower()
        if response in ['n', 'no', 'não']:
            print("Instalação cancelada.")
            return
            
        try:
            self.install_application()
            print("\\n✅ Instalação concluída com sucesso!")
            print(f"Execute o programa através do atalho na área de trabalho.")
        except Exception as e:
            print(f"\\n❌ Erro na instalação: {{e}}")
            
        input("\\nPressione Enter para sair...")
        
    def create_gui_installer(self):
        """Instalador com interface gráfica"""
        self.root = tk.Tk()
        self.root.title(f"Instalador {{self.app_name}}")
        self.root.geometry("550x400")
        self.root.resizable(False, False)
        
        self.root.eval('tk::PlaceWindow . center')
        
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text=self.app_name, 
                font=("Arial", 16, "bold"), 
                fg="white", bg="#2c3e50").pack(pady=20)
        
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        info_text = f"""Bem-vindo ao instalador do {{self.app_name}}!

Sistema de Análise Radiológica com Inteligência Artificial

Diretório de instalação:
{{self.install_dir}}

O instalador irá:
• Instalar o programa e dependências Python
• Criar atalhos no Menu Iniciar e Área de Trabalho
• Associar arquivos DICOM (.dcm) ao programa
• Configurar o sistema para uso imediato

Clique em "Instalar" para continuar."""
        
        tk.Label(main_frame, text=info_text, 
                font=("Arial", 9), justify="left",
                bg="white", wraplength=500).pack(pady=15)
        
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
                                    command=self.start_gui_installation)
        self.install_btn.pack(side="right", padx=5)
        
        tk.Button(button_frame, text="Cancelar",
                 font=("Arial", 11),
                 bg="#e74c3c", fg="white", 
                 width=12, height=2,
                 command=self.root.quit).pack(side="right")
        
    def update_progress(self, value, status):
        """Atualiza progresso (apenas GUI)"""
        if GUI_AVAILABLE and hasattr(self, 'progress_var'):
            self.progress_var.set(value)
            self.status_var.set(status)
            self.root.update()
        else:
            print(f"[{{value:3.0f}}%] {{status}}")
            
    def start_gui_installation(self):
        """Inicia instalação via GUI"""
        self.install_btn.config(state="disabled", text="Instalando...")
        threading.Thread(target=self.install_with_gui_feedback, daemon=True).start()
        
    def install_with_gui_feedback(self):
        """Instalação com feedback visual"""
        try:
            self.install_application()
            messagebox.showinfo("Instalação Concluída!", 
                               f"{{self.app_name}} foi instalado com sucesso!\\n\\n"
                               f"Você pode executar o programa através do atalho "
                               f"na área de trabalho ou no menu iniciar.")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Erro na Instalação", 
                               f"Erro durante a instalação:\\n\\n{{str(e)}}")
            
    def install_application(self):
        """Processo principal de instalação"""
        self.update_progress(10, "Verificando privilégios...")
        if not self.check_admin_privileges():
            raise Exception("Privilégios de administrador necessários!\\n"
                          "Execute o instalador como administrador.")
        
        self.update_progress(20, "Decodificando arquivos...")
        files_data = json.loads(base64.b64decode(EMBEDDED_FILES_DATA).decode())
        
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
        
    def check_admin_privileges(self):
        """Verifica privilégios de administrador"""
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
            self.install_dir / "logs"
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
                    str(requirements_file)
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                basic_deps = ["tensorflow", "numpy", "Pillow"]
                for dep in basic_deps:
                    try:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", dep
                        ], check=True, capture_output=True)
                    except:
                        pass
                        
    def create_configuration(self):
        """Cria configuração do sistema"""
        config = {{
            "app_name": self.app_name,
            "version": self.version,
            "install_path": str(self.install_dir),
            "language": "pt-BR",
            "supported_formats": [".dcm", ".png", ".jpg", ".jpeg"]
        }}
        
        with open(self.install_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    def create_shortcuts(self):
        """Cria atalhos do Windows"""
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            
            desktop = shell.SpecialFolders("Desktop")
            shortcut = shell.CreateShortCut(f"{{desktop}}\\{{self.app_name}}.lnk")
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{{self.install_dir / "src" / "main.py"}}"'
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.save()
            
        except ImportError:
            desktop = Path.home() / "Desktop"
            bat_file = desktop / f"{{self.app_name}}.bat"
            with open(bat_file, 'w', encoding='utf-8') as f:
                f.write(f'''@echo off
cd /d "{{self.install_dir}}"
python src/main.py
pause
''')
                
    def register_application(self):
        """Registra aplicação no Windows"""
        try:
            import winreg
            
            key = winreg.CreateKey(
                winreg.HKEY_LOCAL_MACHINE,
                f"SOFTWARE\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Uninstall\\\\{{self.app_name}}"
            )
            
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, self.app_name)
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, self.version)
            winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, self.company_name)
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(self.install_dir))
            winreg.CloseKey(key)
            
            dcm_key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, ".dcm")
            winreg.SetValueEx(dcm_key, "", 0, winreg.REG_SZ, "MedAI.DICOM")
            winreg.CloseKey(dcm_key)
            
        except Exception as e:
            print(f"Aviso: Registro no Windows falhou: {{e}}")
            
    def run(self):
        """Executa instalador"""
        print("🏥 MedAI Radiologia - Instalador Autônomo")
        print("Instalador Python puro - Não requer NSIS")
        print()
        
        if GUI_AVAILABLE:
            self.create_gui_installer()
            self.root.mainloop()
        else:
            self.run_text_installer()

if __name__ == "__main__":
    installer = MedAIWindowsInstaller()
    installer.run()
'''
    
    return installer_script

def create_pyinstaller_spec():
    """Cria arquivo .spec para PyInstaller"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['MedAI_Radiologia_Installer.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'threading',
        'json',
        'base64',
        'winreg',
        'win32com.client'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',
        'numpy',
        'matplotlib',
        'scipy',
        'pandas'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyd = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyd,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MedAI_Radiologia_Installer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open("MedAI_Installer.spec", "w") as f:
        f.write(spec_content)
    
    print("✅ Arquivo .spec criado")

def main():
    """Função principal"""
    print("🏥 MEDAI RADIOLOGIA - GERADOR DE INSTALADOR STANDALONE")
    print("=" * 70)
    print("Criando instalador que NÃO depende de NSIS ou programas externos")
    print("=" * 70)
    
    try:
        embedded_files = collect_program_files()
        
        installer_script = create_installer_script(embedded_files)
        
        installer_file = Path("MedAI_Radiologia_Installer.py")
        with open(installer_file, 'w', encoding='utf-8') as f:
            f.write(installer_script)
        
        print(f"✅ Instalador criado: {installer_file}")
        print(f"📏 Tamanho: {installer_file.stat().st_size / 1024:.1f} KB")
        
        create_pyinstaller_spec()
        
        print("\n" + "=" * 70)
        print("🎉 INSTALADOR STANDALONE CRIADO COM SUCESSO!")
        print("=" * 70)
        print("📁 Arquivos gerados:")
        print("   • MedAI_Radiologia_Installer.py (script Python)")
        print("   • MedAI_Installer.spec (configuração PyInstaller)")
        print()
        print("🔧 Para criar executável Windows:")
        print("   1. Execute em máquina Windows com Python")
        print("   2. pip install pyinstaller")
        print("   3. pyinstaller MedAI_Installer.spec")
        print()
        print("✅ Características do instalador:")
        print("   • Não depende de NSIS ou programas externos")
        print("   • Interface gráfica amigável (Windows)")
        print("   • Fallback para modo texto")
        print("   • Instalação automática de dependências")
        print("   • Criação de atalhos e associações")
        print("   • Registro no Windows")
        print("   • Adequado para usuários não-técnicos")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    main()
