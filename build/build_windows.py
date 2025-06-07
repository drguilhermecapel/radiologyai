"""
Script para construir executável Windows do MedAI
Usa PyInstaller para criar um executável standalone
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def build_windows_executable():
    """Constrói executável Windows usando PyInstaller"""
    
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    pyinstaller_cmd = [
        "pyinstaller",
        "--onefile",                    # Arquivo único
        "--windowed",                   # Sem console
        "--name=MedAI_Radiologia",      # Nome do executável
        "--icon=icons/medai_icon.ico",  # Ícone (se disponível)
        "--add-data=models;models",     # Incluir modelos
        "--add-data=data;data",         # Incluir dados
        "--hidden-import=tensorflow",
        "--hidden-import=pydicom",
        "--hidden-import=cv2",
        "--hidden-import=PIL",
        "--hidden-import=PyQt5",
        "--hidden-import=numpy",
        "--hidden-import=matplotlib",
        "--hidden-import=scikit-learn",
        "--hidden-import=pandas",
        "--hidden-import=h5py",
        "--hidden-import=SimpleITK",
        "--hidden-import=nibabel",
        "--hidden-import=skimage",
        "--hidden-import=reportlab",
        "--hidden-import=cryptography",
        "--hidden-import=pyqtgraph",
        "--hidden-import=vtk",
        str(src_dir / "main.py")
    ]
    
    try:
        print("Iniciando build do executável Windows...")
        result = subprocess.run(pyinstaller_cmd, cwd=project_root, check=True)
        print("Build concluído com sucesso!")
        
        exe_path = dist_dir / "MedAI_Radiologia.exe"
        if exe_path.exists():
            print(f"Executável criado: {exe_path}")
            return str(exe_path)
        else:
            print("Erro: Executável não foi encontrado")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Erro no build: {e}")
        return None

if __name__ == "__main__":
    build_windows_executable()
