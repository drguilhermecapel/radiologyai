"""
Script para baixar e preparar datasets médicos públicos
"""

import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path
import kaggle  # pip install kaggle
import wget
from sklearn.model_selection import train_test_split

def download_nih_chest_xray(output_dir="datasets/raw_downloads"):
    """Baixa o dataset NIH Chest X-ray"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar API do Kaggle (precisa de kaggle.json)
    # Baixe de: https://www.kaggle.com/settings -> Create New API Token
    
    print("📥 Baixando NIH Chest X-ray Dataset...")
    kaggle.api.dataset_download_files(
        'nih-chest-xrays/data',
        path=output_dir,
        unzip=True
    )
    print("✅ Download concluído!")

def organize_chest_xray_dataset(raw_dir, output_dir, test_size=0.2, val_size=0.1):
    """Organiza imagens em estrutura de pastas para treinamento"""
    
    # Ler CSV com labels
    csv_path = os.path.join(raw_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    
    # Mapear condições para classes simplificadas
    class_mapping = {
        'No Finding': 'normal',
        'Pneumonia': 'pneumonia',
        'Effusion': 'pleural_effusion',
        'Infiltration': 'infiltration',
        'Atelectasis': 'atelectasis',
        'Mass': 'mass',
        'Nodule': 'nodule',
        'Pneumothorax': 'pneumothorax'
    }
    
    # Criar estrutura de diretórios
    for split in ['train', 'val', 'test']:
        for class_name in class_mapping.values():
            os.makedirs(f"{output_dir}/{split}/{class_name}", exist_ok=True)
    
    # Processar imagens
    print("🔄 Organizando imagens...")
    
    # Para cada imagem no DataFrame
    for idx, row in df.iterrows():
        image_name = row['Image Index']
        findings = row['Finding Labels'].split('|')
        
        # Determinar classe principal
        main_class = 'normal'
        for finding in findings:
            if finding in class_mapping:
                main_class = class_mapping[finding]
                break
        
        # Determinar split (train/val/test)
        rand_val = hash(image_name) % 100
        if rand_val < test_size * 100:
            split = 'test'
        elif rand_val < (test_size + val_size) * 100:
            split = 'val'
        else:
            split = 'train'
        
        # Copiar imagem
        src = f"{raw_dir}/images/{image_name}"
        dst = f"{output_dir}/{split}/{main_class}/{image_name}"
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
        
        if idx % 1000 == 0:
            print(f"  Processadas {idx} imagens...")
    
    print("✅ Dataset organizado!")

def prepare_covid_dataset(output_dir="datasets/chest_xray"):
    """Prepara dataset COVID-19"""
    print("📥 Baixando COVID-19 Dataset...")
    
    url = "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/download"
    wget.download(url, out="covid19_dataset.zip")
    
    # Extrair
    with zipfile.ZipFile("covid19_dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("datasets/raw_downloads/covid19")
    
    # Organizar em pastas
    # ... código similar ao anterior ...

if __name__ == "__main__":
    # Baixar datasets
    download_nih_chest_xray()
    
    # Organizar para treinamento
    organize_chest_xray_dataset(
        raw_dir="datasets/raw_downloads/data",
        output_dir="datasets/chest_xray"
    )