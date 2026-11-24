# Script para criar imagens sintéticas de exemplo para teste do pipeline
import numpy as np
import cv2
import os
from pathlib import Path
import pandas as pd

def create_synthetic_chest_xray(width=512, height=512, pathology="No Finding"):
    """Cria uma imagem sintética de raio-X de tórax"""
    
    # Criar imagem base (fundo escuro)
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Adicionar estruturas básicas do tórax
    # Contorno dos pulmões
    center_x, center_y = width // 2, height // 2
    
    # Pulmão esquerdo
    cv2.ellipse(img, (center_x - 80, center_y), (60, 120), 0, 0, 360, 80, -1)
    
    # Pulmão direito
    cv2.ellipse(img, (center_x + 80, center_y), (60, 120), 0, 0, 360, 80, -1)
    
    # Coração (área mais escura)
    cv2.ellipse(img, (center_x - 20, center_y + 40), (40, 60), 0, 0, 360, 60, -1)
    
    # Costelas (linhas horizontais)
    for i in range(8):
        y = center_y - 100 + i * 25
        cv2.line(img, (center_x - 120, y), (center_x + 120, y), 100, 2)
    
    # Adicionar patologias específicas
    if pathology == "Pneumonia":
        # Adicionar opacidades nos pulmões
        cv2.circle(img, (center_x - 60, center_y - 30), 25, 120, -1)
        cv2.circle(img, (center_x + 70, center_y + 20), 20, 110, -1)
        
    elif pathology == "Cardiomegaly":
        # Aumentar o coração
        cv2.ellipse(img, (center_x - 20, center_y + 40), (60, 80), 0, 0, 360, 50, -1)
        
    elif pathology == "Effusion":
        # Adicionar fluido na base dos pulmões
        cv2.rectangle(img, (center_x - 140, center_y + 80), (center_x + 140, center_y + 120), 40, -1)
        
    elif pathology == "Mass":
        # Adicionar massa circular
        cv2.circle(img, (center_x + 50, center_y - 40), 15, 140, -1)
        
    elif pathology == "Nodule":
        # Adicionar nódulos pequenos
        cv2.circle(img, (center_x - 40, center_y - 20), 8, 130, -1)
        cv2.circle(img, (center_x + 30, center_y + 10), 6, 125, -1)
    
    # Adicionar ruído realista
    noise = np.random.normal(0, 10, (height, width))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Aplicar blur leve para simular características de raio-X
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

def create_synthetic_dataset(output_dir="data/nih_chest_xray", num_images_per_class=50):
    """Cria um dataset sintético para teste"""
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    organized_dir = output_path / "organized"
    
    # Criar diretórios
    images_dir.mkdir(parents=True, exist_ok=True)
    
    pathologies = ["No Finding", "Pneumonia", "Cardiomegaly", "Effusion", "Mass", "Nodule"]
    
    # Criar diretórios organizados
    for pathology in pathologies:
        (organized_dir / pathology).mkdir(parents=True, exist_ok=True)
    
    # Lista para CSV
    data_entries = []
    
    image_counter = 1
    
    for pathology in pathologies:
        print(f"Criando {num_images_per_class} imagens para {pathology}...")
        
        for i in range(num_images_per_class):
            # Nome do arquivo
            filename = f"{image_counter:08d}_{i:03d}.png"
            
            # Criar imagem
            img = create_synthetic_chest_xray(pathology=pathology)
            
            # Salvar na pasta images
            img_path = images_dir / filename
            cv2.imwrite(str(img_path), img)
            
            # Salvar na pasta organizada
            organized_path = organized_dir / pathology / filename
            cv2.imwrite(str(organized_path), img)
            
            # Adicionar entrada para CSV
            data_entries.append({
                'Image Index': filename,
                'Finding Labels': pathology,
                'Follow-up #': i,
                'Patient ID': image_counter,
                'Patient Age': np.random.randint(20, 90),
                'Patient Sex': np.random.choice(['M', 'F']),
                'View Position': 'PA',
                'OriginalImage[Width,Height]': '512x512',
                'OriginalImagePixelSpacing[x,y]': '0.143x0.143'
            })
        
        image_counter += 1
    
    # Criar CSV
    df = pd.DataFrame(data_entries)
    csv_path = output_path / "Data_Entry_2017_v2020.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset sintético criado com {len(data_entries)} imagens")
    print(f"Imagens salvas em: {images_dir}")
    print(f"Imagens organizadas em: {organized_dir}")
    print(f"CSV salvo em: {csv_path}")
    
    # Criar estatísticas
    stats = df['Finding Labels'].value_counts().to_dict()
    
    import json
    stats_file = organized_dir / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return df

if __name__ == "__main__":
    # Verificar se pandas está disponível
    try:
        import pandas as pd
    except ImportError:
        print("Pandas não encontrado. Instalando...")
        os.system("pip3 install pandas")
        import pandas as pd
    
    # Criar dataset sintético
    df = create_synthetic_dataset(num_images_per_class=20)  # Menor para teste rápido
    print("Dataset sintético criado com sucesso!")

