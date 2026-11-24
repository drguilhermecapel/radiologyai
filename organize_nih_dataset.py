import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_nih_dataset(source_dir="D:/NIH_CHEST_XRAY", 
                        output_dir="data/nih_chest_xray/organized"):
    """
    Organiza dataset NIH em estrutura train/val/test
    com divisão estratificada por patologia
    """
    
    # Ler arquivo de anotações
    csv_path = os.path.join(source_dir, "Data_Entry_2017_v2020.csv")
    df = pd.read_csv(csv_path)
    
    # Configurar divisão dos dados
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Criar diretórios
    for split in ['train', 'val', 'test']:
        for pathology in ['No Finding', 'Pneumonia', 'Effusion', 'Atelectasis', 
                         'Consolidation', 'Pneumothorax', 'Cardiomegaly', 
                         'Mass', 'Nodule', 'Infiltration', 'Emphysema', 
                         'Fibrosis', 'Pleural_Thickening', 'Hernia']:
            os.makedirs(f"{output_dir}/{split}/{pathology}", exist_ok=True)
    
    # Processar cada imagem
    print("Organizando imagens por patologia...")
    
    # Primeiro, separar imagens com e sem patologia
    df['has_pathology'] = df['Finding Labels'].apply(lambda x: x != 'No Finding')
    
    # Divisão estratificada
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, 
        stratify=df['has_pathology'], 
        random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_ratio/(train_ratio + val_ratio),
        stratify=train_val_df['has_pathology'],
        random_state=42
    )
    
    # Copiar imagens para estrutura organizada
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"n Processando {split_name}: {len(split_df)} imagens")
        
        for idx, row in split_df.iterrows():
            image_name = row['Image Index']
            findings = row['Finding Labels'].split('|')
            
            # Para multi-label, usar primeira patologia como categoria principal
            main_finding = findings[0]
            
            src = os.path.join(source_dir, "images", image_name)
            dst = os.path.join(output_dir, split_name, main_finding, image_name)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
            
            if idx % 1000 == 0:
                print(f"   Processadas {idx} imagens...")
    
    print("n Dataset organizado com sucesso!")
    
    # Estatísticas
    print("n Estatísticas do dataset:")
    print(f"  Total de imagens: {len(df)}")
    print(f"  Treino: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validação: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Teste: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

# Executar organização
if __name__ == "__main__":
    organize_nih_dataset()