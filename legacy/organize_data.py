# Script para organizar dados do NIH ChestX-ray14
import pandas as pd
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def organize_nih_data(data_dir="data/nih_chest_xray"):
    """
    Organiza dados do NIH ChestX-ray14 por patologia
    
    Args:
        data_dir: Diretório base dos dados
    """
    data_path = Path(data_dir)
    
    # Verificar se arquivo de labels existe
    labels_file = data_path / "Data_Entry_2017_v2020.csv"
    if not labels_file.exists():
        logger.error(f"Arquivo de labels não encontrado: {labels_file}")
        return False
    
    # Ler labels
    logger.info("Carregando arquivo de labels...")
    df = pd.read_csv(labels_file)
    
    # Criar pastas por patologia
    pathologies = ['Pneumonia', 'Effusion', 'Atelectasis', 'Consolidation', 
                   'Pneumothorax', 'Cardiomegaly', 'Mass', 'Nodule', 
                   'No Finding']
    
    organized_dir = data_path / "organized"
    for pathology in pathologies:
        (organized_dir / pathology).mkdir(parents=True, exist_ok=True)
    
    # Organizar imagens
    images_dir = data_path / "images"
    if not images_dir.exists():
        logger.error(f"Diretório de imagens não encontrado: {images_dir}")
        return False
    
    logger.info("Organizando imagens por patologia...")
    processed = 0
    
    for idx, row in df.iterrows():
        image_name = row['Image Index']
        labels = row['Finding Labels'].split('|')
        
        # Determinar pasta de destino
        if 'No Finding' in labels:
            dest_folder = 'No Finding'
        else:
            # Usar primeira patologia encontrada
            dest_folder = labels[0]
        
        src_path = images_dir / image_name
        dest_path = organized_dir / dest_folder / image_name
        
        if src_path.exists():
            try:
                shutil.copy2(src_path, dest_path)
                processed += 1
            except Exception as e:
                logger.warning(f"Erro ao copiar {image_name}: {e}")
        
        if processed % 1000 == 0:
            logger.info(f"Processadas {processed} imagens...")
    
    logger.info(f"Organização concluída. Total de imagens processadas: {processed}")
    
    # Criar estatísticas
    stats = {}
    for pathology in pathologies:
        count = len(list((organized_dir / pathology).glob("*.png")))
        stats[pathology] = count
    
    # Salvar estatísticas
    stats_file = organized_dir / "dataset_stats.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Estatísticas do dataset:")
    for pathology, count in stats.items():
        logger.info(f"  {pathology}: {count} imagens")
    
    return True

if __name__ == "__main__":
    organize_nih_data()

