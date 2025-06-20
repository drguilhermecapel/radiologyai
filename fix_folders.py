import os
import shutil

# Diretorio base
base_dir = "data/nih_chest_xray/organized"

# Mapeamento de nomes
rename_map = {
    "No Finding": "normal",
    "Pneumonia": "pneumonia", 
    "Effusion": "pleural_effusion",
    "Mass": "tumor",
    "Nodule": "tumor"  # Agrupar nodulo com tumor
}

# Renomear pastas
for old_name, new_name in rename_map.items():
    old_path = os.path.join(base_dir, old_name)
    new_path = os.path.join(base_dir, new_name)
    
    if os.path.exists(old_path):
        if os.path.exists(new_path):
            # Se ja existe, mover arquivos
            for file in os.listdir(old_path):
                shutil.move(
                    os.path.join(old_path, file),
                    os.path.join(new_path, file)
                )
            os.rmdir(old_path)
        else:
            # Renomear diretamente
            os.rename(old_path, new_path)
        print(f"Renomeado: {old_name} -> {new_name}")

# Criar pasta fracture vazia se necessario
fracture_path = os.path.join(base_dir, "fracture")
if not os.path.exists(fracture_path):
    os.makedirs(fracture_path)
    print("Criada pasta: fracture")

print("\nPastas atualizadas!")