import os
import subprocess

# 1. Criar dataset sintetico se nao existir
if not os.path.exists("data/nih_chest_xray/organized"):
    print("Criando dataset sintetico...")
    subprocess.run(["python", "create_synthetic_dataset.py"])

# 2. Verificar estrutura
print("\nEstrutura do dataset:")
for root, dirs, files in os.walk("data/nih_chest_xray/organized"):
    level = root.replace("data/nih_chest_xray/organized", '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    
# 3. Iniciar treinamento
print("\nIniciando treinamento...")
subprocess.run([
    "python", "train_models.py", 
    "--data_dir", "data/nih_chest_xray/organized",
    "--epochs", "10"
])