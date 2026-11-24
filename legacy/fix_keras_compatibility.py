import os

# Arquivo a ser corrigido
file_path = "src/medai_ml_pipeline.py"

# Ler o arquivo
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Substituir o import incorreto
content = content.replace(
    "tf.keras.utils.image_utils.apply_affine_transform",
    "tf.keras.preprocessing.image.apply_affine_transform"
)

# Salvar arquivo corrigido
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Arquivo corrigido com sucesso!")