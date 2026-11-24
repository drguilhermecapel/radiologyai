import os

file_path = "src/medai_ml_pipeline.py"

# Ler arquivo
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Encontrar a função augment e substitui-la por uma versão simplificada
new_augment = '''def augment(image, label):
    """Augmentação simplificada para compatibilidade"""
    # Aplicar augmentações básicas do TensorFlow
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    
    # Adicionar ruído gaussiano
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label'''

# Encontrar onde começa a função augment
start_idx = content.find("def augment(image, label):")
if start_idx == -1:
    start_idx = content.find("def augment(")

# Encontrar o fim da função (próxima def ou class)
end_idx = content.find("\ndef ", start_idx + 1)
if end_idx == -1:
    end_idx = content.find("\nclass ", start_idx + 1)

# Substituir a função
if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_augment + content[end_idx:]
    
    # Salvar
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Augmentação corrigida!")
else:
    print("Não foi possível encontrar a função augment")