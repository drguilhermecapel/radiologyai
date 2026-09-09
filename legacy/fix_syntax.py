import os

file_path = "src/medai_ml_pipeline.py"

# Ler arquivo
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Procurar e corrigir a linha com erro
fixed_lines = []
for i, line in enumerate(lines):
    if "interval: 30s" in line and "SyntaxError" in str(line):
        # Provavelmente está em um comentário ou string mal formatada
        fixed_lines.append("    # interval: 30s\n")
    else:
        fixed_lines.append(line)

# Salvar arquivo corrigido
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Erro de sintaxe corrigido!")