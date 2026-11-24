# MedAI Radiologia

Plataforma de IA para radiologia **somente para pesquisa** (*research use only*).
Não é um dispositivo médico e não foi validada para uso clínico: toda saída é um
"achado sugerido para confirmação médica".

> **Estado:** em reconstrução. O código anterior foi removido porque não continha
> modelo treinado nem evidência medida; ele permanece no histórico git
> (commit `16fdd47`). Este README será expandido conforme os módulos entrarem.

## Princípios

1. **Regra de ouro.** O software nunca emite predição, métrica ou mapa de calor que
   um modelo real não tenha computado sobre a entrada real. Falha vira erro
   explícito ou abstenção.
2. **Catálogo de condições com nível de evidência.** Cada achado mostra a força da
   evidência que o sustenta. O nível é calculado a partir de arquivos de
   evidência, nunca digitado:

   | Nível | Significado |
   |---|---|
   | A | Validação externa medida por nós, em dataset público no qual o modelo não foi treinado |
   | B | Validação publicada revisada por pares e/ou limiar de diretriz, citação conferida na fonte |
   | C | Validação interna medida por nós (hold-out por paciente) |
   | D | Exploratório; aparece só no "modo pesquisa" e nunca entra no laudo sozinho |

3. **Protocolo pré-registrado.** Os critérios de aceitação estão em
   `src/medai/catalog/protocol.yaml` e foram commitados antes de qualquer medição.
4. **Médico no circuito.** Revisão e assinatura do laudo ficam vinculadas ao
   conteúdo analisado, com trilha de auditoria.

## Desenvolvimento

```bash
uv sync --all-extras          # acrescente --extra cpu em máquinas sem GPU
uv run pytest
uv run ruff check . && uv run mypy
uv run python tools/lint_no_fabrication.py
```

Regras do projeto para quem contribui (humanos e agentes): [`CLAUDE.md`](CLAUDE.md).
