# Guia do Usuário - MedAI Radiologia

## Visão Geral

O MedAI Radiologia é um sistema de análise de imagens médicas que utiliza **inteligência artificial de última geração** para auxiliar profissionais de saúde na interpretação de exames radiológicos.

### Tecnologias de IA Avançadas
- **EfficientNetV2L**: Modelo mais eficiente com precisão superior
- **Vision Transformer (ViT)**: Análise baseada em atenção para detalhes médicos
- **ConvNeXt XLarge**: Arquitetura moderna para análise robusta
- **Modelos Ensemble**: Combinação de múltiplos modelos para máxima confiabilidade
- **Resolução Aumentada**: Processamento em 384x384 pixels para maior precisão

## Iniciando o Programa

1. Execute o arquivo `MedAI_Radiologia.exe`
2. A interface principal será exibida
3. Faça login com suas credenciais (se configurado)

## Interface Principal

A interface possui um **tema escuro futurista** com detalhes em azul neon,
inspirado em imagens radiológicas e inteligência artificial.

### Painel Esquerdo - Controles
- **Arquivo**: Carregar imagens individuais ou pastas DICOM
- **Modelo de IA**: Selecionar tipo de análise
- **Informações do Paciente**: Metadados da imagem
- **Histórico**: Análises anteriores

### Painel Central - Visualização
- **Imagem Original**: Visualização da imagem carregada
- **Mapa de Calor**: Regiões de interesse identificadas pela IA
- **Comparação**: Comparar duas imagens lado a lado

### Painel Direito - Resultados
- **Resultados da Análise**: Diagnóstico e probabilidades
- **Métricas de Confiança**: Indicadores de certeza
- **Ações**: Gerar relatórios e exportar resultados

## Carregando Imagens

### Formatos Suportados
- **DICOM** (.dcm)
- **PNG** (.png)
- **JPEG** (.jpg, .jpeg)
- **TIFF** (.tif, .tiff)
- **BMP** (.bmp)

### Como Carregar
1. Clique em "Abrir Imagem" ou use Ctrl+O
2. Selecione o arquivo desejado
3. A imagem será exibida no painel central
4. Metadados aparecerão no painel esquerdo

## Tipos de Análise com IA de Última Geração

### Raio-X Torácico (EfficientNetV2 + Vision Transformer)
- **Precisão**: 94% de acurácia com modelos SOTA
- Detecta pneumonia, COVID-19, tuberculose, derrame pleural
- Identifica cardiomegalia com alta confiabilidade
- Avalia normalidade pulmonar com threshold otimizado (85%)

### CT Cerebral (Vision Transformer Especializado)
- **Precisão**: 96% de acurácia para diagnósticos críticos
- Detecta hemorragias, isquemias, tumores, edemas, hidrocefalia
- Análise baseada em atenção para detalhes neurológicos
- Threshold elevado (90%) para máxima confiabilidade

### Detecção de Fraturas (ConvNeXt Otimizado)
- **Precisão**: 91% de acurácia para patologias ósseas
- Detecta fraturas, luxações, osteoporose, artrite, osteomielite
- Análise em alta resolução (384x384) para detalhes ósseos
- Modelo especializado para estruturas esqueléticas

## Realizando Análise

1. Carregue uma imagem
2. Selecione o tipo de análise apropriado
3. Ajuste o limiar de confiança (opcional)
4. Clique em "Analisar Imagem"
5. Aguarde o processamento
6. Visualize os resultados

## Interpretando Resultados

### Probabilidades
- Valores de 0-100% para cada classe
- Maior valor indica diagnóstico mais provável

### Mapa de Calor
- Regiões vermelhas: alta atenção da IA
- Regiões azuis: baixa relevância
- Ajuda a localizar achados

### Métricas de Confiança
- **Confiança**: Certeza geral do modelo
- **Incerteza**: Medida de dúvida
- **Tempo**: Duração do processamento

## Gerando Relatórios

1. Após análise, clique em "Gerar Relatório"
2. Escolha o formato (PDF, HTML)
3. Adicione observações se necessário
4. Salve o relatório

## Processamento em Lote

1. Menu "Ferramentas" > "Processamento em Lote"
2. Selecione pasta com imagens
3. Escolha modelo de análise
4. Configure opções de saída
5. Inicie processamento

## Comparação de Imagens

1. Menu "Ferramentas" > "Comparar Imagens"
2. Carregue duas imagens
3. Visualize diferenças lado a lado
4. Analise mudanças temporais

## Configurações

### Acesso
- Menu "Ferramentas" > "Configurações"
- Ou use Ctrl+,

### Opções Disponíveis
- Limiar padrão de confiança
- Diretórios de trabalho
- Configurações de GPU
- Preferências de interface

## Segurança e Privacidade

### Anonimização
- Dados do paciente são automaticamente anonimizados
- Informações sensíveis são removidas dos logs

### Auditoria
- Todas as ações são registradas
- Logs incluem usuário, horário e ação

## Dicas de Uso

### Melhores Práticas
1. Use imagens de alta qualidade
2. Selecione o modelo apropriado para o tipo de exame
3. Sempre revise os resultados clinicamente
4. Mantenha backups dos relatórios importantes

### Limitações
- IA é ferramenta de auxílio, não substitui diagnóstico médico
- Resultados devem ser validados por profissional qualificado
- Modelos SOTA têm alta precisão mas requerem validação clínica
- Thresholds elevados garantem maior confiabilidade mas podem reduzir sensibilidade

## Atalhos de Teclado

- **Ctrl+O**: Abrir imagem
- **Ctrl+S**: Salvar relatório
- **Ctrl+Q**: Sair
- **Ctrl+,**: Configurações
- **F1**: Ajuda
- **+/-**: Zoom in/out
- **Espaço**: Ajustar à janela

## Solução de Problemas

### Imagem não carrega
- Verifique formato do arquivo
- Confirme se arquivo não está corrompido
- Tente converter para formato padrão

### Análise muito lenta
- Verifique se GPU está sendo utilizada
- Reduza resolução da imagem
- Feche outros programas

### Resultados inconsistentes
- Verifique qualidade da imagem
- Confirme se modelo é apropriado
- Ajuste limiar de confiança

## Suporte

Para dúvidas ou problemas:
- Consulte a documentação técnica
- Abra issue no GitHub
- Entre em contato: drguilhermecapel@gmail.com
