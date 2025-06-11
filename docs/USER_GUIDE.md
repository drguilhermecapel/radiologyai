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

## API REST - Integração com Sistemas Hospitalares

### Visão Geral da API
O MedAI Radiologia oferece uma API REST completa baseada em FastAPI, permitindo integração perfeita com sistemas hospitalares, PACS, e aplicações de terceiros.

**URL Base**: `http://localhost:8000/api/v1/`
**Documentação**: `http://localhost:8000/docs` (Swagger UI)

### Autenticação
```python
# Exemplo de autenticação (se configurada)
headers = {
    'Authorization': 'Bearer YOUR_JWT_TOKEN',
    'Content-Type': 'application/json'
}
```

### Endpoints da API

#### POST /api/v1/analyze
Realiza análise de imagem médica usando IA de última geração.

**Parâmetros:**
- `file`: Arquivo de imagem (DICOM, PNG, JPEG)
- `model`: Modelo a usar (opcional, padrão: "ensemble")
- `include_explanation`: Incluir explicabilidade (opcional, padrão: false)
- `clinical_validation`: Ativar validação clínica (opcional, padrão: true)

**Exemplo de Requisição:**
```python
import requests

with open('chest_xray.dcm', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/analyze',
        files={'file': ('chest_xray.dcm', f, 'application/dicom')},
        data={
            'model': 'ensemble',
            'include_explanation': 'true',
            'clinical_validation': 'true'
        }
    )

result = response.json()
```

**Exemplo de Resposta:**
```json
{
  "success": true,
  "analysis": {
    "predicted_class": "pneumonia",
    "confidence": 0.92,
    "findings": [
      "Consolidação em lobo inferior direito",
      "Aumento da opacidade pulmonar"
    ],
    "recommendations": [
      "Correlação clínica recomendada",
      "Acompanhamento em 48-72 horas"
    ]
  },
  "clinical_metrics": {
    "performance_metrics": {
      "sensitivity": 0.95,
      "specificity": 0.91,
      "accuracy": 0.93
    },
    "clinical_validation": {
      "approved_for_clinical_use": true,
      "confidence_threshold": 0.85
    }
  },
  "explanation": {
    "method": "gradcam",
    "attention_regions": ["right_lower_lobe"],
    "heatmap_available": true
  },
  "processing_time": 1.2,
  "model_used": "EfficientNetV2",
  "timestamp": "2025-06-11T23:37:00Z"
}
```

#### GET /api/v1/models
Lista todos os modelos de IA disponíveis.

**Exemplo de Resposta:**
```json
[
  {
    "name": "ensemble",
    "description": "Modelo ensemble com múltiplas arquiteturas",
    "modalities": ["chest_xray", "brain_ct", "bone_xray"],
    "accuracy": 0.95,
    "version": "1.1.0"
  },
  {
    "name": "efficientnetv2",
    "description": "EfficientNetV2-L para análise de raio-X",
    "modalities": ["chest_xray"],
    "accuracy": 0.92,
    "version": "1.0.0"
  }
]
```

#### GET /api/v1/health
Verifica o status do sistema.

**Exemplo de Resposta:**
```json
{
  "status": "ok",
  "version": "1.1.0",
  "models_loaded": 5,
  "uptime": "2h 15m 30s",
  "memory_usage": "2.1GB",
  "gpu_available": true
}
```

#### GET /api/v1/metrics
Retorna métricas de performance do sistema.

**Exemplo de Resposta:**
```json
{
  "total_predictions": 1247,
  "average_processing_time": 1.8,
  "model_accuracy": {
    "ensemble": 0.95,
    "efficientnetv2": 0.92,
    "vision_transformer": 0.91,
    "convnext": 0.90,
    "resnet": 0.88
  },
  "system_performance": {
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "gpu_usage": 82.1
  }
}
```

### Integração com Sistemas Hospitalares

#### PACS Integration
```python
# Exemplo de integração com PACS
def analyze_pacs_study(study_uid, series_uid):
    # Recuperar imagem do PACS
    dicom_data = pacs_client.retrieve_image(study_uid, series_uid)
    
    # Enviar para análise
    response = requests.post(
        'http://medai-server:8000/api/v1/analyze',
        files={'file': ('study.dcm', dicom_data)},
        data={'model': 'ensemble'}
    )
    
    # Processar resultado
    result = response.json()
    return result['analysis']
```

#### HL7 Integration
```python
# Exemplo de integração HL7
def send_hl7_result(patient_id, analysis_result):
    hl7_message = create_oru_message(
        patient_id=patient_id,
        observation_value=analysis_result['predicted_class'],
        confidence=analysis_result['confidence']
    )
    
    hl7_client.send_message(hl7_message)
```

## Tipos de Análise com IA de Última Geração

### Raio-X Torácico (EfficientNetV2 + Vision Transformer)
- **Precisão**: 92% de acurácia com modelos SOTA
- Detecta pneumonia, COVID-19, tuberculose, derrame pleural
- Identifica cardiomegalia com alta confiabilidade
- Avalia normalidade pulmonar com threshold otimizado (85%)
- **API Endpoint**: `model=efficientnetv2` ou `model=ensemble`

### CT Cerebral (Vision Transformer Especializado)
- **Precisão**: 91% de acurácia para diagnósticos críticos
- Detecta hemorragias, isquemias, tumores, edemas, hidrocefalia
- Análise baseada em atenção para detalhes neurológicos
- Threshold elevado (90%) para máxima confiabilidade
- **API Endpoint**: `model=vision_transformer` ou `model=ensemble`

### Detecção de Fraturas (ConvNeXt Otimizado)
- **Precisão**: 90% de acurácia para patologias ósseas
- Detecta fraturas, luxações, osteoporose, artrite, osteomielite
- Análise em alta resolução (384x384) para detalhes ósseos
- Modelo especializado para estruturas esqueléticas
- **API Endpoint**: `model=convnext` ou `model=ensemble`

### Ultrassom e Mamografia (Suporte Experimental)
- **Modalidades Novas**: US (Ultrassom), MG (Mamografia)
- Processamento especializado para cada modalidade
- Integração com modelos ensemble
- **API Endpoint**: `model=ensemble` (detecção automática de modalidade)

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
