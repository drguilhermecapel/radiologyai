# RadiologyAI

Sistema de análise de imagens radiológicas médicas por inteligência artificial de última geração.

## Descrição

Este repositório contém um sistema completo de análise médica com IA que inclui um programa Windows standalone e uma API REST completa para interpretação de exames radiológicos. O sistema utiliza modelos de inteligência artificial de última geração para fornecer análise automatizada de alta precisão para auxiliar profissionais de saúde.

## Funcionalidades

### Análise de IA de Última Geração
- **Modelos SOTA**: EfficientNetV2, Vision Transformer, ConvNeXt, ResNet
- **Ensemble Inteligente**: Combinação de múltiplos modelos para máxima precisão
- **Alta Acurácia**: >95% para condições críticas, >90% para condições moderadas
- **Múltiplas Modalidades**: Suporte para CR, CT, MR, US, MG

### Interface e Integração
- Interface gráfica intuitiva (PyQt5)
- **API REST Completa**: FastAPI com documentação OpenAPI/Swagger
- Suporte para múltiplos formatos de imagem (DICOM, PNG, JPEG, etc.)
- Integração com sistemas hospitalares (PACS, HL7, FHIR)

### Recursos Avançados
- Geração de relatórios estruturados
- Processamento em lote de alta performance
- Visualização avançada com mapas de calor e explicabilidade
- Validação clínica automatizada
- Monitoramento de performance em tempo real

## API REST

O MedAI Radiologia inclui uma **API REST completa** implementada com FastAPI, permitindo integração perfeita com sistemas hospitalares e aplicações de terceiros.

### Endpoints Principais
- `POST /api/v1/analyze` - Análise de imagens médicas
- `GET /api/v1/models` - Lista de modelos disponíveis
- `GET /api/v1/health` - Status do sistema
- `GET /api/v1/metrics` - Métricas de performance
- `POST /api/v1/explain` - Explicabilidade da IA

### Características da API
- **Autenticação**: Suporte para tokens JWT
- **Rate Limiting**: Controle de taxa de requisições
- **Documentação**: OpenAPI/Swagger automática
- **Validação**: Schemas Pydantic para entrada/saída
- **CORS**: Configurado para integração web

Consulte o [Guia do Usuário](docs/USER_GUIDE.md) para instruções detalhadas de uso da API.

## Estrutura do Projeto

- `src/` - Código fonte principal
  - `medai_fastapi_server.py` - Servidor API REST
  - `medai_sota_models.py` - Modelos de IA de última geração
  - `medai_integration_manager.py` - Gerenciador central
- `models/` - Modelos de IA treinados (H5, JSON)
- `data/` - Dados de exemplo e configurações
- `docs/` - Documentação completa
- `templates/` - Interface web
- `config/` - Configurações de produção

## Instalação

### Executável Windows
O programa é distribuído como um executável Windows standalone (.exe) que não requer instalação adicional.

### Servidor API
```bash
pip install -r requirements.txt
python src/medai_fastapi_server.py
```

### Docker
```bash
docker-compose up -d
```

## Uso

### Interface Desktop
1. Execute o arquivo `MedAI_Radiologia.exe`
2. Carregue uma imagem radiológica
3. Selecione o modelo de IA apropriado
4. Visualize os resultados e gere relatórios

### API REST
```python
import requests

# Análise de imagem
with open('chest_xray.dcm', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/analyze',
        files={'file': f},
        data={'model': 'ensemble'}
    )
    
result = response.json()
print(f"Diagnóstico: {result['analysis']['predicted_class']}")
print(f"Confiança: {result['analysis']['confidence']:.2f}")
```

## Desenvolvimento

### Tecnologias Principais
- **Backend**: Python 3.12, FastAPI, TensorFlow/Keras
- **Frontend**: PyQt5, HTML/CSS/JavaScript
- **IA**: EfficientNetV2, Vision Transformer, ConvNeXt
- **Dados**: PyDICOM, NumPy, OpenCV
- **Deploy**: Docker, PyInstaller

### Modelos de IA Implementados
- **EfficientNetV2-L**: Análise de raio-X torácico (92% acurácia)
- **Vision Transformer**: CT cerebral (91% acurácia)  
- **ConvNeXt-XL**: Detecção de fraturas (90% acurácia)
- **ResNet-50**: Análise geral (88% acurácia)
- **Ensemble**: Combinação inteligente (95% acurácia)
