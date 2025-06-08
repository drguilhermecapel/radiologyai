# RadiologyAI

Sistema de análise de imagens radiológicas médicas por inteligência artificial.

## Descrição

Este repositório contém um programa Windows standalone para interpretação de exames radiológicos usando IA. O programa aceita todos os tipos de arquivos de imagem e fornece análise automatizada para auxiliar profissionais de saúde.

## Funcionalidades

- Análise de imagens radiológicas usando modelos de IA
- Suporte para múltiplos formatos de imagem (DICOM, PNG, JPEG, etc.)
- Interface gráfica intuitiva
- Tema futurista inspirado em radiologia e inteligência artificial
- Geração de relatórios
- Processamento em lote
- Visualização avançada com mapas de calor

## Estrutura do Projeto

- `src/` - Código fonte principal
- `models/` - Modelos de IA treinados
- `data/` - Dados de exemplo e configurações
- `docs/` - Documentação
- `build/` - Scripts de build para Windows

## Instalação

O programa será distribuído como um executável Windows standalone (.exe) que não requer instalação adicional.

## Uso

1. Execute o arquivo .exe
2. Carregue uma imagem radiológica
3. Selecione o tipo de análise desejada
4. Visualize os resultados e gere relatórios

## Desenvolvimento

Este projeto foi desenvolvido usando Python com as seguintes tecnologias:
- PyQt5 para interface gráfica
- TensorFlow/Keras para modelos de IA
- PyDICOM para processamento de imagens médicas
- PyInstaller para criação do executável Windows
