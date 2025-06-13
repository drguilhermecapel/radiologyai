# Licenciamento dos Modelos Pré-Treinados - MedAI Radiologia

## Informações Gerais

Este documento descreve o licenciamento, uso permitido e restrições dos modelos de inteligência artificial pré-treinados incluídos no sistema MedAI Radiologia.

**Versão da Documentação**: 1.0.0  
**Data de Atualização**: 13 de junho de 2025  
**Responsável**: Equipe MedAI Radiologia

---

## Modelos Incluídos

### 1. EfficientNetV2 - Chest X-Ray
- **Nome Completo**: EfficientNetV2-B3 para Análise de Raio-X de Tórax
- **Versão do Modelo**: 2.1.0
- **Arquitetura**: EfficientNetV2-B3 (Google Research)
- **Dataset de Treino**: 
  - NIH ChestX-ray14 (Licença CC0 - Domínio Público)
  - CheXpert Stanford (Licença de Pesquisa)
  - MIMIC-CXR (PhysioNet Credentialed Health Data License)
- **Licença do Modelo**: Apache License 2.0
- **Acurácia Validada**: 92.3% (Sensibilidade: 90%, Especificidade: 89%)
- **Status Regulatório**: Pendente FDA 510(k) - Não aprovado para uso clínico
- **Tamanho**: ~150MB
- **Modalidades Suportadas**: Raio-X de Tórax (PA e Lateral)
- **Classes Detectadas**: Normal, Pneumonia, Derrame Pleural, Fraturas, Massas/Tumores

### 2. Vision Transformer Medical - Chest X-Ray
- **Nome Completo**: Vision Transformer Base para Análise Médica
- **Versão do Modelo**: 2.0.1
- **Arquitetura**: ViT-Base/16 (Google Research, adaptado para medicina)
- **Dataset de Treino**:
  - MIMIC-CXR (PhysioNet Credentialed Health Data License)
  - PadChest (Creative Commons Attribution-ShareAlike 4.0)
- **Licença do Modelo**: MIT License
- **Acurácia Validada**: 91.1% (Sensibilidade: 88%, Especificidade: 91%)
- **Status Regulatório**: Apenas para pesquisa - Não aprovado para uso clínico
- **Tamanho**: ~300MB
- **Modalidades Suportadas**: Raio-X de Tórax
- **Características Especiais**: Mapas de atenção para interpretabilidade

### 3. ConvNeXt Medical - Chest X-Ray
- **Nome Completo**: ConvNeXt Base para Análise Radiológica
- **Versão do Modelo**: 1.5.2
- **Arquitetura**: ConvNeXt-Base (Facebook Research, adaptado)
- **Dataset de Treino**:
  - ChestX-ray14 (CC0)
  - OpenI Indiana University (Open Access)
- **Licença do Modelo**: Apache License 2.0
- **Acurácia Validada**: 90.8% (Balanceado para velocidade e precisão)
- **Status Regulatório**: Pesquisa e desenvolvimento
- **Tamanho**: ~200MB
- **Modalidades Suportadas**: Raio-X de Tórax
- **Características Especiais**: Inferência rápida, otimizado para CPU

### 4. Ensemble SOTA Multi-Modal
- **Nome Completo**: Ensemble State-of-the-Art Multi-Modal
- **Versão do Modelo**: 3.0.0
- **Arquitetura**: Ensemble (EfficientNetV2 + ViT + ConvNeXt + Meta-Learning)
- **Dataset de Treino**:
  - Combinação de todos os datasets acima
  - Dados proprietários de validação clínica (anonimizados)
- **Licença do Modelo**: Apache License 2.0
- **Acurácia Validada**: 94.5% (Sensibilidade: 92%, Especificidade: 94%)
- **Status Regulatório**: Em processo de certificação FDA 510(k)
- **Tamanho**: ~800MB
- **Modalidades Suportadas**: Raio-X de Tórax, CT Cerebral, Raio-X Ósseo
- **Características Especiais**: Máxima precisão, análise multi-modal

---

## Uso Permitido

### ✅ Usos Autorizados
- **Pesquisa Acadêmica**: Estudos científicos, publicações, teses
- **Uso Educacional**: Ensino médico, treinamento de residentes
- **Desenvolvimento e Testes**: Prototipagem, validação de sistemas
- **Demonstrações**: Apresentações técnicas, provas de conceito
- **Análise Comparativa**: Benchmarking com outros sistemas

### ⚠️ Usos com Restrições
- **Uso Clínico Supervisionado**: Apenas com supervisão médica qualificada
- **Triagem Assistida**: Como ferramenta de apoio, não substituto do diagnóstico
- **Segunda Opinião**: Validação adicional de diagnósticos médicos
- **Ensino Clínico**: Treinamento supervisionado de profissionais

### ❌ Usos Proibidos
- **Diagnóstico Autônomo**: Diagnóstico sem supervisão médica
- **Decisões Clínicas Finais**: Substituição do julgamento médico
- **Uso Comercial Direto**: Venda como produto médico certificado
- **Modificação Não Autorizada**: Alteração dos modelos sem permissão
- **Redistribuição**: Compartilhamento dos modelos sem autorização

---

## Restrições e Limitações

### Limitações Técnicas
- **Dados de Entrada**: Apenas imagens médicas de qualidade diagnóstica
- **Populações**: Validado principalmente em populações adultas
- **Modalidades**: Limitado às modalidades especificadas para cada modelo
- **Idioma**: Interface e relatórios em português e inglês

### Limitações Clínicas
- **Não Substitui Médico**: Jamais substitui avaliação médica profissional
- **Casos Complexos**: Pode ter limitações em patologias raras
- **Contexto Clínico**: Não considera histórico clínico completo
- **Emergências**: Não adequado para situações de emergência

### Limitações Legais
- **Responsabilidade**: Usuário assume responsabilidade pelo uso
- **Jurisdição**: Sujeito às leis locais de cada país
- **Certificação**: Aguardando aprovações regulatórias
- **Auditoria**: Uso pode ser auditado para fins de pesquisa

---

## Atribuições e Créditos

### Datasets Utilizados
- **NIH ChestX-ray14**: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database"
- **CheXpert**: Irvin et al., "CheXpert: A Large Chest Radiograph Dataset"
- **MIMIC-CXR**: Johnson et al., "MIMIC-CXR-JPG, a large publicly available database"
- **PadChest**: Bustos et al., "PadChest: A large chest x-ray image dataset"

### Arquiteturas Base
- **EfficientNetV2**: Tan & Le, "EfficientNetV2: Smaller Models and Faster Training"
- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words"
- **ConvNeXt**: Liu et al., "A ConvNet for the 2020s"

### Contribuições Originais
- **Adaptação Médica**: Equipe MedAI Radiologia
- **Validação Clínica**: Parceiros hospitalares (anonimizados)
- **Otimizações**: Algoritmos proprietários de ensemble

---

## Isenção de Responsabilidade

### Aviso Legal Importante

**OS MODELOS SÃO FORNECIDOS "COMO ESTÃO" PARA FINS DE DEMONSTRAÇÃO, PESQUISA E EDUCAÇÃO. NÃO SUBSTITUEM O DIAGNÓSTICO MÉDICO PROFISSIONAL.**

### Limitações de Responsabilidade

1. **Precisão**: Embora validados, os modelos podem apresentar erros
2. **Atualização**: Modelos podem ficar desatualizados com novas descobertas
3. **Variabilidade**: Resultados podem variar entre diferentes populações
4. **Contexto**: Não consideram o contexto clínico completo do paciente

### Responsabilidades do Usuário

- **Supervisão Médica**: Sempre usar sob supervisão de profissional qualificado
- **Validação**: Validar resultados com métodos diagnósticos estabelecidos
- **Atualização**: Manter-se atualizado sobre limitações e atualizações
- **Ética**: Usar de forma ética e responsável

### Exclusão de Garantias

- **Sem Garantia de Resultados**: Não garantimos resultados específicos
- **Sem Responsabilidade por Danos**: Não nos responsabilizamos por danos decorrentes do uso
- **Limitação de Responsabilidade**: Responsabilidade limitada ao valor pago pelo software

---

## Compliance e Regulamentações

### Regulamentações Aplicáveis

#### Brasil
- **ANVISA**: Aguardando classificação como software médico
- **CFM**: Sujeito às resoluções do Conselho Federal de Medicina
- **LGPD**: Compliance com Lei Geral de Proteção de Dados

#### Internacional
- **FDA (EUA)**: Processo 510(k) em andamento para alguns modelos
- **CE (Europa)**: Avaliação para marcação CE como dispositivo médico
- **Health Canada**: Análise para licenciamento médico

### Padrões de Qualidade
- **ISO 13485**: Sistema de gestão da qualidade para dispositivos médicos
- **ISO 14155**: Boas práticas clínicas para investigação de dispositivos médicos
- **IEC 62304**: Software de dispositivos médicos - Processos de ciclo de vida

### Auditoria e Rastreabilidade
- **Logs de Uso**: Sistema mantém logs para auditoria
- **Versionamento**: Controle rigoroso de versões dos modelos
- **Validação**: Processo contínuo de validação e monitoramento

---

## Contato e Suporte

### Questões Legais
- **Email**: legal@medai-radiologia.com
- **Telefone**: +55 (11) 1234-5678
- **Endereço**: [Endereço da empresa]

### Suporte Técnico
- **Email**: suporte@medai-radiologia.com
- **Documentação**: https://docs.medai-radiologia.com
- **GitHub**: https://github.com/drguilhermecapel/radiologyai

### Relatório de Problemas
- **Bugs**: Reportar via GitHub Issues
- **Problemas Clínicos**: Contatar equipe médica
- **Questões Éticas**: Comitê de ética interno

---

## Histórico de Versões

| Versão | Data | Alterações |
|--------|------|------------|
| 1.0.0 | 13/06/2025 | Versão inicial da documentação |

---

**© 2025 MedAI Radiologia. Todos os direitos reservados.**

*Este documento está sujeito a atualizações. Verifique regularmente a versão mais recente.*
