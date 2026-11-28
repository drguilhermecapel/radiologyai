#!/usr/bin/env python3
"""Ingere a evidência publicada conferida na fonte para ``evidence/published/``.

Entrada: os arquivos ``*.verified.md`` da pesquisa de evidência publicada (por
padrão em ``/root/medai-cache/research``). Cada um termina com um bloco YAML
``published_evidence`` cujas entradas foram re-verificadas adversarialmente na
fonte: a citação (``quote``) contém literalmente o valor.

Saída (determinística: mesmas entradas, mesmos bytes):

* ``evidence/published/<grupo>.yaml`` - registros
  :class:`~radiologyai.catalog.schema.EvidenceRecord` anexados a um trio
  (condição, modelo, rótulo) **de um modelo que rodamos, com os mesmos pesos**.
  ``passes_protocol`` é decidido mecanicamente pelas barras de publicação do
  protocolo pré-registrado; registro reprovado declara ``failure_reasons``.
* ``evidence/published/context.yaml`` - evidência que NÃO pode ser anexada a
  um modelo nosso (modelo que não rodamos, pesos ou versão diferentes, limiar
  cuja segmentação de base não tem validação documentada para os nossos pesos,
  qualidade de rótulo de dataset). Fica como contexto, com o motivo.
* ``evidence/published/conditions_proposed.yaml`` - entradas ``Condition`` para
  toda condição referenciada; o responsável pelo catálogo as funde em
  ``src/radiologyai/catalog/conditions.yaml``.

Os dois últimos arquivos têm ``records: []`` na raiz porque
``radiologyai.evidence.store.load_records`` lê todo YAML de
``evidence/published/``; com a lista vazia eles são lidos sem efeito.

Uso:
    python scripts/evidence_ingest_published.py            # (re)grava
    python scripts/evidence_ingest_published.py --check    # falha se desatualizado
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiologyai.catalog.schema import Protocol

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESEARCH_DIR = Path("/root/medai-cache/research")
DEFAULT_OUT = REPO / "evidence" / "published"
MANIFESTS_DIR = REPO / "src" / "radiologyai" / "analysis" / "manifests"
GENERATOR = "scripts/evidence_ingest_published.py"

# Arquivo de pesquisa -> nome do arquivo de saída.
INPUTS: tuple[tuple[str, str], ...] = (
    ("evidence-cxr.verified.md", "cxr"),
    ("evidence-ct-mr.verified.md", "ct_mr"),
    ("evidence-echo-mg-us.verified.md", "echo_mg_us"),
)

_YAML_BLOCK = re.compile(r"```yaml\n(published_evidence:.*?)```", re.S)


class IngestError(Exception):
    """Entrada inesperada: a ingestão para em vez de adivinhar."""


# --------------------------------------------------------------------------- #
# Condições propostas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CondSpec:
    id: str
    name_pt: str
    name_en: str
    exam_types: tuple[str, ...]
    system: str
    icd10: str | None
    description_pt: str


_SUFFIX = " Achado sugerido para confirmação médica, nunca diagnóstico."

CONDITIONS: tuple[CondSpec, ...] = (
    # ---- radiografia de tórax
    CondSpec(
        "cxr.atelectasis",
        "Atelectasia",
        "Atelectasis",
        ("cxr",),
        "pulmão",
        "J98.1",
        "Perda de volume pulmonar sugerida na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.cardiomegaly",
        "Cardiomegalia (silhueta cardíaca aumentada)",
        "Cardiomegaly",
        ("cxr",),
        "coração",
        "I51.7",
        "Silhueta cardíaca aumentada na radiografia de tórax; não equivale a disfunção.",
    ),
    CondSpec(
        "cxr.consolidation",
        "Consolidação pulmonar",
        "Consolidation",
        ("cxr",),
        "pulmão",
        None,
        "Opacidade com preenchimento alveolar sugerida na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.pulmonary_edema",
        "Edema pulmonar",
        "Pulmonary edema",
        ("cxr",),
        "pulmão",
        "J81",
        "Padrão radiográfico sugestivo de edema pulmonar.",
    ),
    CondSpec(
        "cxr.pleural_effusion",
        "Derrame pleural",
        "Pleural effusion",
        ("cxr",),
        "pleura",
        "J90",
        "Líquido pleural sugerido na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.emphysema",
        "Enfisema",
        "Emphysema",
        ("cxr",),
        "pulmão",
        "J43",
        "Sinais radiográficos sugestivos de enfisema pulmonar.",
    ),
    CondSpec(
        "cxr.pulmonary_fibrosis",
        "Fibrose pulmonar",
        "Pulmonary fibrosis",
        ("cxr",),
        "pulmão",
        "J84.1",
        "Padrão radiográfico sugestivo de fibrose pulmonar.",
    ),
    CondSpec(
        "cxr.hernia",
        "Hérnia diafragmática ou de hiato",
        "Hernia (hiatal/diaphragmatic)",
        ("cxr",),
        "mediastino",
        "K44",
        "Imagem sugestiva de hérnia de hiato ou diafragmática na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.infiltration",
        "Infiltrado pulmonar (opacidade inespecífica)",
        "Infiltration",
        ("cxr",),
        "pulmão",
        None,
        "Opacidade pulmonar inespecífica (rótulo do NIH).",
    ),
    CondSpec(
        "cxr.lung_mass",
        "Massa pulmonar",
        "Lung mass",
        ("cxr",),
        "pulmão",
        None,
        "Lesão pulmonar com mais de 3 cm sugerida na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.lung_nodule",
        "Nódulo pulmonar",
        "Lung nodule",
        ("cxr",),
        "pulmão",
        None,
        "Nódulo pulmonar sugerido na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.pleural_thickening",
        "Espessamento pleural",
        "Pleural thickening",
        ("cxr",),
        "pleura",
        None,
        "Espessamento pleural sugerido na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.pneumonia",
        "Pneumonia (padrão radiográfico)",
        "Pneumonia",
        ("cxr",),
        "pulmão",
        "J18",
        "Padrão radiográfico compatível com pneumonia; não confirma etiologia.",
    ),
    CondSpec(
        "cxr.pneumothorax",
        "Pneumotórax",
        "Pneumothorax",
        ("cxr",),
        "pleura",
        "J93",
        "Ar no espaço pleural sugerido na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.lung_opacity",
        "Opacidade pulmonar",
        "Lung opacity",
        ("cxr",),
        "pulmão",
        None,
        "Opacidade pulmonar de qualquer natureza na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.fracture",
        "Fratura (arcos costais, clavícula ou úmero) na radiografia de tórax",
        "Fracture on chest radiograph",
        ("cxr",),
        "esqueleto",
        None,
        "Fratura óssea visível na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.enlarged_cardiomediastinum",
        "Alargamento da silhueta cardiomediastinal",
        "Enlarged cardiomediastinum",
        ("cxr",),
        "mediastino",
        None,
        "Silhueta cardiomediastinal alargada na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.lung_lesion",
        "Lesão pulmonar",
        "Lung lesion",
        ("cxr",),
        "pulmão",
        None,
        "Lesão pulmonar focal (nódulo ou massa) na radiografia de tórax.",
    ),
    CondSpec(
        "cxr.tuberculosis",
        "Tuberculose pulmonar (padrão radiográfico)",
        "Pulmonary tuberculosis (radiographic pattern)",
        ("cxr",),
        "pulmão",
        None,
        "Padrão radiográfico sugestivo de tuberculose; não substitui teste bacteriológico.",
    ),
    # ---- TC
    CondSpec(
        "ct.pleural_effusion",
        "Derrame pleural (TC)",
        "Pleural effusion (CT)",
        ("ct_chest",),
        "pleura",
        "J90",
        "Líquido pleural segmentado na TC de tórax.",
    ),
    CondSpec(
        "ct.pericardial_effusion",
        "Derrame pericárdico (TC)",
        "Pericardial effusion (CT)",
        ("ct_chest",),
        "coração",
        "I31.3",
        "Líquido pericárdico segmentado na TC de tórax.",
    ),
    CondSpec(
        "ct.focal_liver_lesion",
        "Lesão hepática focal (TC)",
        "Focal liver lesion (CT)",
        ("ct_abdomen",),
        "fígado",
        None,
        "Lesão focal hepática segmentada na TC.",
    ),
    CondSpec(
        "ct.osteoporosis_l1",
        "Baixa atenuação trabecular de L1 (osteoporose provável)",
        "Low L1 trabecular attenuation (probable osteoporosis)",
        ("ct_abdomen", "ct_chest"),
        "esqueleto",
        None,
        "Atenuação trabecular de L1 abaixo de limiar publicado; não substitui DXA.",
    ),
    CondSpec(
        "ct.hepatic_steatosis",
        "Esteatose hepática (TC)",
        "Hepatic steatosis (CT)",
        ("ct_abdomen",),
        "fígado",
        "K76.0",
        "Atenuação hepática compatível com esteatose ao menos moderada.",
    ),
    CondSpec(
        "ct.emphysema",
        "Enfisema (quantificação por TC)",
        "Emphysema (CT quantification)",
        ("ct_chest",),
        "pulmão",
        "J43",
        "Área pulmonar de baixa atenuação acima de limiar.",
    ),
    CondSpec(
        "ct.abdominal_aortic_aneurysm",
        "Aneurisma da aorta abdominal",
        "Abdominal aortic aneurysm",
        ("ct_abdomen",),
        "vascular",
        "I71.4",
        "Diâmetro da aorta abdominal acima do limiar de diretriz.",
    ),
    CondSpec(
        "ct.ascending_aorta_dilation",
        "Dilatação ou aneurisma da aorta ascendente",
        "Ascending aorta dilation or aneurysm",
        ("ct_chest",),
        "vascular",
        None,
        "Diâmetro da raiz ou da aorta ascendente acima do limiar de diretriz.",
    ),
    CondSpec(
        "ct.splenomegaly",
        "Esplenomegalia (volumetria por TC)",
        "Splenomegaly (CT volume)",
        ("ct_abdomen",),
        "baço",
        "R16.1",
        "Volume esplênico acima de limiar publicado.",
    ),
    CondSpec(
        "ct.hepatomegaly",
        "Hepatomegalia (volumetria por TC)",
        "Hepatomegaly (CT volume)",
        ("ct_abdomen",),
        "fígado",
        "R16.0",
        "Volume hepático acima de limiar publicado.",
    ),
    CondSpec(
        "ct.ventriculomegaly",
        "Ventriculomegalia (índice de Evans)",
        "Ventriculomegaly (Evans index)",
        ("ct_head",),
        "sistema nervoso",
        None,
        "Índice de Evans acima de 0,3; não distingue hidrocefalia de atrofia.",
    ),
    CondSpec(
        "ct.vertebral_compression_fracture",
        "Fratura vertebral por compressão (TC)",
        "Vertebral compression fracture (CT)",
        ("ct_abdomen", "ct_chest"),
        "esqueleto",
        None,
        "Perda de altura do corpo vertebral sugerida na reformatação sagital.",
    ),
    # ---- RM
    CondSpec(
        "mr.glioma_segmentation",
        "Glioma: sub-regiões tumorais (RM)",
        "Glioma sub-region segmentation (MR)",
        ("mr_brain",),
        "sistema nervoso",
        None,
        "Segmentação de sub-regiões de glioma já conhecido; não é detector de tumor.",
    ),
    CondSpec(
        "mr.white_matter_lesions",
        "Lesões de substância branca (padrão de esclerose múltipla)",
        "White-matter lesions (multiple sclerosis pattern)",
        ("mr_brain",),
        "sistema nervoso",
        None,
        "Carga de lesões hiperintensas em FLAIR; não diagnostica esclerose múltipla.",
    ),
    CondSpec(
        "mr.focal_liver_lesion",
        "Lesão hepática focal (RM)",
        "Focal liver lesion (MR)",
        ("mr_body",),
        "fígado",
        None,
        "Lesão focal hepática segmentada na RM.",
    ),
    # ---- mamografia
    CondSpec(
        "mg.breast_cancer",
        "Câncer de mama (achado suspeito na mamografia)",
        "Breast cancer (suspicious mammographic finding)",
        ("mg",),
        "mama",
        "C50",
        "Escore de suspeição de malignidade na mamografia de rastreamento.",
    ),
    CondSpec(
        "mg.breast_density",
        "Densidade mamária (categoria BI-RADS A-D)",
        "Breast density (BI-RADS category A-D)",
        ("mg",),
        "mama",
        None,
        "Categoria de densidade do parênquima mamário.",
    ),
    # ---- ultrassom
    CondSpec(
        "us.hepatic_steatosis",
        "Esteatose hepática (ultrassom)",
        "Hepatic steatosis (ultrasound)",
        ("us_abdomen",),
        "fígado",
        "K76.0",
        "Padrão ultrassonográfico sugestivo de esteatose hepática.",
    ),
    # ---- ecocardiograma
    CondSpec(
        "echo.reduced_lvef",
        "Fração de ejeção do VE reduzida",
        "Reduced left ventricular ejection fraction",
        ("echo",),
        "coração",
        None,
        "Estimativa de fração de ejeção do ventrículo esquerdo abaixo do normal.",
    ),
    CondSpec(
        "echo.aortic_stenosis",
        "Estenose aórtica (moderada ou importante)",
        "Aortic stenosis (moderate or severe)",
        ("echo",),
        "coração",
        "I35.0",
        "Estenose valvar aórtica moderada ou importante.",
    ),
    CondSpec(
        "echo.aortic_regurgitation",
        "Insuficiência aórtica (moderada ou importante)",
        "Aortic regurgitation (moderate or severe)",
        ("echo",),
        "coração",
        "I35.1",
        "Regurgitação valvar aórtica moderada ou importante.",
    ),
    CondSpec(
        "echo.mitral_regurgitation",
        "Insuficiência mitral (moderada ou importante)",
        "Mitral regurgitation (moderate or severe)",
        ("echo",),
        "coração",
        "I34.0",
        "Regurgitação valvar mitral moderada ou importante.",
    ),
    CondSpec(
        "echo.tricuspid_regurgitation",
        "Insuficiência tricúspide (moderada ou importante)",
        "Tricuspid regurgitation (moderate or severe)",
        ("echo",),
        "coração",
        "I36.1",
        "Regurgitação valvar tricúspide moderada ou importante.",
    ),
    CondSpec(
        "echo.mitral_stenosis",
        "Estenose mitral",
        "Mitral stenosis",
        ("echo",),
        "coração",
        None,
        "Estenose valvar mitral.",
    ),
    CondSpec(
        "echo.pericardial_effusion",
        "Derrame pericárdico (ecocardiograma)",
        "Pericardial effusion (echocardiography)",
        ("echo",),
        "coração",
        "I31.3",
        "Derrame pericárdico moderado ou importante ao ecocardiograma.",
    ),
    CondSpec(
        "echo.rv_systolic_dysfunction",
        "Disfunção sistólica do VD",
        "Right ventricular systolic dysfunction",
        ("echo",),
        "coração",
        None,
        "Função sistólica do ventrículo direito reduzida.",
    ),
    CondSpec(
        "echo.rv_dilation",
        "Dilatação do VD",
        "Right ventricular dilation",
        ("echo",),
        "coração",
        None,
        "Ventrículo direito dilatado.",
    ),
    CondSpec(
        "echo.la_dilation",
        "Dilatação do átrio esquerdo",
        "Left atrial dilation",
        ("echo",),
        "coração",
        None,
        "Átrio esquerdo dilatado.",
    ),
    CondSpec(
        "echo.ra_dilation",
        "Dilatação do átrio direito",
        "Right atrial dilation",
        ("echo",),
        "coração",
        None,
        "Átrio direito dilatado.",
    ),
    CondSpec(
        "echo.bicuspid_aortic_valve",
        "Valva aórtica bicúspide",
        "Bicuspid aortic valve",
        ("echo",),
        "coração",
        "Q23.1",
        "Valva aórtica com dois folhetos.",
    ),
    CondSpec(
        "echo.mitral_annular_calcification",
        "Calcificação do anel mitral",
        "Mitral annular calcification",
        ("echo",),
        "coração",
        None,
        "Calcificação do anel valvar mitral.",
    ),
    CondSpec(
        "echo.aortic_root_dilation",
        "Dilatação da raiz aórtica (ecocardiograma)",
        "Aortic root dilation (echocardiography)",
        ("echo",),
        "vascular",
        None,
        "Raiz da aorta dilatada ao ecocardiograma.",
    ),
    CondSpec(
        "echo.dilated_ivc",
        "Veia cava inferior dilatada",
        "Dilated inferior vena cava",
        ("echo",),
        "vascular",
        None,
        "Veia cava inferior dilatada ao ecocardiograma.",
    ),
    CondSpec(
        "echo.diastolic_dysfunction",
        "Disfunção diastólica do VE",
        "Left ventricular diastolic dysfunction",
        ("echo",),
        "coração",
        None,
        "Disfunção diastólica do ventrículo esquerdo.",
    ),
    CondSpec(
        "echo.elevated_pa_pressure",
        "Pressão arterial pulmonar elevada (estimativa)",
        "Elevated pulmonary artery pressure (estimate)",
        ("echo",),
        "coração",
        None,
        "Estimativa ecocardiográfica de pressão sistólica da artéria pulmonar.",
    ),
)
CONDITION_IDS = frozenset(c.id for c in CONDITIONS)

# --------------------------------------------------------------------------- #
# Decisão de anexação: o modelo publicado é um modelo que rodamos, com os mesmos pesos?
# --------------------------------------------------------------------------- #
# reason_code -> texto (pt-BR) do motivo de NÃO anexar.
CONTEXT_REASONS: dict[str, str] = {
    "modelo_nao_executado": "modelo que não rodamos; evidência de outro modelo não se transfere",
    "pesos_diferentes": "pesos diferentes dos que rodamos; evidência não se transfere",
    "versao_diferente": "versão/resolução dos pesos diferente da que rodamos",
    "pesos_nao_confirmados": "identidade dos pesos publicados com os nossos não confirmada",
    "pesos_nao_disponiveis": "os pesos desta tarefa não estão entre os que baixamos/rodamos",
    "limiar_sem_validacao_da_segmentacao": (
        "limiar/método sobre medida cuja segmentação de base não tem validação publicada "
        "para os pesos que rodamos; vira guideline_threshold só com essa validação documentada"
    ),
    "nao_e_evidencia_de_modelo": "caracteriza dataset ou rótulo, não um modelo",
    "nao_corresponde_a_rotulo": "métrica agregada que não corresponde a um rótulo do modelo",
    "saida_nao_exposta": (
        "a saída publicada não é exposta pelo manifesto do modelo que rodamos "
        "(sem rótulo correspondente em analysis/manifests)"
    ),
}


@dataclass(frozen=True)
class ModelRule:
    pattern: str  # regex contra model_or_method
    model_id: str | None  # None -> contexto
    reason_code: str | None = None
    caveat: str | None = None


_XRV_CAVEAT = (
    "Fonte: BENCHMARKS.md do TorchXRayVision (repositório, não revisado por pares), sem IC, "
    "sem n e sem contagem de positivos; split 80/20 por paciente do próprio repositório."
)
_TS_TASK_CAVEAT = (
    "Identidade dos pesos: o README do TotalSegmentator manda citar este artigo para a tarefa; "
    "não há hash publicado que ligue o artigo aos pesos da tarefa baixados."
)

MODEL_RULES: tuple[ModelRule, ...] = (
    # ---- RX de tórax
    ModelRule(
        r"^TorchXRayVision XRV-ResNet-resnet50-res512-all$",
        "xrv-resnet50-res512-all",
        caveat=_XRV_CAVEAT,
    ),
    ModelRule(
        r"^TorchXRayVision XRV-DenseNet121-densenet121-res224-all$",
        "xrv-densenet121-res224-all",
        caveat=_XRV_CAVEAT,
    ),
    ModelRule(
        r"^TorchXRayVision (jfhealthcare|CheXpert)-DenseNet121", None, "modelo_nao_executado"
    ),
    ModelRule(
        r"^CheXzero",
        None,
        "pesos_diferentes",
        caveat="Artigo avalia ensemble dos 10 melhores checkpoints; rodamos 2 checkpoints.",
    ),
    ModelRule(
        r"^ChestX-Det pre-trained PSPNet",
        None,
        "versao_diferente",
        caveat="Artigo descreve modelo de 5 partes; os pesos distribuídos têm 14 saídas.",
    ),
    ModelRule(
        r"(?i)^(Cardiothoracic ratio|CTR)\b|^axial CT cardiothoracic",
        None,
        "limiar_sem_validacao_da_segmentacao",
    ),
    ModelRule(r"(?i)CTR\b|AlbuNet|Soft-voting ensemble", None, "modelo_nao_executado"),
    ModelRule(
        r"\bCAD|qXR|Lunit|TBNet|^EfficientNetV2|Faster R-CNN|stacked ensemble|AlexNet|"
        r"DLAD|Deep-learning models, mean|Google deep-learning",
        None,
        "modelo_nao_executado",
    ),
    ModelRule(
        r"(?i)target product profile|Montgomery County / Shenzhen|NIH NLP labeller|"
        r"Radiologist visual re-review|Majkowska et al\. label release",
        None,
        "nao_e_evidencia_de_modelo",
    ),
    # ---- TC / RM
    ModelRule(
        r"^TotalSegmentator v1",
        None,
        "versao_diferente",
        caveat="Números da v1 (1204 exames; 1,5 mm/3 mm); rodamos pesos v2 (1559 sujeitos).",
    ),
    ModelRule(
        r"^TotalSegmentator MRI",
        None,
        "versao_diferente",
        caveat="Rodamos só o modelo total_mr de 3 mm (Dataset852); o artigo avalia o "
        "modelo de resolução completa.",
    ),
    # Os artigos (Sexauer 2022; Wilder-Smith 2022) avaliam nnU-Nets de um rótulo só (derrame
    # pleural OU pericárdico; 1 mm; treino 224/316). Os pesos que distribuímos são o
    # Dataset315_thoraxCT (3 rótulos: pulmão, derrame pleural, derrame pericárdico; 1,5 mm;
    # numTraining 600; dataset.json conferido no cache). Modelos diferentes: a evidência
    # publicada não se transfere (revisão adversarial de 2026-11-28).
    ModelRule(
        r"task pleural_pericard_effusion|^nnU-Net \((Sexauer|Wilder-Smith) et al\.\)$",
        None,
        "pesos_diferentes",
    ),
    ModelRule(r"task liver_lesions_mr", None, "pesos_nao_disponiveis"),
    # Nicoli et al. 2025 (DOI 10.1007/s10278-025-01716-y) relata o modelo conjunto MR+CT
    # (750 estudos); os pesos distribuídos são o Dataset591_ct_liver_lesions_842subj (só CT,
    # numTraining 842; dataset.json conferido). Modelos diferentes (revisão de 2026-11-28).
    ModelRule(r"task liver_lesions\)$", None, "pesos_diferentes"),
    ModelRule(r"totalseg_get_phase", None, "modelo_nao_executado"),
    ModelRule(
        r"^MONAI brats_mri_segmentation bundle",
        "monai-brats",
        caveat="Valores de desenvolvedor no split de validação usado para seleção de "
        "modelo (README do bundle no Hugging Face).",
    ),
    ModelRule(r"^LST-AI", "lst-ai"),
    # ---- eco / MG / US
    ModelRule(r"^EchoPrime|^PanEcho", None, "modelo_nao_executado"),
    ModelRule(
        r"^MedSAM2",
        None,
        "pesos_nao_confirmados",
        caveat="Preprint; não confirma que os números são do checkpoint "
        "MedSAM2_US_Heart.pt que rodamos.",
    ),
    ModelRule(r"^MedSAM \(", None, "modelo_nao_executado"),
    ModelRule(
        r"^ianpan/mammoscreen",
        "mammoscreen",
        caveat="Autorrelato do desenvolvedor no cartão do modelo (não revisado por pares).",
    ),
    ModelRule(
        r"^MONAI breast_density_classification",
        "monai-breast-density",
        caveat="Única métrica é 'accuracy' em metadata.json do bundle, sem dataset, n ou "
        "split; o bundle declara 'not to be used for diagnostic purposes'.",
    ),
    ModelRule(r"^Mirai$", None, "modelo_nao_executado"),
    ModelRule(
        r"(?i)Byra|Inception-ResNet|Hepatorenal|U-Net family|Cardiologist vs",
        None,
        "modelo_nao_executado",
    ),
    ModelRule(
        r"(?i)CBIS-DDSM automated|Histopathology \(reference", None, "nao_e_evidencia_de_modelo"
    ),
    # ---- métodos e diretrizes de biomarcadores (TC)
    ModelRule(
        r"(?i)ROI|density mask|%LAA|full-inspiration|MSCT|MDCT|diameter|volumetry|multiplanar|"
        r"segmentation volume|craniocaudal|splenic length|cardiothoracic|frontal horn|"
        r"echo-free|vertebral height|liver segmentation|hepatosplenic|automated L1",
        None,
        "limiar_sem_validacao_da_segmentacao",
    ),
)

# Entradas específicas mandadas para contexto apesar de o modelo ser nosso.
ENTRY_CONTEXT_OVERRIDES: dict[str, tuple[str, str]] = {
    "mammoscreen_cancer_auroc": (
        "pesos_diferentes",
        "valor é a média de 3 splits com pesos distintos; distribuímos só os do split 1 "
        "(registro derivado pub.mammoscreen_cancer_auroc_split1)",
    ),
    "mammoscreen_op1_specificity": ("pesos_diferentes", "média de 3 splits com pesos distintos"),
    "mammoscreen_op2_specificity": ("pesos_diferentes", "média de 3 splits com pesos distintos"),
    "mammoscreen_op3_specificity": ("pesos_diferentes", "média de 3 splits com pesos distintos"),
    "mammoscreen_op1_threshold": ("pesos_diferentes", "limiar médio de 3 splits"),
    "mammoscreen_op2_threshold": ("pesos_diferentes", "limiar médio de 3 splits"),
    "mammoscreen_op3_threshold": ("pesos_diferentes", "limiar médio de 3 splits"),
    "brats_bundle_dice_avg": ("nao_corresponde_a_rotulo", "média das sub-regiões TC/WT/ET"),
}

# Registros derivados de uma entrada conferida: o valor está literalmente na mesma citação.
DERIVED: tuple[dict[str, Any], ...] = (
    {
        "id": "pub.mammoscreen_cancer_auroc_split1",
        "from": "mammoscreen_cancer_auroc",
        "metric": "auroc",
        "printed": "0.9464",
        "caveat": (
            "Derivado da mesma citação conferida ('Split 1: 0.9464'); o cartão diz "
            "'The provided weights are from the first data split'."
        ),
    },
)

# Contagens conferidas na fonte que a pesquisa não trazia (a ingestão punha
# ``n_positive: None`` em todo registro). Cada uma vem de leitura literal do texto
# completo citado em ``url`` (XML do PMC via E-utilities, normalizado só em
# espaços, traços e entidades); ``printed`` tem de aparecer literalmente em
# ``quote``. Nenhuma contagem é somada, estimada ou inferida de IC.
COUNT_SUPPLEMENTS: dict[str, dict[str, Any]] = {}

# Entradas fundidas num único registro (mesmo estudo, mesmo conjunto, mesma citação/URL).
MERGES: dict[str, tuple[str, ...]] = {
    "pub.pleff_internal_detection": (
        "pleff_sens_internal",
        "pleff_spec_internal",
        "pleff_auc_internal",
    ),
    "pub.pleff_nlst_external_detection": ("pleff_sens_nlst_external", "pleff_spec_nlst_external"),
    "pub.pericard_eff_detection": ("pericard_eff_sens", "pericard_eff_spec"),
}

# --------------------------------------------------------------------------- #
# Condição e rótulo dos registros anexados
# --------------------------------------------------------------------------- #
XRV_LABEL_TO_CONDITION: dict[str, str] = {
    "Atelectasis": "cxr.atelectasis",
    "Cardiomegaly": "cxr.cardiomegaly",
    "Consolidation": "cxr.consolidation",
    "Edema": "cxr.pulmonary_edema",
    "Effusion": "cxr.pleural_effusion",
    "Emphysema": "cxr.emphysema",
    "Fibrosis": "cxr.pulmonary_fibrosis",
    "Hernia": "cxr.hernia",
    "Infiltration": "cxr.infiltration",
    "Mass": "cxr.lung_mass",
    "Nodule": "cxr.lung_nodule",
    "Pleural_Thickening": "cxr.pleural_thickening",
    "Pneumonia": "cxr.pneumonia",
    "Pneumothorax": "cxr.pneumothorax",
    "Lung Opacity": "cxr.lung_opacity",
    "Fracture": "cxr.fracture",
    "Enlarged Cardiomediastinum": "cxr.enlarged_cardiomediastinum",
    "Lung Lesion": "cxr.lung_lesion",
}
# Saídas do resnet50-res512-all sem limiar de operação (np.nan em models.py): não treinadas.
XRV_RESNET_UNTRAINED = frozenset({"Lung Lesion", "Enlarged Cardiomediastinum"})

# (regex sobre o id da entrada) -> (condition_id, label) para modelos não-XRV.
ENTRY_TRIPLES: tuple[tuple[str, str, str], ...] = (
    (r"^pleff_|^pub\.pleff_", "ct.pleural_effusion", "pleural_effusion"),
    (r"^pericard_eff_|^pub\.pericard_eff_", "ct.pericardial_effusion", "pericardial_effusion"),
    (r"^ts_liver_lesion_ct_", "ct.focal_liver_lesion", "liver_lesions"),
    (r"^brats_bundle_dice_tc$", "mr.glioma_segmentation", "TC"),
    (r"^brats_bundle_dice_wt$", "mr.glioma_segmentation", "WT"),
    (r"^brats_bundle_dice_et$", "mr.glioma_segmentation", "ET"),
    (r"^lstai_", "mr.white_matter_lesions", "lesion"),
    (r"^mammoscreen_|^pub\.mammoscreen_", "mg.breast_cancer", "cancer"),
    (r"^monai_breast_density_", "mg.breast_density", "density"),
)

# Mapeamento (melhor esforço) do texto da condição de contexto para um id do catálogo.
CONTEXT_CONDITION_RULES: tuple[tuple[str, str], ...] = (
    (r"(?i)^cxr:(atelectasis|lobar atelectasis|laminar atelectasis)$", "cxr.atelectasis"),
    (r"(?i)^cxr:(cardiomegaly.*)$", "cxr.cardiomegaly"),
    (r"(?i)^cxr:consolidation$", "cxr.consolidation"),
    (r"(?i)^cxr:pulmonary edema$", "cxr.pulmonary_edema"),
    (r"(?i)^cxr:pleural effusion$", "cxr.pleural_effusion"),
    (r"(?i)^cxr:emphysema$", "cxr.emphysema"),
    (r"(?i)^cxr:pulmonary fibrosis$", "cxr.pulmonary_fibrosis"),
    (r"(?i)^cxr:hiatal hernia$", "cxr.hernia"),
    (r"(?i)^cxr:infiltrates$", "cxr.infiltration"),
    (r"(?i)^cxr:pulmonary mass$", "cxr.lung_mass"),
    (r"(?i)^cxr:nodule$", "cxr.lung_nodule"),
    (r"(?i)^cxr:pleural thickening$", "cxr.pleural_thickening"),
    (r"(?i)^cxr:pneumonia$", "cxr.pneumonia"),
    (r"(?i)^cxr:pneumothorax$", "cxr.pneumothorax"),
    (r"(?i)^cxr:lung lesion$", "cxr.lung_lesion"),
    (r"(?i)^cxr:(fracture|rib fracture|clavicle fracture|humeral fracture)$", "cxr.fracture"),
    (r"(?i)^cxr:(.*tuberculosis.*|tuberculosis)$", "cxr.tuberculosis"),
    (r"(?i)^ct:.*pleural effusion", "ct.pleural_effusion"),
    (r"(?i)^ct:.*pericardial effusion", "ct.pericardial_effusion"),
    (r"(?i)^ct:.*liver lesion.*\bMR\b|^mr:.*liver lesion", "mr.focal_liver_lesion"),
    (r"(?i)^ct:.*liver lesion", "ct.focal_liver_lesion"),
    (r"(?i)^ct:.*osteoporosis|^ct:.*L1 (trabecular )?attenuation", "ct.osteoporosis_l1"),
    (r"(?i)^ct:.*steatosis|^ct:liver fat", "ct.hepatic_steatosis"),
    (r"(?i)^ct:.*emphysema", "ct.emphysema"),
    (r"(?i)^ct:abdominal aortic aneurysm|^ct:arterial aneurysm", "ct.abdominal_aortic_aneurysm"),
    (r"(?i)^ct:aortic root / ascending", "ct.ascending_aorta_dilation"),
    (r"(?i)^ct:.*splenomegaly", "ct.splenomegaly"),
    (r"(?i)^ct:.*hepatomegaly", "ct.hepatomegaly"),
    (r"(?i)^ct:ventriculomegaly", "ct.ventriculomegaly"),
    (r"(?i)^ct:vertebral compression fracture", "ct.vertebral_compression_fracture"),
    (
        r"(?i)^echo:left ventricular ejection fraction|^echo:moderate or worse left ventricular",
        "echo.reduced_lvef",
    ),
    (r"(?i)^echo:(severe )?aortic stenosis", "echo.aortic_stenosis"),
    (r"(?i)^echo:aortic regurgitation", "echo.aortic_regurgitation"),
    (r"(?i)^echo:mitral regurgitation", "echo.mitral_regurgitation"),
    (r"(?i)^echo:tricuspid regurgitation", "echo.tricuspid_regurgitation"),
    (r"(?i)^echo:mitral stenosis", "echo.mitral_stenosis"),
    (r"(?i)^echo:pericardial effusion", "echo.pericardial_effusion"),
    (r"(?i)^echo:(reduced )?right ventricular systolic", "echo.rv_systolic_dysfunction"),
    (r"(?i)^echo:right ventricular dilation", "echo.rv_dilation"),
    (r"(?i)^echo:left atrial dilation", "echo.la_dilation"),
    (r"(?i)^echo:right atrial dilation", "echo.ra_dilation"),
    (r"(?i)^echo:bicuspid aortic valve", "echo.bicuspid_aortic_valve"),
    (r"(?i)^echo:mitral annular calcification", "echo.mitral_annular_calcification"),
    (r"(?i)^echo:aortic root dilation", "echo.aortic_root_dilation"),
    (r"(?i)^echo:dilated inferior vena cava", "echo.dilated_ivc"),
    (r"(?i)^echo:left ventricular diastolic", "echo.diastolic_dysfunction"),
    (r"(?i)^echo:pulmonary artery pressure", "echo.elevated_pa_pressure"),
    (r"(?i)^mg:breast cancer on screening", "mg.breast_cancer"),
    (r"(?i)^us:hepatic steatosis \(fatty liver\)$", "us.hepatic_steatosis"),
)

# --------------------------------------------------------------------------- #
# Normalização de campos
# --------------------------------------------------------------------------- #
FRACTION_METRICS = frozenset({"auroc", "sensitivity", "specificity", "dice", "accuracy"})
_NUM = re.compile(r"\d+(?:[.,·]\d+)?")


def normalize_text(text: str) -> str:
    """Espaços colapsados, ponto médio e traços tipográficos normalizados."""
    t = text.replace("·", ".").replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", t).strip()


def _to_float(token: str) -> float:
    return float(token.replace("·", ".").replace(",", "."))


@dataclass(frozen=True)
class ParsedValue:
    value: float
    printed: str
    bound: str | None  # ">" ou "<" quando a fonte publica só um limite
    percent: bool


def parse_value(raw: Any, metric: str) -> ParsedValue:
    """Converte o valor impresso na fonte em fração (métricas de 0 a 1).

    Raises:
        IngestError: o valor não é um número único reconhecível.
    """
    printed = str(raw).strip()
    bound = printed[0] if printed[:1] in (">", "<") else None
    body = printed.lstrip("><=≥≤ ").strip()
    percent = body.endswith("%")
    body = body.rstrip("%").strip()
    if not re.fullmatch(r"\d+(?:[.,·]\d+)?", body):
        raise IngestError(f"valor não numérico para métrica {metric!r}: {printed!r}")
    value = _to_float(body)
    if metric in FRACTION_METRICS and (percent or value > 1.0):
        value = value / 100.0
        percent = True
    return ParsedValue(round(value, 6), printed, bound, percent)


def parse_ci(raw: Any, metric: str, percent: bool) -> tuple[float, float] | None:
    """IC 95% como (inferior, superior) na mesma escala do valor; None se ausente."""
    if raw is None:
        return None
    if isinstance(raw, list):
        nums = [float(x) for x in raw]
    else:
        text = str(raw)
        if "±" in text or "+/-" in text:
            return None  # desvio, não intervalo
        nums = [_to_float(x) for x in _NUM.findall(text)]
    if len(nums) != 2:
        raise IngestError(f"IC não reconhecido para {metric!r}: {raw!r}")
    low, high = nums
    if metric in FRACTION_METRICS and (percent or max(low, high) > 1.0):
        low, high = low / 100.0, high / 100.0
    return round(low, 6), round(high, 6)


def value_in_quote(value: float, quote: str) -> bool:
    """O valor (em fração ou percentual) aparece literalmente na citação?"""
    text = normalize_text(quote)
    candidates = set()
    for v in (value, value * 100.0):
        for fmt in ("{:g}", "{:.1f}", "{:.2f}", "{:.3f}", "{:.4f}"):
            s = fmt.format(round(v, 6))
            candidates.add(s)
            if s.startswith("0."):
                candidates.add(s[1:])
    pattern = r"(?<![\d.])(?:" + "|".join(re.escape(c) for c in candidates) + r")(?!\d)"
    return re.search(pattern, text) is not None


def reference_standard(text: str | None, endpoint: str) -> str:
    """Traduz o padrão de referência textual para o vocabulário do esquema."""
    t = (text or "").lower()
    if re.search(r"biops|histolog|patholog", t):
        return "histopathology"
    if endpoint == "segmentation" or re.search(r"segmentation|mask|annotations|outline", t):
        return "expert_segmentation"
    if re.search(r"adjudicat|consensus|majority vote", t):
        return "radiologist_adjudicated"
    if re.search(r"nlp|text-mined|labeler|labeller|regex|report", t):
        return "report_nlp"
    if re.search(r"\bdxa\b|pdff|echocardiograph|xpert", t):
        return "clinical_reference_measurement"
    if re.search(r"(one|single) (board-certified )?radiologist|reader review", t):
        return "radiologist_single_read"
    return "other"


_DATASETS: tuple[tuple[str, str], ...] = (
    (r"^NIH ChestX-ray14", "NIH ChestX-ray14"),
    (r"^Google", "Google (NIH ChestX-ray14 relabel)"),
    (r"^RSNA \(", "RSNA Pneumonia (NIH-derived)"),
    (r"^SIIM", "SIIM-ACR (NIH-derived)"),
    (r"^PadChest", "PadChest"),
    (r"^VinBrain", "VinDr-CXR"),
    (r"^CheXpert", "CheXpert"),
    (r"^MIMIC-CXR", "MIMIC-CXR"),
    (r"NLST", "NLST"),
    (r"PleThora", "PleThora"),
    (r"LiTS", "LiTS"),
    (r"Univ\. Hospital Basel", "Univ. Hospital Basel (interno)"),
    (r"BraTS 2018", "BraTS 2018 (split do desenvolvedor)"),
    (r"MSSEG-1", "MSSEG-1"),
    (r"^103 test cases", "LST-AI test (103 casos públicos)"),
    (r"RSNA Screening Mammography", "RSNA Screening Mammography (hold-out)"),
    (r"Mayo Clinic", "Mayo Clinic (não descrito)"),
)


def dataset_name(population: str) -> str | None:
    for pattern, name in _DATASETS:
        if re.search(pattern, population):
            return name
    return None


# --------------------------------------------------------------------------- #
# Julgamento mecânico pela barra de publicação do protocolo
# --------------------------------------------------------------------------- #
def judge(
    *,
    kind: str,
    endpoint: str,
    metrics: dict[str, dict[str, float | None]],
    n_total: int | None,
    n_positive: int | None,
    peer_reviewed: bool,
    verified_at_source: bool,
    protocol: Protocol,
) -> tuple[bool, tuple[str, ...]]:
    """Aplica a barra de evidência publicada (nível B) do protocolo pré-registrado.

    Devolve ``(passa, motivos_de_reprovação)``. Nenhum número é ajustado: falta
    de n, de positivos ou de IC reprova, com o motivo declarado.
    """
    reasons: list[str] = []
    if kind == "published" and not peer_reviewed:
        reasons.append(
            "fonte não revisada por pares (repositório, cartão de modelo ou preprint); "
            "a barra exige publicação revisada por pares ou diretriz"
        )
    if not verified_at_source:
        reasons.append("citação não conferida byte a byte na fonte")
    if n_total is None:
        reasons.append("n do conjunto de teste não publicado")
    if n_positive is None:
        reasons.append(
            "número de positivos não publicado; a barra exige "
            f">= {protocol.published_min_positive} positivos"
        )
    elif n_positive < protocol.published_min_positive:
        reasons.append(f"{n_positive} positivos < mínimo de {protocol.published_min_positive}")

    def _check(name: str, minimum: float, label: str) -> None:
        m = metrics.get(name)
        if m is None or m.get("value") is None:
            reasons.append(f"{label} não publicado(a) neste registro")
            return
        value = float(m["value"] or 0.0)
        if value < minimum:
            reasons.append(f"{label} {value:.3f} < barra de {minimum:.2f}")
        if m.get("ci_low") is None or m.get("ci_high") is None:
            reasons.append(f"IC 95% de {label} não publicado")

    if endpoint == "discrimination":
        _check("auroc", protocol.published_auroc_min, "AUROC")
        others = sorted(set(metrics) - {"auroc"})
        if "auroc" not in metrics and others:
            reasons.append(f"métrica publicada ({', '.join(others)}) não tem barra no protocolo")
    elif endpoint == "detection":
        _check("sensitivity", protocol.published_sens_spec_min, "sensibilidade")
        _check("specificity", protocol.published_sens_spec_min, "especificidade")
    elif endpoint == "segmentation":
        _check("dice", protocol.segmentation_dice_min, "Dice")
    else:
        reasons.append(f"desfecho {endpoint!r} sem barra de publicação aplicável")
    return (not reasons), tuple(reasons)


# --------------------------------------------------------------------------- #
# Leitura das entradas
# --------------------------------------------------------------------------- #
def read_entries(path: Path) -> list[dict[str, Any]]:
    """Lê o bloco YAML ``published_evidence`` do fim de um ``*.verified.md``."""
    import yaml

    text = path.read_text(encoding="utf-8")
    match = _YAML_BLOCK.search(text)
    if match is None:
        raise IngestError(f"{path}: bloco YAML 'published_evidence' não encontrado")
    data = yaml.safe_load(match.group(1))
    entries = data.get("published_evidence") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise IngestError(f"{path}: 'published_evidence' vazio ou inválido")
    ids = [e.get("id") for e in entries]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise IngestError(f"{path}: ids duplicados {dup}")
    return entries


def match_model(entry: dict[str, Any]) -> ModelRule:
    for rule in MODEL_RULES:
        if re.search(rule.pattern, str(entry["model_or_method"])):
            return rule
    raise IngestError(
        f"entrada {entry['id']!r}: modelo {entry['model_or_method']!r} sem regra de anexação; "
        "decida explicitamente em MODEL_RULES"
    )


def is_peer_reviewed(entry: dict[str, Any]) -> bool:
    url = str(entry["url"])
    ident = str(entry.get("identifier") or "")
    if re.search(r"raw\.githubusercontent\.com|huggingface\.co|arxiv\.org|github\.com", url):
        return False
    return not re.search(r"(?i)preprint|model card|README", ident)


def is_verified(entry: dict[str, Any]) -> bool:
    """Ligado só quando o arquivo conferido confirma leitura literal na fonte."""
    note = str(entry.get("note") or "")
    pop = str(entry.get("population_or_dataset") or "")
    if entry["id"] == "pericardial_effusion_size":
        return False  # citação obtida por ferramenta de extração [WF], não byte a byte
    return not re.search(r"(?i)\[WF\]|web-fetch|not byte", note + pop)


def _group_key(entry: dict[str, Any], source_file: str) -> str:
    if source_file.startswith("evidence-cxr"):
        return "cxr"
    mod = str(entry.get("modality", "")).lower()
    if mod.startswith("echo"):
        return "echo"
    if mod.startswith("mammo"):
        return "mg"
    if "ultrasound" in mod:
        return "us"
    if mod.startswith("mr"):
        return "mr"
    return "ct"


def context_condition(entry: dict[str, Any], source_file: str) -> str | None:
    key = f"{_group_key(entry, source_file)}:{normalize_text(str(entry['condition']))}"
    for pattern, cid in CONTEXT_CONDITION_RULES:
        if re.search(pattern, key):
            return cid
    return None


def _endpoint_for(metrics: set[str]) -> str:
    if "sensitivity" in metrics or "specificity" in metrics:
        return "detection"
    if "dice" in metrics:
        return "segmentation"
    return "discrimination"


def _metric_name(entry: dict[str, Any]) -> str:
    metric = str(entry["metric"])
    if metric == "other" and entry["id"].startswith("monai_breast_density"):
        return "accuracy"
    return metric


def _triple(entry_id: str, model_id: str, population: str) -> tuple[str, str]:
    if model_id.startswith("xrv-"):
        col = re.search(r"column '([^']+)'", population)
        if col is None or col.group(1) not in XRV_LABEL_TO_CONDITION:
            raise IngestError(f"{entry_id}: coluna XRV não reconhecida em {population!r}")
        return XRV_LABEL_TO_CONDITION[col.group(1)], col.group(1)
    for pattern, cid, label in ENTRY_TRIPLES:
        if re.search(pattern, entry_id):
            return cid, label
    raise IngestError(f"{entry_id}: sem (condição, rótulo) em ENTRY_TRIPLES")


def _leakage(validation_type: str, model_id: str) -> tuple[str, str | None]:
    if validation_type in ("external", "zero_shot_external"):
        return "none", None
    if validation_type == "internal":
        note = (
            "validação interna: teste da mesma fonte/instituição do treino; independência do "
            "split em relação aos pesos distribuídos não verificável por nós"
        )
        if model_id.startswith("xrv-"):
            note += (
                "; SIIM/RSNA/Google são imagens do NIH (Filice 2020; Shih 2019), presente no "
                "treino destes pesos"
            )
        return "possible", note
    return "none", None


@dataclass
class Built:
    records: list[dict[str, Any]]
    context: list[dict[str, Any]]


def _context_item(
    entry: dict[str, Any], source_file: str, code: str, why: str | None, rule_caveat: str | None
) -> dict[str, Any]:
    reason = CONTEXT_REASONS[code] + (f": {why}" if why else "")
    if rule_caveat:
        reason += f" ({rule_caveat})"
    return {
        "source_entry": entry["id"],
        "source_file": source_file,
        "condition_text": entry["condition"],
        "condition_id": context_condition(entry, source_file),
        "model_or_method": entry["model_or_method"],
        "reason_code": code,
        "reason": reason,
        "metric": entry["metric"],
        "value": str(entry["value"]),
        "ci95": None if entry["ci95"] is None else str(entry["ci95"]),
        "n": entry["n"],
        "validation_type": entry["validation_type"],
        "population_or_dataset": entry["population_or_dataset"],
        "reference_standard": entry["reference_standard"],
        "note": entry.get("note"),
        "source": {
            "identifier": entry.get("identifier"),
            "url": entry["url"],
            "quote": entry["quote"],
            "peer_reviewed": is_peer_reviewed(entry),
            "verified_at_source": is_verified(entry),
        },
    }


class NotExposedError(IngestError):
    """O rótulo publicado não corresponde a nenhuma saída do manifesto: vai para contexto."""


def manifest_label(model_id: str, label: str, condition_id: str) -> tuple[str, str | None]:
    """Chave do rótulo no manifesto do modelo (o pipeline busca o nível por ela).

    Devolve ``(chave, ressalva)``. Manifesto sem lista de rótulos (modelos de
    segmentação sem saída por achado) mantém o rótulo publicado; correspondência
    a mais de uma saída mantém o rótulo publicado com ressalva.

    Raises:
        IngestError: o modelo não tem manifesto (id desatualizado).
        NotExposedError: o manifesto não expõe a saída publicada.
    """
    import yaml

    path = MANIFESTS_DIR / f"{model_id}.yaml"
    if not path.is_file():
        raise IngestError(f"modelo {model_id!r} sem manifesto em {MANIFESTS_DIR}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels = data.get("labels") or []
    if not labels:
        return label, None
    keys = [
        str(spec["key"])
        for spec in labels
        if label in (spec.get("key"), spec.get("source_label"))
        and spec.get("condition_id") == condition_id
    ]
    if not keys:
        raise NotExposedError(f"{model_id}: rótulo {label!r} ({condition_id}) fora do manifesto")
    if len(keys) > 1:
        return label, (
            f"rótulo publicado {label!r} corresponde a {len(keys)} saídas do manifesto "
            f"({', '.join(sorted(keys))}); mantido sem trio único de nível"
        )
    return keys[0], None


def _build_record(
    rec_id: str,
    entries: list[dict[str, Any]],
    rule: ModelRule,
    source_file: str,
    protocol: Protocol,
    metric_override: tuple[str, str] | None = None,
    extra_caveat: str | None = None,
) -> dict[str, Any]:
    from radiologyai.catalog.schema import EvidenceRecord

    first = entries[0]
    assert rule.model_id is not None
    urls = {e["url"] for e in entries}
    if len(urls) != 1:
        raise IngestError(f"{rec_id}: fusão com URLs diferentes {sorted(urls)}")
    quotes: list[str] = []
    for e in entries:
        if e["quote"] not in quotes:
            quotes.append(e["quote"])
    quote = " [...] ".join(quotes)

    metrics: dict[str, dict[str, float | None]] = {}
    caveats: list[str] = []
    sources = (
        [(metric_override[0], metric_override[1], None, first)]
        if metric_override
        else [(_metric_name(e), e["value"], e["ci95"], e) for e in entries]
    )
    for name, raw, raw_ci, e in sources:
        parsed = parse_value(raw, name)
        if not value_in_quote(parsed.value, quote):
            raise IngestError(f"{rec_id}: valor {parsed.printed!r} não está na citação")
        ci = parse_ci(raw_ci, name, parsed.percent)
        metrics[name] = {
            "value": parsed.value,
            "ci_low": ci[0] if ci else None,
            "ci_high": ci[1] if ci else None,
        }
        if parsed.bound:
            caveats.append(f"{name}: a fonte publica apenas um limite ({parsed.printed!r})")
        if raw_ci is not None and ci is None:
            caveats.append(f"{name}: dispersão publicada ({raw_ci!r}) não é IC 95%")
        if e.get("note"):
            caveats.append(f"nota do verificador: {e['note']}")

    endpoint = _endpoint_for(set(metrics))
    population = str(first["population_or_dataset"])
    condition_id, label = _triple(rec_id.removeprefix("pub."), rule.model_id, population)
    if condition_id not in CONDITION_IDS:
        raise IngestError(f"{rec_id}: condição {condition_id!r} fora de CONDITIONS")
    label, label_caveat = manifest_label(rule.model_id, label, condition_id)
    if label_caveat:
        caveats.append(label_caveat)
    n_values = {e["n"] for e in entries}
    n_total = first["n"] if len(n_values) == 1 else None
    if re.search(r"n counts lesions", population):
        caveats.append(f"n publicado ({first['n']}) conta lesões, não exames")
        n_total = None
    leakage, leak_note = _leakage(str(first["validation_type"]), rule.model_id)
    peer = is_peer_reviewed(first)
    verified = all(is_verified(e) for e in entries)
    n_positive: int | None = None
    supplement = COUNT_SUPPLEMENTS.get(rec_id)
    if supplement is not None:
        printed = str(supplement["printed"])
        if int(printed) != supplement["n_positive"] or not re.search(
            rf"(?<![\d.]){re.escape(printed)}(?![\d.])", normalize_text(supplement["quote"])
        ):
            raise IngestError(f"{rec_id}: contagem {printed!r} não está na citação do suplemento")
        n_positive = int(supplement["n_positive"])
        quote += " [...] " + supplement["quote"]
    passes, reasons = judge(
        kind="published",
        endpoint=endpoint,
        metrics=metrics,
        n_total=n_total,
        n_positive=n_positive,
        peer_reviewed=peer,
        verified_at_source=verified,
        protocol=protocol,
    )
    if rule.caveat:
        caveats.insert(0, rule.caveat)
    if extra_caveat:
        caveats.append(extra_caveat)
    if supplement is not None:
        caveats.append(
            f"n_positive={n_positive}: {supplement['location']} ({supplement['url']}): "
            f"'{supplement['quote']}'"
        )
        caveats.append(str(supplement["caveat"]))
    caveats.append(f"tipo de validação na fonte: {first['validation_type']}")
    caveats.append(f"padrão de referência (fonte): {first['reference_standard']}")
    caveats.append("origem: " + ", ".join(f"{source_file}#{e['id']}" for e in entries))
    record: dict[str, Any] = {
        "id": rec_id,
        "condition_id": condition_id,
        "model_id": rule.model_id,
        "label": label,
        "kind": "published",
        "endpoint": endpoint,
        "dataset": dataset_name(population),
        "population": population,
        "reference_standard": reference_standard(first["reference_standard"], endpoint),
        "n_total": n_total,
        "n_positive": n_positive,
        "metrics": metrics,
        "leakage": leakage,
        "leakage_note": leak_note,
        "passes_protocol": passes,
        "failure_reasons": list(reasons),
        "source": {
            "title": _title(first),
            "identifier": first.get("identifier"),
            "url": first["url"],
            "quote": quote,
            "peer_reviewed": peer,
            "verified_at_source": verified,
        },
        "caveats": caveats,
    }
    # Validação pelo esquema: falha alto em vez de gravar registro inválido.
    validated: dict[str, Any] = EvidenceRecord.model_validate(record).model_dump(mode="json")
    return validated


def _title(entry: dict[str, Any]) -> str:
    ident = str(entry.get("identifier") or "").strip()
    return f"{entry['model_or_method']} - {entry['condition']}" + (f" ({ident})" if ident else "")


def build(research_dir: Path, protocol: Protocol) -> tuple[dict[str, Built], dict[str, str]]:
    """Constrói registros e contexto por arquivo de saída. Devolve também os sha256 das entradas."""
    built: dict[str, Built] = {}
    hashes: dict[str, str] = {}
    used_supplements: set[str] = set()
    for filename, out_name in INPUTS:
        path = research_dir / filename
        if not path.is_file():
            raise IngestError(f"entrada ausente: {path}")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        entries = read_entries(path)
        by_id = {e["id"]: e for e in entries}
        merged_members = {m for members in MERGES.values() for m in members}
        records: list[dict[str, Any]] = []
        context: list[dict[str, Any]] = []
        for entry in entries:
            rule = match_model(entry)
            if entry["id"] in ENTRY_CONTEXT_OVERRIDES:
                code, why = ENTRY_CONTEXT_OVERRIDES[entry["id"]]
                context.append(_context_item(entry, filename, code, why, rule.caveat))
                continue
            if rule.model_id is None:
                assert rule.reason_code is not None
                context.append(_context_item(entry, filename, rule.reason_code, None, rule.caveat))
                continue
            if entry["id"] in merged_members:
                continue
            try:
                rec = _build_record(f"pub.{entry['id']}", [entry], rule, filename, protocol)
            except NotExposedError as exc:
                why = str(exc)
                if rule.model_id == "xrv-resnet50-res512-all" and any(
                    f"'{name}'" in why for name in XRV_RESNET_UNTRAINED
                ):
                    why += "; saída sem limiar de operação em models.py (np.nan): não treinada"
                context.append(
                    _context_item(entry, filename, "saida_nao_exposta", why, rule.caveat)
                )
                continue
            records.append(rec)
        for rec_id, members in MERGES.items():
            if not set(members) <= set(by_id):
                continue
            group = [by_id[m] for m in members]
            rule = match_model(group[0])
            if rule.model_id is None:  # membros já foram para o contexto, um a um
                continue
            records.append(_build_record(rec_id, group, rule, filename, protocol))
        for spec in DERIVED:
            if spec["from"] not in by_id:
                continue
            src = by_id[spec["from"]]
            records.append(
                _build_record(
                    spec["id"],
                    [src],
                    match_model(src),
                    filename,
                    protocol,
                    metric_override=(spec["metric"], spec["printed"]),
                    extra_caveat=spec["caveat"],
                )
            )
        records.sort(key=lambda r: r["id"])
        used_supplements.update(r["id"] for r in records if r["id"] in COUNT_SUPPLEMENTS)
        context.sort(key=lambda c: c["source_entry"])
        built[out_name] = Built(records, context)
    unused = sorted(set(COUNT_SUPPLEMENTS) - used_supplements)
    if unused:
        raise IngestError(f"suplementos de contagem sem registro correspondente: {unused}")
    return built, hashes


def proposed_conditions(
    built: dict[str, Built],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Condições referenciadas: (por registros anexados, só por contexto)."""
    from radiologyai.catalog.schema import Condition

    attached = {r["condition_id"] for b in built.values() for r in b.records}
    in_context = {c["condition_id"] for b in built.values() for c in b.context} - {None}
    out_attached: list[dict[str, Any]] = []
    out_context: list[dict[str, Any]] = []
    for spec in CONDITIONS:
        if spec.id not in attached and spec.id not in in_context:
            continue
        cond = Condition.model_validate(
            {
                "id": spec.id,
                "name_pt": spec.name_pt,
                "name_en": spec.name_en,
                "group": spec.id.split(".", 1)[0],
                "exam_types": list(spec.exam_types),
                "system": spec.system,
                "code": {"scheme": "99MEDAI", "value": spec.id, "meaning": spec.name_en},
                "icd10": spec.icd10,
                "description_pt": spec.description_pt + _SUFFIX,
            }
        ).model_dump(mode="json", exclude_defaults=True)
        (out_attached if spec.id in attached else out_context).append(cond)
    return out_attached, out_context


# --------------------------------------------------------------------------- #
# Renderização determinística
# --------------------------------------------------------------------------- #
_HEADER = (
    "# GERADO por scripts/evidence_ingest_published.py - não edite à mão.\n"
    "# Regere com: python scripts/evidence_ingest_published.py\n"
)


def _dump(data: dict[str, Any], comment: str) -> str:
    import yaml

    body = yaml.safe_dump(
        data, allow_unicode=True, sort_keys=True, width=100, default_flow_style=False
    )
    return _HEADER + comment + body


def render(built: dict[str, Built], hashes: dict[str, str], protocol: Protocol) -> dict[str, str]:
    """Conteúdo de cada arquivo de saída, por nome."""
    files: dict[str, str] = {}
    inputs = {name: {"sha256": digest} for name, digest in hashes.items()}
    for filename, out_name in INPUTS:
        b = built[out_name]
        files[f"{out_name}.yaml"] = _dump(
            {
                "generator": GENERATOR,
                "protocol_id": protocol.protocol_id,
                "input": {"file": filename, "sha256": hashes[filename]},
                "records": b.records,
            },
            "# Registros publicados anexados a modelos que rodamos com os mesmos pesos.\n",
        )
    context = [c for _, out_name in INPUTS for c in built[out_name].context]
    files["context.yaml"] = _dump(
        {
            "generator": GENERATOR,
            "inputs": inputs,
            "records": [],
            "context": context,
        },
        "# Evidência publicada NÃO anexável a modelo nosso (ver reason). records: [] é\n"
        "# proposital: radiologyai.evidence.store lê todo YAML deste diretório.\n",
    )
    attached, ctx_only = proposed_conditions(built)
    files["conditions_proposed.yaml"] = _dump(
        {
            "generator": GENERATOR,
            "records": [],
            "conditions": attached,
            "conditions_context_only": ctx_only,
        },
        "# Condições propostas para src/radiologyai/catalog/conditions.yaml.\n"
        "# 'conditions': referenciadas por registros anexados (obrigatórias para\n"
        "# evidence_build_levels). 'conditions_context_only': só por contexto.\n"
        "# records: [] é proposital (ver context.yaml).\n",
    )
    return files


def summarize(built: dict[str, Built]) -> dict[str, Any]:
    by_kind: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    passing: Counter[str] = Counter()
    failing: Counter[str] = Counter()
    ctx: Counter[str] = Counter()
    for name, b in built.items():
        for r in b.records:
            by_kind[r["kind"]] += 1
            by_model[r["model_id"]] += 1
            group = r["condition_id"].split(".", 1)[0]
            (passing if r["passes_protocol"] else failing)[group] += 1
        for c in b.context:
            ctx[f"{name}:{c['reason_code']}"] += 1
    return {
        "records_by_kind": dict(sorted(by_kind.items())),
        "records_by_model": dict(sorted(by_model.items())),
        "passing_by_group": dict(sorted(passing.items())),
        "failing_by_group": dict(sorted(failing.items())),
        "context_by_reason": dict(sorted(ctx.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check", action="store_true", help="falha se a saída estiver desatualizada"
    )
    args = parser.parse_args(argv)

    from radiologyai.catalog import load_protocol

    try:
        protocol = load_protocol()
        built, hashes = build(args.research_dir, protocol)
        files = render(built, hashes, protocol)
    except IngestError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2

    if args.check:
        stale = [
            name
            for name, text in files.items()
            if not (args.out / name).is_file()
            or (args.out / name).read_text(encoding="utf-8") != text
        ]
        if stale:
            print(f"desatualizado em {args.out}: {', '.join(sorted(stale))}", file=sys.stderr)
            return 1
        print(f"{args.out} atualizado ({len(files)} arquivos)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text, encoding="utf-8")
    import json

    print(json.dumps(summarize(built), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
