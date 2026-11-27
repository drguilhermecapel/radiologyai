"""Tabela explícita de vazamento entre dados de treino e dataset de avaliação.

O executor de 2026-09 decidia vazamento por **substring** do nome do dataset
contra o texto livre ``trained_on`` do model card — ``"NIH"`` em
``"NIH ChestX-ray14"`` funcionava; ``"SIIM-ACR"`` (anotações de imagens do NIH)
passaria como externo. Aqui o vazamento sai de uma tabela de linhagem com
**chaves** de dataset: chave desconhecida é erro, nunca "externo" por omissão.

Regras (``docs/protocolo-evidencia.md`` §8):

* ``known`` — a linhagem do dataset de avaliação (ele mesmo e tudo de que deriva)
  intersecta a linhagem de algum dataset de treino do modelo, **ou** de algum
  dataset usado para SELECIONAR o modelo (codificador, hiperparâmetro,
  pré-processamento escolhidos comparando desempenho nele);
* ``possible`` — o modelo não declara dados de treino, ou o par (linhagem do
  treino, linhagem da avaliação) está na tabela :data:`POSSIBLE_OVERLAP`
  (sobreposição plausível, não demonstrada);
* ``none`` — nenhum dos casos acima.

Reuso na seleção é vazamento: escolher o codificador pela AUROC externa num
conjunto e depois reportar a AUROC desse mesmo conjunto como validação externa
torna a estimativa otimista (o conjunto deixou de ser independente da escolha).
O vazamento só por seleção fica separado (:func:`selection_overlap`) porque ele
só pode **inflar** a estimativa: impede a promoção a A, mas não livra o modelo de
uma refutação (ver :func:`radiologyai.catalog.schema.compute_level`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from radiologyai.catalog.schema import Leakage
from radiologyai.errors import EvidenceError


@dataclass(frozen=True)
class DatasetInfo:
    """Uma entrada da tabela de linhagem."""

    key: str
    derived_from: tuple[str, ...] = ()
    note: str = ""


DATASETS: dict[str, DatasetInfo] = {
    d.key: d
    for d in (
        # --- radiografia de tórax
        DatasetInfo("nih-cxr14", note="NIH ChestX-ray14 (NIH Clinical Center)"),
        DatasetInfo(
            "nih-cxr14-google",
            ("nih-cxr14",),
            "rótulos adjudicados do Google sobre imagens do NIH (Majkowska 2020)",
        ),
        DatasetInfo(
            "siim-acr-pneumothorax",
            ("nih-cxr14",),
            "anotações de imagens do NIH (Filice 2020, DOI 10.1007/s10278-019-00299-9)",
        ),
        DatasetInfo(
            "rsna-pneumonia",
            ("nih-cxr14",),
            "anotações de imagens do NIH (Shih 2019, DOI 10.1148/ryai.2019180041)",
        ),
        DatasetInfo("padchest", note="PadChest (Hospital San Juan, Alicante)"),
        DatasetInfo("chexpert", note="CheXpert (Stanford)"),
        DatasetInfo("chexpert-valid", ("chexpert",), "validação oficial do CheXpert"),
        DatasetInfo("mimic-cxr", note="MIMIC-CXR (BIDMC)"),
        DatasetInfo("openi", note="Open-i / Indiana University"),
        DatasetInfo("vindr-cxr"),
        DatasetInfo("brax", note="BRAX (Hospital Israelita Albert Einstein, São Paulo)"),
        DatasetInfo("tb-shenzhen"),
        DatasetInfo("tb-montgomery"),
        DatasetInfo("kaggle-pneumonia-pediatric", note="Guangzhou pediátrico (Kermany 2018)"),
        # --- musculoesquelético
        DatasetInfo("fracatlas"),
        DatasetInfo("grazpedwri-dx"),
        DatasetInfo(
            "grazpedwri-dx-sample",
            ("grazpedwri-dx",),
            "amostra local de 400 pacientes (numpy default_rng(0) sobre os 1500 pacientes "
            "do images_part3.zip; DATASET.json do cache)",
        ),
        DatasetInfo(
            "grazpedwri-dx-sample-test30",
            ("grazpedwri-dx-sample",),
            "30% dos pacientes da amostra (ordem sha256 do id), hold-out das sondas GRAZPEDWRI",
        ),
        DatasetInfo("mura"),
        # --- TC / RM
        DatasetInfo("cq500"),
        DatasetInfo("rsna-ich"),
        DatasetInfo("totalsegmentator-ct"),
        DatasetInfo("totalsegmentator-mr"),
        DatasetInfo("lidc-idri"),
        DatasetInfo("brats"),
        # --- mamografia
        DatasetInfo("cbis-ddsm", ("ddsm",), "subconjunto curado do DDSM"),
        DatasetInfo("ddsm"),
        DatasetInfo("cmmd"),
        DatasetInfo("vindr-mammo"),
        DatasetInfo("rsna-screening-mammo"),
        DatasetInfo("inbreast"),
        # --- ultrassom / eco
        DatasetInfo("busi"),
        DatasetInfo("bus-bra"),
        DatasetInfo("breast-lesions-usg"),
        DatasetInfo("byra-fatty-liver"),
        DatasetInfo("camus"),
        DatasetInfo("echonet-dynamic"),
        DatasetInfo(
            "liver-lesion-us-7272660",
            note="Annotated Ultrasound Liver images (Xu et al., Zenodo 7272660)",
        ),
        # --- corpora de pré-treino de codificadores e de modelos de fundação
        DatasetInfo(
            "pmc-15m",
            note="PMC-15M: 15 milhões de pares figura-legenda do PubMed Central (BiomedCLIP)",
        ),
        DatasetInfo(
            "medsam2-training-corpus",
            note="bases públicas de US/vídeo do MedSAM2 (Tabela Suplementar 1 não conferida)",
        ),
    )
}

# Datasets cujo termo de uso proíbe ou restringe uso comercial. Um modelo treinado
# em algum deles declara ``license.commercial_use: 'no'`` no manifesto (REQ-188).
NONCOMMERCIAL_TERMS: dict[str, str] = {
    "chexpert": "Stanford Research Use Agreement (não comercial)",
    "camus": "CC BY-NC-SA 4.0",
    "mimic-cxr": "PhysioNet Credentialed Health Data License (uso sob o DUA)",
}

POSSIBLE_OVERLAP: frozenset[frozenset[str]] = frozenset(
    {
        # BUSI não tem id de paciente e tem duplicatas (Pawłowska 2023,
        # DOI 10.1016/j.dib.2023.109247); sua origem (Baheya, Cairo) não exclui
        # sobreposição com outras coleções egípcias públicas.
        frozenset({"busi", "breast-lesions-usg"}),
        # PMC-15M (pré-treino do BiomedCLIP) reúne figuras de artigos de acesso aberto
        # do PubMed Central; artigos que descrevem ou usam estes datasets públicos de US
        # trazem imagens de exemplo com legenda. Sobreposição plausível, não demonstrada.
        *(
            frozenset({"pmc-15m", key})
            for key in (
                "busi",
                "breast-lesions-usg",
                "bus-bra",
                "byra-fatty-liver",
                "liver-lesion-us-7272660",
            )
        ),
        # O artigo do MedSAM2 (arXiv 2504.03600) avalia no CAMUS e chama de hold-out, mas
        # a lista de bases de treino (Tabela Suplementar 1) não foi conferida: não se
        # sabe se pacientes do CAMUS entraram no checkpoint MedSAM2_US_Heart.pt.
        frozenset({"medsam2-training-corpus", "camus"}),
    }
)


def _require(key: str) -> DatasetInfo:
    try:
        return DATASETS[key]
    except KeyError:
        raise EvidenceError(
            f"chave de dataset desconhecida: {key!r}. Registre-a em "
            "radiologyai.evidence.leakage.DATASETS com a sua linhagem — vazamento "
            "nunca é presumido ausente para um dataset sem linhagem declarada."
        ) from None


def lineage(key: str) -> frozenset[str]:
    """O dataset e tudo de que ele deriva (fecho transitivo)."""
    seen: set[str] = set()
    stack = [key]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(_require(current).derived_from)
    return frozenset(seen)


def selection_overlap(dataset_key: str, selection_data: Iterable[str]) -> tuple[str, ...]:
    """Chaves de seleção cuja linhagem intersecta a do dataset de avaliação (ordenadas).

    Raises:
        EvidenceError: alguma chave fora da tabela.
    """
    evaluated = lineage(dataset_key)
    return tuple(sorted({s for s in selection_data if evaluated & lineage(s)}))


def _training_leakage(evaluated: frozenset[str], train: tuple[str, ...]) -> tuple[Leakage, str]:
    if not train:
        return "possible", "o modelo não declara dados de treino; vazamento não pode ser excluído"
    hits: list[str] = []
    for t in train:
        common = sorted(evaluated & lineage(t))
        if common:
            hits.append(f"{t} (linhagem comum: {', '.join(common)})")
    if hits:
        return "known", "dataset de avaliação compartilha linhagem com o treino: " + "; ".join(hits)
    # O par é procurado na linhagem dos DOIS lados: um treino derivado (ex.: uma cópia
    # limpa de um dataset da tabela) herda a sobreposição plausível da sua origem.
    possible = [
        t
        for t in train
        for tl in lineage(t)
        for e in evaluated
        if frozenset({tl, e}) in POSSIBLE_OVERLAP
    ]
    if possible:
        return "possible", "sobreposição plausível declarada na tabela: " + ", ".join(
            sorted(set(possible))
        )
    return "none", "nenhuma linhagem comum entre avaliação e treino (" + ", ".join(train) + ")"


def assess_training_leakage(dataset_key: str, training_data: Iterable[str]) -> tuple[Leakage, str]:
    """Vazamento só pela linhagem dos dados de TREINO (sem a seleção de modelo).

    Raises:
        EvidenceError: alguma chave (avaliação ou treino) fora da tabela.
    """
    return _training_leakage(lineage(dataset_key), tuple(training_data))


def assess_leakage(
    dataset_key: str,
    training_data: Iterable[str],
    selection_data: Iterable[str] = (),
) -> tuple[Leakage, str]:
    """``(leakage, nota)`` para um dataset de avaliação, o treino e a seleção do modelo.

    ``selection_data`` são as chaves dos datasets em que o modelo (ou o seu
    codificador, hiperparâmetros, pré-processamento) foi **escolhido** comparando
    desempenho. Se a linhagem da avaliação as encontra, o vazamento é ``known``.

    Raises:
        EvidenceError: alguma chave (avaliação, treino ou seleção) fora da tabela.
    """
    evaluated = lineage(dataset_key)
    leakage, note = _training_leakage(evaluated, tuple(training_data))
    reused = selection_overlap(dataset_key, selection_data)
    if not reused:
        return leakage, note
    selection_note = (
        "dataset de avaliação reutilizado na seleção do modelo/codificador "
        f"({', '.join(reused)}): a estimativa é otimista e não é validação externa"
    )
    if leakage == "known":
        return "known", f"{note}; {selection_note}"
    return "known", f"{selection_note}; treino: {note}"
