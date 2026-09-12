"""CheXpert — rótulos adjudicados, sem mapeamento forçado."""

from __future__ import annotations

import csv

import pytest

from radiologyai.data.chexpert import (
    CHEXPERT_LABELS,
    CHEXPERT_TO_XRV,
    LIMITATIONS,
    NOT_MAPPED,
    _parse_label,
    build_valid_manifest,
    find_valid_csv,
    resolve_image_path,
)
from radiologyai.errors import EvaluationError

CAMPOS = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", *CHEXPERT_LABELS]


def escrever_valid(root, linhas, nome="valid.csv"):
    caminho = root / nome
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(linhas)
    return caminho


def linha(path, *, frontal="Frontal", vista="AP", sexo="Male", idade="57", **rotulos):
    base = dict.fromkeys(CHEXPERT_LABELS, "")
    base.update(rotulos)
    return {
        "Path": path,
        "Sex": sexo,
        "Age": idade,
        "Frontal/Lateral": frontal,
        "AP/PA": vista,
        **base,
    }


@pytest.fixture
def chexpert_root(tmp_path):
    escrever_valid(
        tmp_path,
        [
            linha(
                "CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg",
                Cardiomegaly="1.0",
                **{"Pleural Effusion": "0.0"},
            ),
            linha(
                "CheXpert-v1.0/valid/patient64541/study2/view1_frontal.jpg",
                Edema="1.0",
                Pneumonia="0.0",
            ),
            linha(
                "CheXpert-v1.0/valid/patient64542/study1/view2_lateral.jpg",
                frontal="Lateral",
                vista="",
                Cardiomegaly="1.0",
            ),
            linha(
                "CheXpert-v1.0/valid/patient64543/study1/view1_frontal.jpg",
                sexo="Female",
                idade="72",
                Atelectasis="1.0",
                Consolidation="-1.0",
            ),
        ],
    )
    return tmp_path


class TestLocalizacao:
    def test_encontra_valid_csv(self, chexpert_root):
        assert find_valid_csv(chexpert_root).name == "valid.csv"

    def test_aceita_layout_com_diretorio_raiz(self, tmp_path):
        escrever_valid(tmp_path / "CheXpert-v1.0", [linha("x.jpg")], nome="valid.csv")
        assert find_valid_csv(tmp_path).is_file()

    def test_ausente_aponta_para_o_portal(self, tmp_path):
        with pytest.raises(EvaluationError, match="aimi.stanford.edu"):
            find_valid_csv(tmp_path)


class TestRotulos:
    @pytest.mark.parametrize(("valor", "esperado"), [("1.0", 1), ("1", 1), ("0.0", 0), ("0", 0)])
    def test_positivo_e_negativo(self, valor, esperado):
        assert _parse_label(valor) == esperado

    @pytest.mark.parametrize("valor", ["-1.0", "", "  ", "abc", None])
    def test_incerto_e_ausente_viram_none(self, valor):
        assert _parse_label(valor) is None

    def test_incerto_nao_vira_negativo(self, chexpert_root):
        """Contar incerto como negativo inflaria a especificidade artificialmente."""
        m = build_valid_manifest(chexpert_root)
        linha_incerta = next(r for r in m if "patient64543" in r.image_id)
        assert "Consolidation" not in linha_incerta.labels
        assert linha_incerta.labels["Atelectasis"] == 1


class TestMapeamento:
    def test_effusion_e_o_unico_renomeado(self):
        renomeados = {k: v for k, v in CHEXPERT_TO_XRV.items() if k != v}
        assert renomeados == {"Pleural Effusion": "Effusion"}

    def test_nenhum_achado_distinto_colapsado(self):
        """O legado mapeava Pneumothorax->pneumonia. Aqui a relação é injetiva."""
        destinos = list(CHEXPERT_TO_XRV.values())
        assert len(destinos) == len(set(destinos))

    def test_sem_contrapartida_e_declarado_nao_forcado(self):
        for nome in NOT_MAPPED:
            assert nome in CHEXPERT_LABELS
            assert nome not in CHEXPERT_TO_XRV

    def test_todos_os_mapeados_existem_no_csv_oficial(self):
        for nome in CHEXPERT_TO_XRV:
            assert nome in CHEXPERT_LABELS


class TestManifest:
    def test_filtra_laterais_por_default(self, chexpert_root):
        """O uso pretendido (REG-01 §2) cobre apenas incidências frontais."""
        m = build_valid_manifest(chexpert_root)
        assert len(m) == 3
        assert all("lateral" not in r.image_id for r in m)

    def test_pode_incluir_laterais(self, chexpert_root):
        assert len(build_valid_manifest(chexpert_root, frontal_only=False)) == 4

    def test_patient_id_extraido_do_caminho(self, chexpert_root):
        m = build_valid_manifest(chexpert_root)
        assert {r.patient_id for r in m} == {"patient64541", "patient64543"}

    def test_duas_imagens_do_mesmo_paciente_agrupam(self, chexpert_root):
        m = build_valid_manifest(chexpert_root)
        assert len(m) == 3
        assert len(m.patient_ids) == 2

    def test_demografia_extraida(self, chexpert_root):
        m = build_valid_manifest(chexpert_root)
        r = next(x for x in m if "patient64543" in x.image_id)
        assert (r.patient_sex, r.patient_age_years) == ("F", 72.0)

    def test_rotulos_usam_nomenclatura_xrv(self, chexpert_root):
        m = build_valid_manifest(chexpert_root)
        assert "Effusion" in m.label_names
        assert "Pleural Effusion" not in m.label_names

    def test_csv_sem_colunas_do_chexpert_falha(self, tmp_path):
        (tmp_path / "valid.csv").write_text("Path,Sex\nx.jpg,Male\n", encoding="utf-8")
        with pytest.raises(EvaluationError, match="colunas esperadas"):
            build_valid_manifest(tmp_path)

    def test_manifest_vazio_falha(self, tmp_path):
        escrever_valid(tmp_path, [linha("x.jpg", frontal="Lateral")])
        with pytest.raises(EvaluationError, match="nenhuma linha"):
            build_valid_manifest(tmp_path)

    def test_hash_independente_da_ordem(self, chexpert_root):
        import random

        a = build_valid_manifest(chexpert_root)
        b = build_valid_manifest(chexpert_root)
        random.Random(3).shuffle(b.rows)
        assert a.sha256() == b.sha256()


class TestCaminhoDeImagem:
    def test_resolve_com_prefixo(self, tmp_path):
        alvo = tmp_path / "CheXpert-v1.0" / "valid" / "a.jpg"
        alvo.parent.mkdir(parents=True)
        alvo.write_bytes(b"x")
        assert resolve_image_path("CheXpert-v1.0/valid/a.jpg", tmp_path) == alvo

    def test_resolve_sem_prefixo(self, tmp_path):
        alvo = tmp_path / "valid" / "a.jpg"
        alvo.parent.mkdir(parents=True)
        alvo.write_bytes(b"x")
        assert resolve_image_path("CheXpert-v1.0/valid/a.jpg", tmp_path) == alvo

    def test_ausente_falha(self, tmp_path):
        with pytest.raises(EvaluationError, match="não encontrada"):
            resolve_image_path("valid/x.jpg", tmp_path)


def test_limitacoes_declaram_a_diferenca_em_relacao_ao_nih():
    texto = " ".join(LIMITATIONS)
    assert "3 radiologistas" in texto
    assert "NÃO uma validação clínica" in texto
