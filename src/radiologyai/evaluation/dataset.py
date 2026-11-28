"""Utilidades de dataset para avaliação: faixa etária, hash v2 e rótulos ausentes.

Corrige três defeitos do caminho de avaliação de 2026-09 sem alterar
``radiologyai.data`` (fora do escopo deste módulo):

1. **Faixa etária.** ``ManifestRow.age_band()`` coloca menores de 18 anos em
   ``"18-39"``. O teste oficial do NIH tem 1.070 imagens de menores de 18; elas
   eram contadas como adultos jovens. :func:`age_band` tem a faixa ``"<18"``
   separada.
2. **Hash do manifesto.** ``Manifest.sha256()`` cobre só ``image_id``,
   ``patient_id`` e rótulos: duas versões com demografia diferente (sexo,
   idade, incidência) tinham o mesmo hash, embora os subgrupos mudem.
   :func:`manifest_sha256_v2` cobre todas as colunas. O hash antigo continua
   gravado para que os artefatos já commitados permaneçam verificáveis.
3. **Rótulo ausente.** ``Manifest.label_matrix()`` usa ``labels.get(nome, 0)``:
   no CheXpert, "incerto" ou "não mencionado" (excluídos pelo carregador) voltam
   como **negativos**. :func:`label_matrix_with_missing` devolve ``-1`` para
   ausente, e o executor exclui essas imagens daquele rótulo.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.data.manifest import Manifest

MISSING_LABEL = -1
"""Valor na matriz de verdade para "rótulo não disponível nesta imagem"."""

AGE_BANDS: tuple[str, ...] = ("<18", "18-39", "40-59", "60-74", "75+")
UNKNOWN_BAND = "desconhecida"


def age_band(age_years: float | None) -> str:
    """Faixa etária para subgrupos, com menores de 18 anos em faixa própria."""
    if age_years is None or (isinstance(age_years, float) and math.isnan(age_years)):
        return UNKNOWN_BAND
    if age_years < 18:
        return "<18"
    if age_years < 40:
        return "18-39"
    if age_years < 60:
        return "40-59"
    if age_years < 75:
        return "60-74"
    return "75+"


def _canonical(value: Any) -> Any:
    """Normaliza um valor para serialização estável (float com repr fixo, None, str)."""
    if value is None:
        return None
    if hasattr(value, "item") and callable(value.item):  # escalares numpy/pandas
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(f"{value:.6g}") if not value.is_integer() else int(value)
    return str(value)


def canonical_rows_sha256(
    rows: Iterable[Mapping[str, Any]], columns: Sequence[str], *, key: str = "image_id"
) -> str:
    """sha256 de uma tabela, independente da ordem das linhas.

    Cada linha vira um JSON compacto com as ``columns`` na ordem dada; as linhas
    são ordenadas por ``key`` antes de hashear. Cabeçalho (nomes e ordem das
    colunas) entra no hash: mudar o conjunto de colunas muda o hash.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(list(columns), ensure_ascii=False).encode("utf-8"))
    digest.update(b"\n")
    materialized = [{c: _canonical(r.get(c)) for c in columns} for r in rows]
    for row in sorted(materialized, key=lambda r: str(r.get(key))):
        digest.update(
            json.dumps([row[c] for c in columns], ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_sha256_v2(manifest: Manifest) -> str:
    """Hash v2 do manifesto: todas as colunas, inclusive demografia e incidência.

    Rótulo ausente é serializado como ``null`` — distinto de ``0`` (negativo).
    """
    label_cols = [f"label:{n}" for n in manifest.label_names]
    columns = [
        "image_id",
        "patient_id",
        "view_position",
        "patient_sex",
        "patient_age_years",
        "sha256",
        *label_cols,
    ]
    rows = []
    for r in manifest.rows:
        row: dict[str, Any] = {
            "image_id": r.image_id,
            "patient_id": r.patient_id,
            "view_position": r.view_position,
            "patient_sex": r.patient_sex,
            "patient_age_years": r.patient_age_years,
            "sha256": r.sha256,
        }
        for name, col in zip(manifest.label_names, label_cols, strict=True):
            row[col] = r.labels.get(name)
        rows.append(row)
    return canonical_rows_sha256(rows, columns)


def label_matrix_with_missing(manifest: Manifest) -> npt.NDArray[np.int64]:
    """Matriz ``n_imagens x n_rótulos`` com :data:`MISSING_LABEL` onde não há rótulo."""
    import numpy as np

    return np.array(
        [
            [row.labels.get(name, MISSING_LABEL) for name in manifest.label_names]
            for row in manifest.rows
        ],
        dtype=np.int64,
    ).reshape(len(manifest.rows), len(manifest.label_names))
