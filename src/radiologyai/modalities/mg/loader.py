"""Carregador de mamografia: ``load_dicom`` + VOI completa + lateralidade/incidência.

:func:`load_mammogram` devolve o mesmo ``ImageData`` do carregador comum
(pixels após a Modality LUT, polaridade MONOCHROME2, nenhuma janela aplicada)
com dois acréscimos que os modelos de mamografia exigem:

* ``extra["dicom_voi"]``: parâmetros VOI completos do arquivo (função
  ``LINEAR``/``LINEAR_EXACT``/``SIGMOID`` e ``VOILUTSequence``), ver
  :mod:`radiologyai.modalities.mg.voi`;
* lateralidade e incidência resolvidas por :mod:`radiologyai.modalities.mg.views`
  quando o carregador comum as deixou vazias (``PatientOrientation`` como último
  recurso; a origem fica em ``extra["laterality_source"]``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from radiologyai.analysis.types import ImageData, Modality
from radiologyai.errors import UnsupportedInputError
from radiologyai.modalities.mg.views import parse_laterality, parse_view
from radiologyai.modalities.mg.voi import VOI_EXTRA_KEY, voi_from_dataset

__all__ = ["load_mammogram", "with_dataset_details"]


def with_dataset_details(image: ImageData, ds: Any) -> ImageData:
    """Acrescenta VOI completa, lateralidade e incidência de ``ds`` a ``image``.

    Valores já presentes em ``image.meta`` (por exemplo, sobrescritos pelo
    chamador) são mantidos.
    """
    updates: dict[str, Any] = {}
    extra = dict(image.extra)
    if image.meta.laterality is None:
        lat, source = parse_laterality(
            getattr(ds, "ImageLaterality", None),
            getattr(ds, "Laterality", None),
            getattr(ds, "PatientOrientation", None),
        )
        if lat is not None:
            updates["laterality"] = lat
            extra["laterality_source"] = source
    if image.meta.view_position is None:
        view = parse_view(getattr(ds, "ViewPosition", None), getattr(ds, "ViewCodeSequence", None))
        if view is not None:
            updates["view_position"] = view
    extra[VOI_EXTRA_KEY] = voi_from_dataset(ds)
    meta = image.meta.model_copy(update=updates) if updates else image.meta
    return dataclasses.replace(image, meta=meta, extra=extra)


def load_mammogram(path: str | Path, **meta_overrides: Any) -> ImageData:
    """Um arquivo DICOM de mamografia (MG, quadro único) → ``ImageData``.

    Raises:
        InvalidDICOMError: arquivo ilegível.
        UnsupportedInputError: modalidade diferente de MG, multiframe
            (tomossíntese) ou fotometria não suportada.
    """
    import pydicom

    from radiologyai.analysis.imaging import load_dicom

    image = load_dicom(path, **meta_overrides)
    if image.modality is not Modality.MG:
        raise UnsupportedInputError(f"{path}: modalidade {image.modality.value}, esperada MG")
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    return with_dataset_details(image, ds)
