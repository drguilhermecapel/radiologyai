"""Motor de inferência — falha fechada, sempre.

Este módulo é a antítese direta de ``legacy/src/medai_inference_system.py``, que
devolvia ``_create_dummy_model()`` quando não havia pesos e caía em
``_analyze_image_fallback()`` — uma heurística de OpenCV com linhas como
``fracture_score = min(0.3, pneumonia_score * 0.5)``.

Aqui não existe caminho de degradação. Pesos ausentes, hash divergente ou
backend não instalado levantam exceção e o processamento para.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from radiologyai import __version__
from radiologyai.errors import BackendUnavailableError, ModelNotFoundError
from radiologyai.io.metadata import extract_metadata
from radiologyai.io.reader import read_dicom, to_pixel_array
from radiologyai.modalities.base import ModalityPlugin, ValidationReport

if TYPE_CHECKING:
    from radiologyai.models.card import ModelCard


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """sha256 de um arquivo, lido em blocos."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class InferenceEngine:
    """Executa inferência sobre um exame, com gate de escopo obrigatório.

    Args:
        card: model card já validado.
        weights_path: caminho local dos pesos. O sha256 é verificado antes de
            qualquer carga.

    Raises:
        ModelNotFoundError: pesos ausentes.
        WeightsIntegrityError: hash divergente (verificado em ``__init__``).
    """

    def __init__(self, card: ModelCard, weights_path: str | Path) -> None:
        self._card = card
        self._weights_path = Path(weights_path)

        if not self._weights_path.is_file():
            raise ModelNotFoundError(
                f"pesos não encontrados para {card.card_id}: {self._weights_path}. "
                "Nenhum modelo substituto será usado."
            )
        card.verify_weights(self._weights_path)
        self._plugin = resolve_for_card(card)
        self._backend: object | None = None

    @property
    def card(self) -> ModelCard:
        return self._card

    def validate_study(self, path: str | Path) -> ValidationReport:
        """Aplica o gate de escopo da modalidade sem executar o modelo."""
        ds = read_dicom(path, stop_before_pixels=True)
        report: ValidationReport = self._plugin.validate(extract_metadata(ds))
        return report

    def predict(self, path: str | Path) -> object:
        """Executa a inferência sobre um arquivo DICOM.

        O gate de escopo roda primeiro e recusa entrada fora do uso pretendido
        (perigo H-03). Só então os pixels são lidos.

        Raises:
            OutOfScopeError: entrada fora do uso pretendido.
            BackendUnavailableError: backend do card não instalado.
        """
        ds = read_dicom(path)
        metadata = extract_metadata(ds)

        self._plugin.validate(metadata).raise_if_rejected()

        array = to_pixel_array(ds, scale="modality")
        tensor = self._plugin.preprocess(array, metadata)

        # Levanta se o pacote do backend declarado estiver ausente.
        self._require_backend()

        # O backend de pesos reais entra na Fase 2. Até lá o pipeline lê,
        # valida e pré-processa corretamente, mas NENHUMA predição é fabricada.
        raise BackendUnavailableError(
            f"backend {self._card.backend!r} ainda não implementado nesta versão. "
            f"Entrada validada e pré-processada com shape {tensor.shape}; "
            "nenhuma predição é fabricada. Ver ROADMAP.md §5, Fase 2."
        )

    def _require_backend(self) -> None:
        """Exige o pacote do backend declarado. Nunca recorre a substituto."""
        import importlib.util

        required = {"torch": "torch", "onnx": "onnxruntime"}[self._card.backend]
        if importlib.util.find_spec(required) is None:
            raise BackendUnavailableError(
                f"model card {self._card.card_id} exige backend {self._card.backend!r}, "
                f"que precisa do pacote {required!r}. "
                f"Instale com: pip install 'radiologyai[ml]'. "
                "Nenhum caminho alternativo será usado."
            )


def resolve_for_card(card: ModelCard) -> ModalityPlugin:
    """Resolve o plugin de modalidade declarado por um model card."""
    from radiologyai.modalities.registry import available_modalities

    for plugin in available_modalities().values():
        if plugin.code == card.modality.upper():
            return plugin
    raise ModelNotFoundError(
        f"nenhum plugin registrado para a modalidade {card.modality!r} do card {card.card_id}"
    )


def code_version() -> str:
    """Versão do código, gravada em todo resultado para rastreabilidade."""
    return __version__
