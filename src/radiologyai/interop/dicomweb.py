"""Cliente DICOMweb (QIDO-RS, WADO-RS, STOW-RS) para Orthanc e afins.

Usa ``dicomweb-client`` (MIT). A importação passa pela MESMA ingestão do envio
de arquivos — des-identificação incluída — e a busca devolve somente atributos
não identificadores (nunca nome nem ID do paciente).

Limitação declarada: como a ingestão pseudonimiza os UIDs, um SR/SEG exportado
referencia o estudo des-identificado, não o original no PACS.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from radiologyai.errors import BackendUnavailableError, ConfigurationError, RadiologyAIError

if TYPE_CHECKING:
    from radiologyai.persistence.config import PlatformSettings

__all__ = [
    "DicomwebError",
    "DicomwebGateway",
    "ImportTooLargeError",
    "StoreOutcome",
    "StudySummary",
    "store_outcome",
]

_SAFE_TAGS = {
    "0020000D": "study_instance_uid",
    "00081030": "study_description",
    "00080061": "modalities_in_study",
    "00201208": "number_of_instances",
    "00080020": "study_date",
}


class DicomwebError(RadiologyAIError):
    """Falha de comunicação ou resposta inválida do servidor DICOMweb."""


class ImportTooLargeError(DicomwebError):
    """Estudo remoto acima dos limites de importação (instâncias ou bytes)."""


@dataclass(frozen=True, slots=True)
class StoreOutcome:
    """O que o servidor CONFIRMOU no STOW-RS: armazenados e recusados (com motivo)."""

    stored: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]


def store_outcome(response: Any, sent: Sequence[str]) -> StoreOutcome:
    """Lê a resposta STOW-RS (PS3.18 §10.5): ``ReferencedSOPSequence`` e ``FailedSOPSequence``.

    Só conta como armazenado o que o servidor listou em ``ReferencedSOPSequence``;
    o enviado que não aparece em lugar nenhum é falha "não confirmado". Uma
    resposta 202 com ``FailedSOPSequence`` NÃO é sucesso.
    """
    stored: list[str] = []
    failed: list[tuple[str, str]] = []
    for item in getattr(response, "ReferencedSOPSequence", None) or []:
        uid = str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
        if uid:
            stored.append(uid)
    for item in getattr(response, "FailedSOPSequence", None) or []:
        uid = str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
        reason = getattr(item, "FailureReason", None)
        failed.append((uid, f"FailureReason {reason}" if reason is not None else "recusado"))
    seen = set(stored) | {uid for uid, _ in failed}
    failed.extend((uid, "não confirmado pelo servidor") for uid in sent if uid not in seen)
    return StoreOutcome(tuple(stored), tuple(failed))


class StudySummary(BaseModel):
    """Resultado de busca sem identificadores diretos do paciente."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_instance_uid: str
    study_description: str | None = None
    modalities_in_study: tuple[str, ...] = ()
    number_of_instances: int | None = None
    study_year: str | None = None


def _value(item: dict[str, Any], tag: str) -> Any:
    return (item.get(tag) or {}).get("Value") or []


class DicomwebGateway:
    """Fachada fina sobre ``dicomweb_client.DICOMwebClient``."""

    def __init__(
        self,
        url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
    ):
        if importlib.util.find_spec("dicomweb_client") is None:
            raise BackendUnavailableError(
                "dicomweb-client não instalado; instale o extra: "
                "pip install 'radiologyai[platform]'"
            )
        import requests
        from dicomweb_client.api import DICOMwebClient

        session = requests.Session()
        if username:
            session.auth = (username, password or "")
        self.url = url.rstrip("/")
        self._client = DICOMwebClient(url=self.url, session=session, timeout=timeout)
        self._client.set_http_retry_params(retry=False)

    @classmethod
    def from_settings(cls, settings: PlatformSettings) -> DicomwebGateway:
        """Raises: ConfigurationError: ``MEDAI_DICOMWEB_URL`` não configurada."""
        if not settings.dicomweb_url:
            raise ConfigurationError("MEDAI_DICOMWEB_URL não configurada")
        return cls(
            settings.dicomweb_url,
            username=settings.dicomweb_username,
            password=settings.dicomweb_password,
        )

    def search_studies(
        self, *, modality: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[StudySummary]:
        """QIDO-RS ``/studies``; só atributos não identificadores."""
        filters: dict[str, Any] = {}
        if modality:
            filters["ModalitiesInStudy"] = modality
        try:
            items = self._client.search_for_studies(
                search_filters=filters or None, limit=limit, offset=offset, get_remaining=False
            )
        except Exception as exc:  # noqa: BLE001 - reembalado com contexto
            raise DicomwebError(f"QIDO falhou em {self.url}: {exc}") from exc
        out: list[StudySummary] = []
        for item in items:
            uid = _value(item, "0020000D")
            if not uid:
                continue
            desc = _value(item, "00081030")
            date = _value(item, "00080020")
            count = _value(item, "00201208")
            out.append(
                StudySummary(
                    study_instance_uid=str(uid[0]),
                    study_description=str(desc[0]) if desc else None,
                    modalities_in_study=tuple(str(m) for m in _value(item, "00080061")),
                    number_of_instances=int(count[0]) if count else None,
                    study_year=str(date[0])[:4] if date else None,
                )
            )
        return out

    def retrieve_study(self, study_instance_uid: str) -> list[Any]:
        """WADO-RS do estudo inteiro (``pydicom.Dataset`` ainda identificados)."""
        try:
            return list(self._client.retrieve_study(study_instance_uid))
        except Exception as exc:  # noqa: BLE001 - reembalado com contexto
            raise DicomwebError(f"WADO falhou para o estudo: {exc}") from exc

    def count_instances(self, study_instance_uid: str, *, limit: int) -> int:
        """QIDO-RS das instâncias do estudo (só a contagem, limitada a ``limit``).

        Uma única página de ``limit`` itens: basta para saber se o estudo cabe no
        limite, sem paginar indefinidamente um servidor que ignore ``offset``.
        """
        try:
            items = self._client.search_for_instances(
                study_instance_uid=study_instance_uid, limit=limit, get_remaining=False
            )
        except Exception as exc:  # noqa: BLE001 - reembalado com contexto
            raise DicomwebError(f"QIDO de instâncias falhou: {exc}") from exc
        return len(items)

    def iter_study(self, study_instance_uid: str) -> Iterator[Any]:
        """WADO-RS instância a instância (não carrega o estudo inteiro na memória)."""
        try:
            yield from self._client.iter_study(study_instance_uid)
        except Exception as exc:  # noqa: BLE001 - reembalado com contexto
            raise DicomwebError(f"WADO falhou para o estudo: {exc}") from exc

    def store(self, datasets: Sequence[Any]) -> Any:
        """STOW-RS de objetos (SR/SEG gerados aqui), sempre como DICOM Part 10.

        Devolve a resposta do servidor; leia-a com :func:`store_outcome`.
        """
        import io

        import pydicom

        part10 = []
        for ds in datasets:
            buf = io.BytesIO()
            pydicom.dcmwrite(buf, ds, enforce_file_format=True)
            part10.append(pydicom.dcmread(io.BytesIO(buf.getvalue())))
        try:
            return self._client.store_instances(part10)
        except Exception as exc:  # noqa: BLE001 - reembalado com contexto
            raise DicomwebError(f"STOW falhou em {self.url}: {exc}") from exc
