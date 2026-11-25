"""Registro de modelos: manifestos, integridade dos pesos, carga de adaptadores.

CONTRATO (métodos públicos).

Regras impostas aqui
--------------------
* Modelo com ``status: planned`` (ou ``retired``) nunca é carregado nem baixado.
* Pesos só são baixados por :meth:`Registry.pull`, para ``<weights_dir>/<model_id>/``,
  e conferidos contra o sha256 fixado no manifesto ANTES de qualquer adaptador
  tocá-los. Arquivo ausente ou divergente levanta
  :class:`~radiologyai.errors.ModelUnavailableError` — quem chama não tem fallback.
* :func:`pin` escreve no manifesto o sha256 e o tamanho calculados dos bytes
  reais (baixados ou já presentes). Nunca se digita um hash.
* Adaptadores só são importados de ``radiologyai.analysis.adapters``.
* Arquivos compactados são extraídos com recusa de caminho absoluto, ``..`` e links.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import tarfile
import tempfile
import threading
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from radiologyai import __version__
from radiologyai.analysis.artifacts import sha256_file
from radiologyai.analysis.base import ModelAdapter, ResolvedWeights
from radiologyai.analysis.manifest import ModelManifest, WeightFile, load_manifest, parse_manifest
from radiologyai.analysis.types import ModelProvenance
from radiologyai.errors import (
    BackendUnavailableError,
    IntegrityCheckError,
    ManifestError,
    ModelUnavailableError,
)

if TYPE_CHECKING:
    import httpx

    from radiologyai.config.settings import Settings

__all__ = [
    "PACKAGE_MANIFEST_DIR",
    "ProgressFn",
    "Registry",
    "WeightState",
    "pin",
    "safe_extract",
]

PACKAGE_MANIFEST_DIR = Path(__file__).parent / "manifests"
ProgressFn = Callable[[str, int, int], None]  # (chave do peso, bytes feitos, bytes totais)
_EXTRACT_MARKER = ".radiologyai-archive-sha256"


@dataclass(frozen=True, slots=True)
class WeightState:
    """Estado de integridade de um arquivo de pesos."""

    key: str
    path: Path
    state: Literal["ok", "missing", "mismatch", "unpinned"]
    detail: str | None = None


class Registry:
    """Manifestos + pesos verificados + cache LRU de adaptadores carregados."""

    def __init__(
        self,
        manifest_dir: Path | None = None,
        weights_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        from radiologyai.config.settings import get_settings

        self.settings = settings or get_settings()
        self.manifest_dir = Path(manifest_dir or PACKAGE_MANIFEST_DIR)
        self.weights_dir = Path(weights_dir or self.settings.weights_dir)
        self._manifests: dict[str, ModelManifest] = {}
        self._manifest_sha: dict[str, str] = {}
        self._manifest_path: dict[str, Path] = {}
        self._loaded: OrderedDict[str, ModelAdapter] = OrderedDict()
        self._hash_cache: dict[tuple[str, int, int], str] = {}
        self._lock = threading.RLock()
        self._load_manifests()

    # ------------------------------------------------------------------ #
    # Manifestos
    # ------------------------------------------------------------------ #
    def _load_manifests(self) -> None:
        if not self.manifest_dir.is_dir():
            raise ManifestError(f"diretório de manifestos não encontrado: {self.manifest_dir}")
        for path in sorted(self.manifest_dir.glob("*.yaml")):
            manifest, digest = load_manifest(path)
            if manifest.id in self._manifests:
                raise ManifestError(f"id de modelo duplicado {manifest.id!r}")
            self._manifests[manifest.id] = manifest
            self._manifest_sha[manifest.id] = digest
            self._manifest_path[manifest.id] = path
        for manifest in self._manifests.values():
            for dep in manifest.depends_on:
                if dep not in self._manifests:
                    raise ManifestError(f"{manifest.id}: depende de modelo desconhecido {dep!r}")
        self.execution_order(self._manifests)  # levanta em ciclo de dependência

    def ids(self, status: str | None = None) -> list[str]:
        """Ids dos modelos, opcionalmente filtrados por status."""
        return [m.id for m in self._manifests.values() if status is None or m.status == status]

    def manifests(self) -> list[ModelManifest]:
        return list(self._manifests.values())

    def has(self, model_id: str) -> bool:
        return model_id in self._manifests

    def manifest(self, model_id: str) -> ModelManifest:
        """Manifesto de um modelo.

        Raises:
            ModelUnavailableError: id inexistente no registro.
        """
        try:
            return self._manifests[model_id]
        except KeyError:
            raise ModelUnavailableError(model_id, "modelo inexistente no registro") from None

    def manifest_sha256(self, model_id: str) -> str:
        self.manifest(model_id)
        return self._manifest_sha[model_id]

    def manifest_path(self, model_id: str) -> Path:
        self.manifest(model_id)
        return self._manifest_path[model_id]

    def execution_order(self, model_ids: Iterable[str]) -> list[str]:
        """Ordem topológica de ``model_ids`` mais todas as suas dependências.

        Raises:
            ManifestError: ciclo de dependência.
            ModelUnavailableError: id desconhecido.
        """
        order: list[str] = []
        state: dict[str, int] = {}  # 1 = visitando, 2 = concluído

        def visit(model_id: str, trail: tuple[str, ...]) -> None:
            if state.get(model_id) == 2:
                return
            if state.get(model_id) == 1:
                raise ManifestError("ciclo de dependência: " + " -> ".join((*trail, model_id)))
            state[model_id] = 1
            for dep in self.manifest(model_id).depends_on:
                visit(dep, (*trail, model_id))
            state[model_id] = 2
            order.append(model_id)

        for model_id in model_ids:
            visit(model_id, ())
        return order

    # ------------------------------------------------------------------ #
    # Pesos
    # ------------------------------------------------------------------ #
    def model_dir(self, model_id: str) -> Path:
        """Diretório dos pesos de um modelo: ``<weights_dir>/<model_id>/``."""
        self.manifest(model_id)
        return self.weights_dir / model_id

    def _hash(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self._hash_cache:
            self._hash_cache[key] = sha256_file(path)
        return self._hash_cache[key]

    def weight_states(self, model_id: str) -> list[WeightState]:
        """Estado de integridade de cada peso; nunca levanta por arquivo ausente."""
        manifest = self.manifest(model_id)
        states: list[WeightState] = []
        for weight in manifest.weights:
            path = self.model_dir(model_id) / weight.filename
            if not weight.sha256:
                states.append(WeightState(weight.key, path, "unpinned"))
            elif not path.is_file():
                states.append(WeightState(weight.key, path, "missing"))
            else:
                actual = self._hash(path)
                if actual != weight.sha256:
                    states.append(
                        WeightState(
                            weight.key,
                            path,
                            "mismatch",
                            f"esperado {weight.sha256}, obtido {actual}",
                        )
                    )
                else:
                    states.append(WeightState(weight.key, path, "ok"))
        return states

    def verify(self, model_id: str) -> ResolvedWeights:
        """Confere todos os pesos e devolve os caminhos verificados.

        Raises:
            ModelUnavailableError: planejado/aposentado, licença bloqueada por
                configuração, peso sem hash ou ausente.
            IntegrityCheckError: hash divergente.
        """
        manifest = self.manifest(model_id)
        if manifest.status != "available":
            reason = manifest.planned_reason or f"status é {manifest.status!r}"
            raise ModelUnavailableError(model_id, reason)
        if manifest.license.commercial_use == "no" and not self.settings.allow_noncommercial_models:
            raise ModelUnavailableError(
                model_id, "pesos de licença não comercial desabilitados pela configuração"
            )
        files: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        unsafe: dict[str, bool] = {}
        by_key = {w.key: w for w in manifest.weights}
        for state in self.weight_states(model_id):
            if state.state == "unpinned":
                raise ModelUnavailableError(model_id, f"peso {state.key!r} sem sha256 fixado")
            if state.state == "missing":
                raise ModelUnavailableError(
                    model_id,
                    f"peso {state.key!r} não encontrado em {state.path}; "
                    f"rode `radiologyai models pull {model_id}` (Registry.pull)",
                )
            if state.state == "mismatch":
                raise IntegrityCheckError(
                    model_id, f"peso {state.key!r} reprovado na integridade: {state.detail}"
                )
            weight = by_key[state.key]
            hashes[state.key] = weight.sha256
            unsafe[state.key] = weight.unsafe_pickle
            files[state.key] = (
                state.path if weight.extract == "none" else self._extracted_dir(model_id, weight)
            )
        return ResolvedWeights(files=files, sha256=hashes, unsafe_pickle=unsafe)

    def _extracted_dir(self, model_id: str, weight: WeightFile) -> Path:
        target = self.model_dir(model_id) / f"{weight.key}.extracted"
        marker = target / _EXTRACT_MARKER
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == weight.sha256:
            return target
        archive = self.model_dir(model_id) / weight.filename
        tmp = Path(tempfile.mkdtemp(dir=self.model_dir(model_id), prefix=".extract-"))
        try:
            safe_extract(archive, tmp, weight.extract, model_id=model_id)
            (tmp / _EXTRACT_MARKER).write_text(weight.sha256, encoding="utf-8")
            if target.exists():
                shutil.rmtree(target)
            Path(tmp).replace(target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return target

    def pull(
        self,
        model_id: str,
        *,
        client: httpx.Client | None = None,
        progress: ProgressFn | None = None,
    ) -> dict[str, tuple[str, int]]:
        """Baixa os pesos para ``<weights_dir>/<model_id>/`` e confere o sha256 fixado.

        Arquivo já presente e íntegro não é baixado de novo. Um download cujo
        hash diverge é apagado antes de levantar. Devolve ``{chave: (sha256, bytes)}``.

        Raises:
            ModelUnavailableError: modelo planejado/aposentado.
            ManifestError: peso sem sha256 fixado (use :func:`pin`).
            IntegrityCheckError: bytes baixados não conferem.
        """
        manifest = self.manifest(model_id)
        if manifest.status != "available":
            raise ModelUnavailableError(model_id, manifest.planned_reason or manifest.status)
        target_dir = self.model_dir(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, tuple[str, int]] = {}
        with _http_client(client) as http:
            for weight in manifest.weights:
                if not weight.sha256:
                    raise ManifestError(f"{model_id}: peso {weight.key!r} sem sha256 fixado")
                target = target_dir / weight.filename
                if target.is_file() and self._hash(target) == weight.sha256:
                    result[weight.key] = (weight.sha256, target.stat().st_size)
                    continue
                digest, size = _download(http, weight, target, progress, expected=weight.sha256)
                if digest != weight.sha256:
                    raise IntegrityCheckError(
                        model_id,
                        f"download de {weight.key!r} tem sha256 {digest}; "
                        f"o manifesto fixa {weight.sha256}",
                    )
                result[weight.key] = (digest, size)
        return result

    # ------------------------------------------------------------------ #
    # Adaptadores
    # ------------------------------------------------------------------ #
    def resolve_device(self) -> str:
        """Dispositivo efetivo: ``auto`` vira ``cuda`` se disponível, senão ``cpu``."""
        choice = self.settings.device
        if choice != "auto":
            return choice
        if importlib.util.find_spec("torch") is None:
            return "cpu"
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, model_id: str) -> ModelAdapter:
        """Adaptador carregado (cache LRU).

        Raises:
            ModelUnavailableError: planejado, pesos ausentes/divergentes, ou
                dependência opcional ausente para o adaptador.
        """
        with self._lock:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id]
            manifest = self.manifest(model_id)
            weights = self.verify(model_id)
            adapter = _instantiate(manifest)
            device = self.resolve_device()
            adapter.device = device
            try:
                adapter.load(weights, device)
            except BackendUnavailableError as exc:
                raise ModelUnavailableError(model_id, str(exc)) from exc
            adapter.loaded = True
            self._loaded[model_id] = adapter
            while len(self._loaded) > max(1, self.settings.max_loaded_models):
                _, evicted = self._loaded.popitem(last=False)
                evicted.unload()
                evicted.loaded = False
            return adapter

    def loaded_ids(self) -> list[str]:
        return list(self._loaded)

    def unload_all(self) -> None:
        with self._lock:
            while self._loaded:
                _, adapter = self._loaded.popitem()
                adapter.unload()
                adapter.loaded = False

    def provenance(self, model_id: str) -> ModelProvenance:
        """Proveniência do modelo para anexar ao resultado."""
        manifest = self.manifest(model_id)
        runtime = {"python": platform.python_version(), "device": self.resolve_device()}
        for dist in ("torch", "numpy", "pydicom"):
            try:
                runtime[dist] = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                continue
        return ModelProvenance(
            model_id=manifest.id,
            model_version=manifest.version,
            weights_sha256={w.key: w.sha256 for w in manifest.weights},
            manifest_sha256=self._manifest_sha[model_id],
            code_version=__version__,
            license_code=manifest.license.code,
            license_weights=manifest.license.weights,
            commercial_use=manifest.license.commercial_use,
            runtime=runtime,
        )


# ---------------------------------------------------------------------- #
# Auxiliares
# ---------------------------------------------------------------------- #
def _instantiate(manifest: ModelManifest) -> ModelAdapter:
    if not manifest.adapter:
        raise ModelUnavailableError(manifest.id, "manifesto não nomeia adaptador")
    module_name, _, class_name = manifest.adapter.partition(":")
    if importlib.util.find_spec(module_name) is None:
        raise ModelUnavailableError(manifest.id, f"módulo do adaptador inexistente: {module_name}")
    # Adaptadores importam dependências pesadas sob demanda (em ``load``) e
    # levantam BackendUnavailableError nomeando o extra; ``Registry.load`` a
    # converte em ModelUnavailableError. Nada é substituído.
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ManifestError(f"{manifest.id}: classe de adaptador {manifest.adapter!r} inexistente")
    if not (isinstance(cls, type) and issubclass(cls, ModelAdapter)):
        raise ManifestError(f"{manifest.id}: {manifest.adapter!r} não é ModelAdapter")
    adapter: ModelAdapter = cls(manifest)
    return adapter


class _http_client:  # noqa: N801 - gerenciador de contexto com nome de função
    """Usa o cliente dado ou cria (e fecha) um httpx.Client próprio."""

    def __init__(self, client: httpx.Client | None) -> None:
        self._given = client
        self._own: httpx.Client | None = None

    def __enter__(self) -> httpx.Client:
        if self._given is not None:
            return self._given
        if importlib.util.find_spec("httpx") is None:
            raise BackendUnavailableError(
                "httpx não instalado; instale o extra: pip install 'radiologyai[platform]'"
            )
        import httpx

        self._own = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=600.0))
        return self._own

    def __exit__(self, *exc: object) -> None:
        if self._own is not None:
            self._own.close()


def _download(
    http: httpx.Client,
    weight: WeightFile,
    target: Path,
    progress: ProgressFn | None,
    *,
    expected: str | None,
) -> tuple[str, int]:
    """Baixa para arquivo temporário; só move para o destino se o hash conferir."""
    digest = hashlib.sha256()
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".download-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh, http.stream("GET", weight.url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or weight.bytes or 0)
            for chunk in response.iter_bytes(1 << 20):
                fh.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if progress is not None:
                    progress(weight.key, size, total)
        hexdigest = digest.hexdigest()
        if expected is not None and hexdigest != expected:
            tmp.unlink(missing_ok=True)
            return hexdigest, size
        Path(tmp).replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return hexdigest, size


def safe_extract(archive: Path, dest: Path, kind: str, *, model_id: str = "?") -> None:
    """Extrai um arquivo compactado recusando caminho absoluto, ``..`` e links.

    Raises:
        IntegrityCheckError: membro escapa do destino ou é link.
        ManifestError: tipo de arquivo desconhecido.
    """
    root = dest.resolve()

    def check(name: str) -> None:
        if name.startswith(("/", "\\")) or ":" in name.split("/")[0]:
            raise IntegrityCheckError(model_id, f"membro com caminho absoluto: {name!r}")
        resolved = (root / name).resolve()
        if resolved != root and root not in resolved.parents:
            raise IntegrityCheckError(model_id, f"membro escapa do diretório de destino: {name!r}")

    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            for info in members:
                check(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise IntegrityCheckError(model_id, f"arquivo contém link: {info.filename!r}")
            zf.extractall(root, members=members)  # noqa: S202  # nosec B202 - membros validados acima
    elif kind == "tar":
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                check(member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise IntegrityCheckError(model_id, f"arquivo contém link: {member.name!r}")
            tf.extractall(root, filter="data")
    else:
        raise ManifestError(f"tipo de arquivo compactado desconhecido {kind!r}")


def pin(
    manifest_path: Path,
    weights_dir: Path,
    *,
    download: bool = True,
    client: httpx.Client | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, tuple[str, int]]:
    """Escreve no manifesto o sha256 e o tamanho calculados dos bytes REAIS.

    Para cada peso sem ``sha256``: usa o arquivo em
    ``<weights_dir>/<id>/<filename>`` se existir; senão (com ``download=True``)
    baixa da ``url``. Pesos já fixados são conferidos e nunca reescritos — um
    hash fixado divergente levanta. O manifesto resultante é validado antes de
    ser gravado.

    Raises:
        ManifestError: manifesto inválido antes ou depois do pin.
        ModelUnavailableError: arquivo ausente com ``download=False``.
        IntegrityCheckError: arquivo local diverge de um hash já fixado.
    """
    import yaml

    data: Any = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "id" not in data:
        raise ManifestError(f"{manifest_path}: manifesto sem id")
    model_id = str(data["id"])
    model_dir = Path(weights_dir) / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    pins: dict[str, tuple[str, int]] = {}
    with contextlib.ExitStack() as stack:
        http: httpx.Client | None = None
        for raw in data.get("weights", []) or []:
            weight = WeightFile.model_validate(raw)
            target = model_dir / weight.filename
            if target.is_file():
                digest, size = sha256_file(target), target.stat().st_size
            elif download:
                if http is None:
                    http = stack.enter_context(_http_client(client))
                digest, size = _download(http, weight, target, progress, expected=None)
            else:
                raise ModelUnavailableError(model_id, f"peso {weight.key!r} ausente em {target}")
            if weight.sha256 and weight.sha256 != digest:
                raise IntegrityCheckError(
                    model_id, f"{weight.key!r}: fixado {weight.sha256}, arquivo tem {digest}"
                )
            raw["sha256"], raw["bytes"] = digest, size
            pins[weight.key] = (digest, size)
    parse_manifest(data, origin=str(manifest_path))
    manifest_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    return pins
