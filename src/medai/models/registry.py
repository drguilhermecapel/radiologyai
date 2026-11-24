"""Model registry: manifests, weight integrity, adapter loading.

FROZEN CONTRACT (public methods).

Rules enforced here
-------------------
* A model with ``status: planned`` can never be loaded.
* Weights are downloaded only by :meth:`Registry.pull`; they are verified against
  the SHA-256 pinned in the manifest before any adapter touches them. A missing or
  mismatching file raises :class:`ModelUnavailable` - callers must not fall back.
* Adapters are imported only from ``medai.models.adapters`` / ``medai.probes``.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import tarfile
import tempfile
import threading
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import torch
import yaml

from medai import __version__
from medai.core.artifacts import sha256_file
from medai.core.errors import IntegrityError, ManifestError, ModelUnavailable
from medai.core.settings import Settings, get_settings
from medai.core.types import ModelProvenance
from medai.models.base import ModelAdapter, ResolvedWeights
from medai.models.manifest import ModelManifest, WeightFile, load_manifest

PACKAGE_MANIFEST_DIR = Path(__file__).parent / "manifests"
ProgressFn = Callable[[str, int, int], None]  # (weight key, bytes done, bytes total)


@dataclass(frozen=True, slots=True)
class WeightState:
    key: str
    path: Path
    state: Literal["ok", "missing", "mismatch", "unpinned"]
    detail: str | None = None


class Registry:
    def __init__(
        self,
        manifest_dir: Path | None = None,
        weights_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.manifest_dir = Path(manifest_dir or PACKAGE_MANIFEST_DIR)
        self.weights_dir = Path(weights_dir or self.settings.weights_path)
        self._manifests: dict[str, ModelManifest] = {}
        self._manifest_sha: dict[str, str] = {}
        self._manifest_path: dict[str, Path] = {}
        self._loaded: OrderedDict[str, ModelAdapter] = OrderedDict()
        self._verified: dict[tuple[str, int, int], str] = {}  # (path, size, mtime_ns) -> sha256
        self._lock = threading.RLock()
        self._load_manifests()

    # ------------------------------------------------------------------ #
    # Manifests
    # ------------------------------------------------------------------ #
    def _load_manifests(self) -> None:
        if not self.manifest_dir.is_dir():
            raise ManifestError(f"manifest directory not found: {self.manifest_dir}")
        for path in sorted(self.manifest_dir.glob("*.yaml")):
            manifest, digest = load_manifest(path)
            if manifest.id in self._manifests:
                raise ManifestError(f"duplicate model id {manifest.id!r}")
            self._manifests[manifest.id] = manifest
            self._manifest_sha[manifest.id] = digest
            self._manifest_path[manifest.id] = path
        for manifest in self._manifests.values():
            for dep in manifest.depends_on:
                if dep not in self._manifests:
                    raise ManifestError(f"{manifest.id}: depends on unknown model {dep!r}")
        self.execution_order(self._manifests)  # raises on dependency cycles

    def ids(self, status: str | None = None) -> list[str]:
        return [m.id for m in self._manifests.values() if status is None or m.status == status]

    def manifests(self) -> list[ModelManifest]:
        return list(self._manifests.values())

    def manifest(self, model_id: str) -> ModelManifest:
        try:
            return self._manifests[model_id]
        except KeyError:
            raise ModelUnavailable(model_id, "no such model in the registry") from None

    def manifest_sha256(self, model_id: str) -> str:
        self.manifest(model_id)
        return self._manifest_sha[model_id]

    def manifest_path(self, model_id: str) -> Path:
        self.manifest(model_id)
        return self._manifest_path[model_id]

    def execution_order(self, model_ids: Iterable[str]) -> list[str]:
        """Topological order of ``model_ids`` plus all their dependencies."""
        order: list[str] = []
        state: dict[str, int] = {}  # 1 = visiting, 2 = done

        def visit(model_id: str, trail: tuple[str, ...]) -> None:
            if state.get(model_id) == 2:
                return
            if state.get(model_id) == 1:
                raise ManifestError("dependency cycle: " + " -> ".join((*trail, model_id)))
            state[model_id] = 1
            for dep in self.manifest(model_id).depends_on:
                visit(dep, (*trail, model_id))
            state[model_id] = 2
            order.append(model_id)

        for model_id in model_ids:
            visit(model_id, ())
        return order

    # ------------------------------------------------------------------ #
    # Weights
    # ------------------------------------------------------------------ #
    def model_dir(self, model_id: str) -> Path:
        return self.weights_dir / model_id

    def _hash(self, path: Path) -> str:
        stat = path.stat()
        cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        if cache_key not in self._verified:
            self._verified[cache_key] = sha256_file(path)
        return self._verified[cache_key]

    def weight_states(self, model_id: str) -> list[WeightState]:
        """Integrity state of each weight file; never raises for missing files."""
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
                            weight.key, path, "mismatch", f"expected {weight.sha256}, got {actual}"
                        )
                    )
                else:
                    states.append(WeightState(weight.key, path, "ok"))
        return states

    def verify(self, model_id: str) -> ResolvedWeights:
        manifest = self.manifest(model_id)
        if manifest.status != "available":
            reason = manifest.planned_reason or f"status is {manifest.status}"
            raise ModelUnavailable(model_id, reason)
        if manifest.license.commercial_use == "no" and not self.settings.allow_noncommercial_models:
            raise ModelUnavailable(model_id, "non-commercial weights are disabled by configuration")
        files: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        by_key = {w.key: w for w in manifest.weights}
        for state in self.weight_states(model_id):
            if state.state == "unpinned":
                raise ModelUnavailable(model_id, f"weight {state.key!r} has no pinned sha256")
            if state.state == "missing":
                raise ModelUnavailable(
                    model_id, f"weight {state.key!r} not found; run `medai models pull {model_id}`"
                )
            if state.state == "mismatch":
                raise IntegrityError(
                    model_id, f"weight {state.key!r} failed integrity check: {state.detail}"
                )
            weight = by_key[state.key]
            hashes[state.key] = weight.sha256
            files[state.key] = (
                state.path if weight.extract == "none" else self._extracted_dir(model_id, weight)
            )
        return ResolvedWeights(files=files, sha256=hashes)

    def _extracted_dir(self, model_id: str, weight: WeightFile) -> Path:
        target = self.model_dir(model_id) / f"{weight.key}.extracted"
        marker = target / ".medai-archive-sha256"
        if marker.is_file() and marker.read_text().strip() == weight.sha256:
            return target
        archive = self.model_dir(model_id) / weight.filename
        tmp = Path(tempfile.mkdtemp(dir=self.model_dir(model_id), prefix=".extract-"))
        try:
            _safe_extract(archive, tmp, weight.extract)
            (tmp / ".medai-archive-sha256").write_text(weight.sha256)
            if target.exists():
                _rmtree(target)
            os.replace(tmp, target)
        except BaseException:
            _rmtree(tmp)
            raise
        return target

    def pull(
        self,
        model_id: str,
        *,
        pin: bool = False,
        client: httpx.Client | None = None,
        progress: ProgressFn | None = None,
    ) -> dict[str, tuple[str, int]]:
        """Download weights and check them against the pinned SHA-256.

        Returns ``{key: (sha256, bytes)}``. With ``pin=True`` unpinned files are
        accepted and their real hash is returned so that a tool can write it into
        the manifest; without it, an unpinned manifest is refused.
        """
        manifest = self.manifest(model_id)
        if manifest.status == "planned":
            raise ModelUnavailable(model_id, manifest.planned_reason or "planned")
        self.model_dir(model_id).mkdir(parents=True, exist_ok=True)
        own_client = client is None
        http = client or httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)
        )
        result: dict[str, tuple[str, int]] = {}
        try:
            for weight in manifest.weights:
                if not weight.sha256 and not pin:
                    raise ManifestError(f"{model_id}: weight {weight.key!r} is not pinned")
                target = self.model_dir(model_id) / weight.filename
                if target.is_file() and weight.sha256 and self._hash(target) == weight.sha256:
                    result[weight.key] = (weight.sha256, target.stat().st_size)
                    continue
                digest, size = _download(http, weight, target, progress)
                if weight.sha256 and digest != weight.sha256:
                    target.unlink(missing_ok=True)
                    raise IntegrityError(
                        model_id,
                        f"downloaded {weight.key!r} has sha256 {digest}, manifest pins {weight.sha256}",
                    )
                result[weight.key] = (digest, size)
        finally:
            if own_client:
                http.close()
        return result

    # ------------------------------------------------------------------ #
    # Adapters
    # ------------------------------------------------------------------ #
    def resolve_device(self) -> torch.device:
        choice = self.settings.device
        if choice == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(choice)

    def load(self, model_id: str) -> ModelAdapter:
        """Return a loaded adapter (LRU-cached). Raises :class:`ModelUnavailable`."""
        with self._lock:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id]
            manifest = self.manifest(model_id)
            weights = self.verify(model_id)
            adapter = _instantiate(manifest)
            adapter.device = self.resolve_device()
            adapter.load(weights, adapter.device)
            adapter._loaded = True
            self._loaded[model_id] = adapter
            while len(self._loaded) > max(1, self.settings.max_loaded_models):
                _, evicted = self._loaded.popitem(last=False)
                evicted.unload()
            return adapter

    def unload_all(self) -> None:
        with self._lock:
            while self._loaded:
                _, adapter = self._loaded.popitem()
                adapter.unload()

    def provenance(self, model_id: str) -> ModelProvenance:
        manifest = self.manifest(model_id)
        return ModelProvenance(
            model_id=manifest.id,
            model_version=manifest.version,
            weights_sha256={w.key: w.sha256 for w in manifest.weights},
            manifest_sha256=self._manifest_sha[model_id],
            code_version=__version__,
            license_weights=manifest.license.weights,
            commercial_use=manifest.license.commercial_use,
            runtime={
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(self.resolve_device()),
            },
        )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _instantiate(manifest: ModelManifest) -> ModelAdapter:
    if not manifest.adapter:
        raise ModelUnavailable(manifest.id, "manifest names no adapter")
    module_name, _, class_name = manifest.adapter.partition(":")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except ImportError as exc:
        raise ModelUnavailable(
            manifest.id, f"adapter import failed ({exc}); an optional dependency may be missing"
        ) from exc
    except AttributeError as exc:
        raise ManifestError(f"{manifest.id}: adapter class {manifest.adapter!r} not found") from exc
    if not (isinstance(cls, type) and issubclass(cls, ModelAdapter)):
        raise ManifestError(f"{manifest.id}: {manifest.adapter!r} is not a ModelAdapter")
    return cls(manifest)


def _download(
    http: httpx.Client, weight: WeightFile, target: Path, progress: ProgressFn | None
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".download-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh, http.stream("GET", weight.url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", weight.bytes or 0))
            for chunk in response.iter_bytes(1 << 20):
                fh.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if progress:
                    progress(weight.key, size, total)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size


def _safe_extract(archive: Path, dest: Path, kind: str) -> None:
    """Extract an archive refusing absolute paths, parent traversal and links."""
    root = dest.resolve()

    def check(name: str) -> None:
        resolved = (root / name).resolve()
        if resolved != root and root not in resolved.parents:
            raise IntegrityError(archive.name, f"archive member escapes target directory: {name!r}")

    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                check(info.filename)
            zf.extractall(root)  # noqa: S202 - members validated above
    elif kind == "tar":
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                check(member.name)
                if member.issym() or member.islnk():
                    raise IntegrityError(archive.name, f"archive contains a link: {member.name!r}")
            tf.extractall(root, filter="data")
    else:  # pragma: no cover - guarded by the manifest schema
        raise ManifestError(f"unknown archive kind {kind!r}")


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def write_pins(manifest_path: Path, pins: dict[str, tuple[str, int]]) -> None:
    """Write real SHA-256 / byte sizes into a manifest file (used by `medai models pin`)."""
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for weight in data.get("weights", []):
        if weight["key"] in pins:
            weight["sha256"], weight["bytes"] = pins[weight["key"]]
    ModelManifest.model_validate(data)
    manifest_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
