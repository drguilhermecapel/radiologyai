"""Adaptador TotalSegmentator (TC e RM): nnU-Net real em subprocesso, offline, sem telemetria.

Um único adaptador serve todas as tarefas abertas do TotalSegmentator; o
manifesto (``config``) diz qual tarefa, que pastas de pesos (``DatasetXXX``)
ela usa, que volumes medir e de que modelo a montante depende o portão de
aplicabilidade.

Garantias (REQ-130, REQ-131, REQ-132):

* pesos: só os arquivos verificados pelo registro (sha256 do manifesto) são
  expostos ao TotalSegmentator, por uma árvore de *links* temporária montada em
  ``load``; o processo filho recusa rodar se faltar qualquer pasta esperada;
* rede: o filho roda com todos os proxies apontando para ``127.0.0.1:9`` e com
  ``send_usage_stats: false`` forçado; qualquer tentativa de download ou de
  estatística de uso falha em vez de sair do host;
* versão: a versão instalada do TotalSegmentator deve ser exatamente a do
  manifesto (``config.library_version``); divergência é falha;
* geometria: o mapa de rótulos devolvido deve ter a mesma grade e afim da
  entrada; senão é falha (nunca reamostragem silenciosa);
* volumes em mL = voxels x volume do voxel na grade de origem.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from radiologyai.analysis.base import (
    AdapterOutput,
    LabelMap,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
    ResolvedWeights,
)
from radiologyai.analysis.types import AbstentionReason, ImageData, Modality, QACheck, QAReport
from radiologyai.errors import BackendUnavailableError, InferenceError, ManifestError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.analysis.manifest import ModelManifest

__all__ = [
    "DEAD_PROXY",
    "DatasetSpec",
    "TotalSegConfig",
    "TotalSegmentatorAdapter",
    "TotalSegmentatorRunError",
    "VolumeSpec",
    "write_nifti",
]

DEAD_PROXY = "http://127.0.0.1:9"
_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
_RUNNER_MODULE = "radiologyai.analysis.adapters.ct_totalseg_runner"


class TotalSegmentatorRunError(InferenceError):
    """O processo do TotalSegmentator falhou, divergiu de versão ou devolveu geometria errada."""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetSpec(_Strict):
    """Uma pasta de pesos nnU-Net: ``<folder>/<trainer_dir>/{plans,dataset}.json, fold_0/...``."""

    folder: str
    trainer_dir: str
    checkpoint: str  # chave do peso no manifesto
    plans: str
    dataset: str


class VolumeSpec(_Strict):
    """Volume (mL) a medir sobre a união de rótulos da própria tarefa."""

    key: str
    labels: tuple[str, ...]
    require_complete: bool = False  # órgão: cortado pelo campo de visão => abstenção
    absent: Literal["zero", "abstain"] = "zero"  # lesão ausente = 0 mL; órgão ausente = abstenção


class UpstreamGate(_Strict):
    """Estrutura que precisa estar no campo de visão (mapa do modelo a montante)."""

    model: str
    any_of: tuple[str, ...]
    min_ml: float = Field(default=1.0, gt=0)


class TotalSegConfig(_Strict):
    task: str
    mode: Literal["default", "fast", "fastest"] = "default"
    library_version: str
    datasets: tuple[DatasetSpec, ...]
    volumes: tuple[VolumeSpec, ...] = ()
    upstream_gate: UpstreamGate | None = None
    postprocess: tuple[Literal["evans_index"], ...] = ()
    min_slices: int = Field(default=10, ge=2)
    timeout_s: int = Field(default=3600, ge=10)


def _device_arg(device: str) -> str:
    if device.startswith("cuda"):
        return "gpu" + device[4:]
    return device


def write_nifti(image: ImageData, path: Path) -> npt.NDArray[np.float64]:
    """Grava ``image`` (ZYX + afim LPS) como NIfTI (XYZ + afim RAS). Devolve a afim RAS.

    TC com valores inteiros cabe em int16 (HU); o resto vai em float32.
    """
    import nibabel as nib
    import numpy as np

    if image.affine_lps is None:
        raise TotalSegmentatorRunError("imagem sem afim: geometria ausente")
    ras: npt.NDArray[np.float64] = np.diag([-1.0, -1.0, 1.0, 1.0]) @ image.affine_lps
    data: npt.NDArray[Any] = np.transpose(image.pixels, (2, 1, 0))
    if (
        image.modality is Modality.CT
        and float(data.min()) >= -32768
        and float(data.max()) <= 32767
        and bool(np.all(np.equal(np.round(data), data)))
    ):
        data = data.astype(np.int16)
    else:
        data = data.astype(np.float32)
    nib_any: Any = nib
    img = nib_any.Nifti1Image(np.ascontiguousarray(data), ras)
    img.set_qform(ras, code=1)
    img.set_sform(ras, code=1)
    nib.save(img, str(path))
    return ras


class TotalSegmentatorAdapter(ModelAdapter):
    """Roda uma tarefa do TotalSegmentator num processo filho e devolve o mapa de rótulos."""

    #: comando do processo filho; ``None`` = ``python -m ...ct_totalseg_runner``.
    #: Testes substituem por um executável falso (nunca distribuído).
    command: ClassVar[tuple[str, ...] | None] = None

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        try:
            self.config = TotalSegConfig.model_validate(manifest.config)
        except ValueError as exc:
            raise ManifestError(
                f"{manifest.id}: config do TotalSegmentator inválida: {exc}"
            ) from exc
        self._root: Path | None = None

    # -- ciclo de vida ---------------------------------------------------- #
    def load(self, weights: ResolvedWeights, device: str) -> None:
        for module, extra in (("nibabel", "volumetric"), ("numpy", "volumetric")):
            if importlib.util.find_spec(module) is None:
                raise BackendUnavailableError(
                    f"{module} não instalado; instale o extra: pip install 'radiologyai[{extra}]'"
                )
        if self.command is None and importlib.util.find_spec("totalsegmentator") is None:
            raise BackendUnavailableError(
                "TotalSegmentator não instalado; instale o extra: "
                "pip install 'radiologyai[volumetric]'"
            )
        root = Path(tempfile.mkdtemp(prefix=f"medai-{self.manifest.id}-"))
        tree = root / "weights"
        for spec in self.config.datasets:
            base = tree / spec.folder / spec.trainer_dir
            (base / "fold_0").mkdir(parents=True)
            for key, rel in (
                (spec.checkpoint, "fold_0/checkpoint_final.pth"),
                (spec.plans, "plans.json"),
                (spec.dataset, "dataset.json"),
            ):
                try:
                    source = weights.path(key)
                except KeyError:
                    shutil.rmtree(root, ignore_errors=True)
                    raise ManifestError(
                        f"{self.manifest.id}: config cita peso {key!r} não declarado"
                    ) from None
                (base / rel).symlink_to(Path(source).resolve())
        (root / "home").mkdir()
        (root / "work").mkdir()
        self._root = root
        self.device = device

    def unload(self) -> None:
        if self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)
            self._root = None

    # -- portões ---------------------------------------------------------- #
    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        checks = list(base.checks)
        reasons = list(base.reasons)
        if len(inputs.images) == 1:
            image = inputs.images[0]
            geo = image.spacing_mm is not None and image.affine_lps is not None
            checks.append(QACheck(name="geometry", passed=geo))
            if not geo:
                reasons.append(AbstentionReason.GEOMETRY)
            n = int(image.pixels.shape[0]) if image.pixels.ndim == 3 else 0
            ok_n = n >= self.config.min_slices
            checks.append(
                QACheck(
                    name="n_slices",
                    passed=ok_n,
                    value=float(n),
                    threshold=float(self.config.min_slices),
                )
            )
            if not ok_n:
                reasons.append(AbstentionReason.INPUT)
            gate = self.config.upstream_gate
            if gate is not None and geo:
                present = self._upstream_volume_ml(inputs, gate)
                ok = present is not None and present >= gate.min_ml
                checks.append(
                    QACheck(
                        name=f"upstream:{'|'.join(gate.any_of)}",
                        passed=ok,
                        value=present,
                        threshold=gate.min_ml,
                        detail=None if ok else "estrutura-alvo ausente do campo de visão",
                    )
                )
                if not ok:
                    reasons.append(
                        AbstentionReason.UPSTREAM if present is None else AbstentionReason.BODY_PART
                    )
        unique = tuple(dict.fromkeys(reasons))
        return QAReport(applicable=not unique, checks=tuple(checks), reasons=unique)

    @staticmethod
    def _upstream_volume_ml(inputs: ModelInputs, gate: UpstreamGate) -> float | None:
        import numpy as np

        from radiologyai.analysis.biomarkers.geometry import voxel_volume_ml

        output = inputs.upstream.get(gate.model)
        if output is None or not output.labelmaps:
            return None
        labelmap = output.labelmaps[0]
        wanted = [i for i, name in labelmap.labels.items() if name in gate.any_of]
        spacing = inputs.images[0].spacing_mm
        if spacing is None:
            return None
        count = int(np.isin(labelmap.array, wanted).sum()) if wanted else 0
        return round(count * voxel_volume_ml(spacing), 3)

    # -- inferência ------------------------------------------------------- #
    def _command(self) -> list[str]:
        if self.command is not None:
            return list(self.command)
        return [sys.executable, "-m", _RUNNER_MODULE]

    def _env(self) -> dict[str, str]:
        assert self._root is not None
        env = {k: v for k, v in os.environ.items() if not k.startswith("nnUNet_")}
        env.update(dict.fromkeys(_PROXY_VARS, DEAD_PROXY))
        env["NO_PROXY"] = env["no_proxy"] = ""
        env["TOTALSEG_HOME_DIR"] = str(self._root / "home")
        env["TOTALSEG_WEIGHTS_PATH"] = str(self._root / "weights")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def run_task(self, image: ImageData) -> tuple[npt.NDArray[np.uint8], dict[str, Any]]:
        """Roda o processo filho e devolve (mapa de rótulos ZYX, JSON lateral)."""
        import nibabel as nib
        import numpy as np

        if self._root is None:
            raise TotalSegmentatorRunError(f"{self.manifest.id}: adaptador não carregado")
        work = Path(tempfile.mkdtemp(dir=self._root / "work"))
        try:
            src = work / "input.nii.gz"
            ras = write_nifti(image, src)
            out = work / "seg.nii.gz"
            side = work / "sidecar.json"
            cmd = [
                *self._command(),
                "--input",
                str(src),
                "--output",
                str(out),
                "--sidecar",
                str(side),
                "--task",
                self.config.task,
                "--mode",
                self.config.mode,
                "--device",
                _device_arg(self.device),
            ]
            for spec in self.config.datasets:
                cmd += ["--expect", spec.folder]
            try:
                proc = subprocess.run(  # noqa: S603 - comando fixo do manifesto/registro
                    cmd,
                    env=self._env(),
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TotalSegmentatorRunError(
                    f"{self.manifest.id}: excedeu {self.config.timeout_s} s"
                ) from exc
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-2000:]
                raise TotalSegmentatorRunError(
                    f"{self.manifest.id}: processo terminou com código {proc.returncode}: {tail}"
                )
            if not out.is_file() or not side.is_file():
                raise TotalSegmentatorRunError(f"{self.manifest.id}: saída ausente")
            sidecar: dict[str, Any] = json.loads(side.read_text(encoding="utf-8"))
            version = str(sidecar.get("totalsegmentator_version"))
            if version != self.config.library_version:
                raise TotalSegmentatorRunError(
                    f"{self.manifest.id}: TotalSegmentator {version} instalado; manifesto fixa "
                    f"{self.config.library_version}"
                )
            if sidecar.get("task") != self.config.task:
                raise TotalSegmentatorRunError(f"{self.manifest.id}: tarefa divergente no filho")
            seg: Any = nib.load(str(out))
            data = np.asarray(seg.dataobj)
            if data.shape != image.pixels.shape[::-1]:
                raise TotalSegmentatorRunError(
                    f"{self.manifest.id}: grade da saída {data.shape} difere da entrada "
                    f"{image.pixels.shape[::-1]} (x, y, z)"
                )
            if not np.allclose(np.asarray(seg.affine), ras, atol=1e-3):
                raise TotalSegmentatorRunError(
                    f"{self.manifest.id}: afim da saída difere da entrada"
                )
            if data.min() < 0 or data.max() > 255:
                raise TotalSegmentatorRunError(f"{self.manifest.id}: rótulos fora de 0..255")
            labels: npt.NDArray[np.uint8] = np.ascontiguousarray(
                np.transpose(data, (2, 1, 0)).astype(np.uint8)
            )
            return labels, sidecar
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        image = inputs.single
        labels, sidecar = self.run_task(image)
        class_map = {int(k): str(v) for k, v in sidecar["class_map"].items()}
        names = {v: k for k, v in class_map.items()}
        for spec in self.config.volumes:
            unknown = [n for n in spec.labels if n not in names]
            if unknown:
                raise ManifestError(f"{self.manifest.id}: volume {spec.key!r} cita {unknown}")
        labelmap = LabelMap(name=self.config.task, array=labels, labels=class_map)
        measurements = [self._volume(image, labels, names, spec) for spec in self.config.volumes]
        if "evans_index" in self.config.postprocess:
            measurements.append(self._evans(inputs, image, labels, names))
        return AdapterOutput(
            labelmaps=[labelmap],
            measurements=measurements,
            extras={"runner": sidecar},
        )

    def _volume(
        self,
        image: ImageData,
        labels: npt.NDArray[np.uint8],
        names: dict[str, int],
        spec: VolumeSpec,
    ) -> RawMeasurement:
        import numpy as np

        from radiologyai.analysis.biomarkers.geometry import (
            component_volumes_ml,
            touches_border,
            voxel_volume_ml,
        )

        assert image.spacing_mm is not None
        method = (
            f"voxels de {'+'.join(spec.labels)} (TotalSegmentator {self.config.task}) x volume "
            "do voxel na grade de origem"
        )
        mask = np.isin(labels, [names[n] for n in spec.labels])
        source = [self.config.task]
        if not mask.any() and spec.absent == "abstain":
            return RawMeasurement(
                key=spec.key,
                value=None,
                unit="mL",
                method=method,
                derived_from=source,
                abstention=AbstentionReason.ACQUISITION,
                note="estrutura ausente do campo de visão",
            )
        if spec.require_complete and touches_border(mask):
            return RawMeasurement(
                key=spec.key,
                value=None,
                unit="mL",
                method=method,
                derived_from=source,
                abstention=AbstentionReason.ACQUISITION,
                note="estrutura cortada pelo campo de visão: volume seria limite inferior",
            )
        components = component_volumes_ml(mask, image.spacing_mm) if mask.any() else []
        note = None
        if components:
            largest = ", ".join(f"{v:.2f}" for v in components[:5])
            note = f"{len(components)} componente(s) conexo(s); maiores (mL): {largest}"
            if not spec.require_complete and touches_border(mask):
                note += "; segmentação encosta no limite do campo de visão"
        return RawMeasurement(
            key=spec.key,
            value=round(float(mask.sum()) * voxel_volume_ml(image.spacing_mm), 3),
            unit="mL",
            method=method,
            derived_from=source,
            preconditions={"n_components": str(len(components))},
            note=note,
        )

    def _evans(
        self,
        inputs: ModelInputs,
        image: ImageData,
        labels: npt.NDArray[np.uint8],
        names: dict[str, int],
    ) -> RawMeasurement:
        import numpy as np

        from radiologyai.analysis.biomarkers.ct import EVANS_KEY, EVANS_METHOD, evans_index

        gate = self.config.upstream_gate
        upstream = inputs.upstream.get(gate.model) if gate else None
        if upstream is None or not upstream.labelmaps or image.spacing_mm is None:
            return RawMeasurement(
                key=EVANS_KEY,
                value=None,
                unit="1",
                method=EVANS_METHOD,
                abstention=AbstentionReason.UPSTREAM,
                note="mapa 'brain' do modelo a montante ausente",
            )
        total = upstream.labelmaps[0]
        brain_idx = [i for i, n in total.labels.items() if n == "brain"]
        if total.array.shape != labels.shape:
            raise TotalSegmentatorRunError("mapa a montante em grade diferente")
        horns = np.isin(
            labels,
            [names["ventricle_frontal_horn_left"], names["ventricle_frontal_horn_right"]],
        )
        brain = np.isin(total.array, brain_idx)
        assert image.affine_lps is not None
        spacing = image.spacing_mm
        return evans_index(
            image.pixels,
            (spacing[0], spacing[1], spacing[2]),
            image.affine_lps,
            image.meta,
            horns,
            brain,
            source=self.config.task,
        )
