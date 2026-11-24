"""Pré-condições de aquisição de TC lidas dos metadados (``ImageMeta``).

Nada aqui adivinha: um fato ausente é ``unknown``, e quem exige o fato se abstém.

* Fase de contraste: só ``ImageMeta.contrast_phase`` explícito (definido por um
  humano ou por um classificador de fase validado) conta como confirmado.
  ``ContrastBolusAgent`` preenchido prova que HOUVE contraste, mas não diz a
  fase; tag ausente/vazia NÃO prova exame sem contraste (a fonte verificada diz
  "phase gate, not just a DICOM tag").
* Kernel: classificação conservadora por fabricante; o que não é reconhecido é
  ``unknown`` (regra de engenharia, documentada em ``KERNEL_RULE``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from radiologyai.analysis.types import AbstentionReason

if TYPE_CHECKING:
    from radiologyai.analysis.types import ImageMeta

__all__ = [
    "KERNEL_RULE",
    "Gate",
    "Phase",
    "acquisition_facts",
    "classify_kernel",
    "contrast_phase",
    "require_adult",
    "require_kvp",
    "require_phase",
    "require_smooth_kernel",
    "require_thin_slices",
]

Phase = Literal[
    "none",
    "arterial",
    "portal_venous",
    "delayed",
    "contrast_unknown_phase",
    "none_by_tag",
    "unknown",
]
KernelClass = Literal["smooth", "sharp", "unknown"]

KERNEL_RULE = (
    "kernel suave/padrão reconhecido: GE STANDARD/SOFT; Siemens B/Br/I/Q/J com número <= 45; "
    "Philips A/B/C; Canon/Toshiba FC01-FC18. Qualquer outro valor é 'unknown' (regra de "
    "engenharia do projeto, não limite publicado)."
)

_NO_CONTRAST = re.compile(
    r"(?:^|[^A-Z])(?:NONE|NO|W/?O|WITHOUT|SEM|NATIVE?|NATIVO|PLAIN)(?:[^A-Z]|$)"
)
_SIEMENS = re.compile(r"^\s*(?:BR|B|I|Q|J)(\d{2})[A-Z]?\b")
_TOSHIBA = re.compile(r"^\s*FC(\d{2})\b")
_SHARP_WORDS = ("LUNG", "BONE", "SHARP", "DETAIL", "EDGE", "HIGH", "BONEPLUS", "YA", "YB", "YC")


def classify_kernel(kernel: str | None) -> KernelClass:
    """Classe do kernel de reconstrução (ver :data:`KERNEL_RULE`)."""
    if not kernel:
        return "unknown"
    text = str(kernel).strip().upper()
    first = text.split("\\")[0].strip()
    if first in ("STANDARD", "SOFT", "STND", "STD"):
        return "smooth"
    if any(word == first or first.startswith(word) for word in _SHARP_WORDS):
        return "sharp"
    if m := _SIEMENS.match(first):
        return "smooth" if int(m.group(1)) <= 45 else "sharp"
    if m := _TOSHIBA.match(first):
        number = int(m.group(1))
        if 1 <= number <= 18:
            return "smooth"
        return "sharp" if number >= 30 else "unknown"
    if first in ("A", "B", "C"):
        return "smooth"
    if first in ("D", "E", "L"):
        return "sharp"
    return "unknown"


def agent_says_no_contrast(agent: str | None) -> bool:
    """O texto de ``ContrastBolusAgent`` NEGA contraste (ex.: ``W/O_IV_CONTRAST``, ``NONE``)?"""
    return bool(agent) and bool(_NO_CONTRAST.search(str(agent).upper()))


def contrast_phase(meta: ImageMeta) -> Phase:
    """Fase de contraste: confirmada (``ImageMeta.contrast_phase``) ou o que o cabeçalho sugere.

    * ``none_by_tag``: o cabeçalho NEGA contraste (``W/O_IV_CONTRAST``), mas a fonte
      verificada exige confirmação além da tag; não satisfaz ``require_phase("none")``.
    * ``contrast_unknown_phase``: o cabeçalho nomeia um agente; fase desconhecida.
    * ``unknown``: nada declarado (tag ausente NÃO prova exame sem contraste).
    """
    agent = (meta.contrast_agent or "").strip()
    negated = agent_says_no_contrast(agent)
    declared = meta.contrast_phase
    if declared is not None and declared != "unknown":
        if declared == "none" and agent and not negated:
            # Declaração conflita com o cabeçalho: não há fato confirmado.
            return "unknown"
        return declared
    if negated:
        return "none_by_tag"
    return "contrast_unknown_phase" if agent else "unknown"


def acquisition_facts(meta: ImageMeta) -> dict[str, str]:
    """Fatos de aquisição usados pelos biomarcadores (texto, para ``preconditions``)."""

    def show(value: object) -> str:
        return "ausente" if value is None or value == "" else str(value)

    return {
        "contrast_phase": contrast_phase(meta),
        "contrast_agent_tag": show(meta.contrast_agent),
        "kvp": show(meta.kvp),
        "convolution_kernel": show(meta.convolution_kernel),
        "kernel_class": classify_kernel(meta.convolution_kernel),
        "slice_thickness_mm": show(meta.slice_thickness_mm),
        "series_description": show(meta.series_description),
        "patient_age_years": show(meta.patient_age_years),
    }


@dataclass(frozen=True, slots=True)
class Gate:
    """Resultado de uma pré-condição: ``ok`` ou motivo + explicação."""

    ok: bool
    reason: AbstentionReason | None = None
    note: str | None = None
    facts: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def passed() -> Gate:
        return Gate(ok=True)

    @staticmethod
    def failed(reason: AbstentionReason, note: str) -> Gate:
        return Gate(ok=False, reason=reason, note=note)


def require_phase(meta: ImageMeta, *allowed: Phase) -> Gate:
    """Exige fase de contraste CONFIRMADA dentre ``allowed``."""
    phase = contrast_phase(meta)
    if phase in allowed:
        return Gate.passed()
    return Gate.failed(
        AbstentionReason.ACQUISITION,
        f"fase de contraste {phase!r}; exigida {'/'.join(allowed)} CONFIRMADA (humano ou "
        "classificador de fase); tag DICOM ausente ou só textual não basta",
    )


def require_kvp(meta: ImageMeta, kvp: float, *, tolerance: float = 0.5) -> Gate:
    """Exige tensão de tubo igual a ``kvp`` (os limiares publicados são a 120 kV)."""
    if meta.kvp is None:
        return Gate.failed(AbstentionReason.ACQUISITION, f"kVp ausente; exigido {kvp:g} kV")
    if abs(meta.kvp - kvp) > tolerance:
        return Gate.failed(
            AbstentionReason.ACQUISITION, f"kVp {meta.kvp:g}; limiar publicado vale a {kvp:g} kV"
        )
    return Gate.passed()


def require_thin_slices(meta: ImageMeta, max_mm: float) -> Gate:
    """Exige espessura de corte conhecida e <= ``max_mm``."""
    thickness = meta.slice_thickness_mm
    if thickness is None:
        return Gate.failed(AbstentionReason.ACQUISITION, "espessura de corte ausente")
    if thickness > max_mm + 1e-6:
        return Gate.failed(
            AbstentionReason.ACQUISITION, f"espessura {thickness:g} mm > {max_mm:g} mm exigidos"
        )
    return Gate.passed()


def require_smooth_kernel(meta: ImageMeta) -> Gate:
    """Exige kernel suave/padrão reconhecido (ver :data:`KERNEL_RULE`)."""
    klass = classify_kernel(meta.convolution_kernel)
    if klass == "smooth":
        return Gate.passed()
    return Gate.failed(
        AbstentionReason.ACQUISITION,
        f"kernel {meta.convolution_kernel!r} classificado como {klass!r}; exigido suave/padrão",
    )


def require_adult(meta: ImageMeta) -> Gate:
    """Exige idade conhecida >= 18 anos (limiares derivados em adultos)."""
    age = meta.patient_age_years
    if age is None:
        return Gate.failed(
            AbstentionReason.POPULATION, "idade ausente: população adulta não verificável"
        )
    if age < 18:
        return Gate.failed(AbstentionReason.POPULATION, f"idade {age:g} anos < 18")
    return Gate.passed()


def first_failure(*gates: Gate) -> Gate:
    """Primeira pré-condição que falhou, ou ``Gate.passed()``."""
    for gate in gates:
        if not gate.ok:
            return gate
    return Gate.passed()
