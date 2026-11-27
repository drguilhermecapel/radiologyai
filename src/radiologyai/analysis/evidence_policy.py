"""Política de decisão lastreada em evidência medida (pontos do protocolo).

Para um achado cujo trio (condição, modelo, rótulo) tem nível A, B ou C em
``levels.json``, os pontos de operação vêm SOMENTE do registro de evidência que
determina esse nível (``LevelEntry.basis``). Pontos do manifesto nunca decidem
um achado A/B/C nesta política; nível D mantém o comportamento anterior (pontos
do manifesto, achado exploratório fora do laudo).

Regra de escolha do registro determinante (determinística), entre os registros
da base que existem em ``evidence/`` e pertencem ao trio:

1. maior contribuição de nível — o nível que o registro sustentaria sozinho por
   :func:`~radiologyai.catalog.schema.compute_level` (A > B > C > D);
2. maior ``n_total`` (registro sem ``n_total`` conta como 0);
3. menor caminho do arquivo (relativo à raiz do repositório), em ordem lexicográfica.

Só o registro escolhido é usado. Se ele não serve (registro publicado sem
pontos, outro protocolo, pesos diferentes dos do manifesto, escala diferente da
emitida pelo adaptador), o achado fica ``score_only`` com aviso; a política não
recorre a outro registro nem ao manifesto.

Espaço do escore. ``evaluate_binary`` ajusta os limiares na escala BRUTA das
predições (``report.score_scale``: ``"logit"`` ou ``"probability"``); a
temperatura do relatório é ajustada à parte e usada só para o ECE — nunca é
aplicada aos limiares. O ponto do registro declara ``on`` (``"logit"`` ou
``"sigmoid"``). O ponto só é usado quando ``on`` coincide exatamente com a escala
que o adaptador emite (``score_kind`` do manifesto) e com ``report.score_scale``,
e quando o limiar é idêntico ao do relatório. Nenhuma conversão é feita: escala
diferente vira abstenção (``score_only`` + aviso), nunca limiar aproximado.

Suspensões de decisão (``evidence/decision_holds.yaml``, REQ-193). O responsável pode
suspender a decisão de um trio cujo nível é sustentado pela regra mas que uma análise
posterior mostrou frágil. A suspensão só retira decisão (o achado fica ``score_only`` com o
motivo); nunca altera o nível do catálogo nem promove nada. Cada suspensão traz motivo,
data, quem decidiu e a referência que a justifica.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from radiologyai.analysis.policy import PointsRequest, PointsResolution
from radiologyai.analysis.types import ScoreKind
from radiologyai.catalog.schema import EvidenceLevel, compute_level

if TYPE_CHECKING:
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.evidence.store import LoadedEvidence

__all__ = ["HOLDS_FILE", "DecisionHold", "EvidencePolicy", "load_holds", "select_basis"]

HOLDS_FILE = "decision_holds.yaml"


class DecisionHold(BaseModel):
    """Decisão suspensa para um trio (condição, modelo, rótulo): só escore."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_id: str
    model_id: str
    label: str
    reason: str
    decided_on: date
    decided_by: str
    evidence_ref: str


class _HoldsFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["radiologyai.evidence/decision-holds-v1"]
    holds: tuple[DecisionHold, ...] = ()


def load_holds(root: Path) -> dict[tuple[str, str, str], DecisionHold]:
    """Suspensões de ``<root>/decision_holds.yaml``; arquivo ausente = nenhuma.

    Raises:
        EvidenceError: arquivo ilegível, fora do formato ou com trio repetido.
    """
    import yaml
    from pydantic import ValidationError

    from radiologyai.errors import EvidenceError

    path = root / HOLDS_FILE
    if not path.is_file():
        return {}
    try:
        parsed = _HoldsFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise EvidenceError(f"{path}: suspensões de decisão inválidas: {exc}") from exc
    out: dict[tuple[str, str, str], DecisionHold] = {}
    for hold in parsed.holds:
        key = (hold.condition_id, hold.model_id, hold.label)
        if key in out:
            raise EvidenceError(f"{path}: trio repetido {key}")
        out[key] = hold
    return out


_LEVEL_RANK: dict[EvidenceLevel, int] = {
    EvidenceLevel.A: 3,
    EvidenceLevel.B: 2,
    EvidenceLevel.C: 1,
    EvidenceLevel.D: 0,
}
# ``report.score_scale`` -> escala do ponto de operação que lhe corresponde exatamente.
_SCALE_OF_REPORT: dict[str, str] = {"logit": "logit", "probability": "sigmoid"}
_PROTOCOL_ROLES: tuple[str, ...] = ("rule_out", "rule_in")


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def select_basis(candidates: Sequence[LoadedEvidence], base: Path) -> LoadedEvidence:
    """Registro que determina o nível, pela regra documentada no módulo.

    Ordem: maior contribuição de nível, maior ``n_total``, menor caminho relativo a ``base``.

    Raises:
        ValueError: ``candidates`` vazio.
    """
    if not candidates:
        raise ValueError("select_basis exige ao menos um registro candidato")

    def key(item: LoadedEvidence) -> tuple[int, int, str]:
        level, _, _ = compute_level([item.record])
        return (-_LEVEL_RANK[level], -(item.record.n_total or 0), _relative(item.path, base))

    return min(candidates, key=key)


class EvidencePolicy:
    """Pontos de operação dos registros de evidência que sustentam o nível do catálogo.

    Sem calibração: ``calibrate`` devolve ``None`` (nenhuma probabilidade é
    reportada); a temperatura dos registros não é aplicada ao escore.
    """

    def __init__(
        self,
        catalog: Catalog,
        evidence: Sequence[LoadedEvidence],
        *,
        base: Path,
        holds: Mapping[tuple[str, str, str], DecisionHold] | None = None,
    ) -> None:
        """``base`` é a raiz contra a qual os caminhos dos registros são mostrados."""
        self.catalog = catalog
        self.base = base
        self._by_id: dict[str, LoadedEvidence] = {item.record.id: item for item in evidence}
        self.holds: dict[tuple[str, str, str], DecisionHold] = dict(holds or {})

    @classmethod
    def load(cls, catalog: Catalog, root: Path | None = None) -> EvidencePolicy:
        """Carrega ``evidence/`` (padrão: o diretório do repositório).

        Diretório ausente não é erro: sem registros, todo achado A/B/C fica
        ``score_only`` com aviso (padrão conservador). Lê também as suspensões de
        decisão (``decision_holds.yaml``).

        Raises:
            EvidenceError: arquivo de evidência ou de suspensões inválido, ou id duplicado.
        """
        from radiologyai.evidence.store import EVIDENCE_ROOT, load_evidence

        root = root or EVIDENCE_ROOT
        evidence = load_evidence(root) if root.is_dir() else []
        return cls(catalog, evidence, base=root.parent, holds=load_holds(root))

    # -- SafetyPolicy -------------------------------------------------------- #
    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: ScoreKind
    ) -> tuple[float, str] | None:
        del model_id, label, raw_score, score_kind  # sem calibração aplicada no produto
        return None

    def resolve_points(self, request: PointsRequest) -> PointsResolution:
        hold = self.holds.get((request.condition_id, request.model_id, request.label))
        if hold is not None:
            return PointsResolution(
                warnings=(
                    f"nível {request.level.value}: decisão suspensa em {hold.decided_on} "
                    f"({hold.decided_by}): {hold.reason} [{hold.evidence_ref}]; somente "
                    "escore, sem decisão",
                )
            )
        if request.level is EvidenceLevel.D:
            # Comportamento anterior: exploratório, só no modo pesquisa, fora do laudo.
            return PointsResolution(points=request.manifest_points)
        return self._from_basis(request)

    # -- interno ------------------------------------------------------------- #
    def _abstain(self, request: PointsRequest, why: str) -> PointsResolution:
        return PointsResolution(
            warnings=(
                f"nível {request.level.value}: {why}; somente escore, sem decisão "
                "(pontos do manifesto não são usados)",
            )
        )

    def _from_basis(self, request: PointsRequest) -> PointsResolution:
        entry = self.catalog.entry(request.condition_id, request.model_id, request.label)
        if entry is None or entry.level is not request.level:
            return self._abstain(request, "entrada do catálogo não corresponde ao nível efetivo")
        levels_file = self.catalog.levels_file
        if levels_file is None:
            return self._abstain(request, "catálogo sem levels.json")

        triple = (request.condition_id, request.model_id, request.label)
        candidates: list[LoadedEvidence] = []
        for record_id in entry.basis:
            item = self._by_id.get(record_id)
            if item is None:
                continue
            rec = item.record
            if (rec.condition_id, rec.model_id, rec.label) == triple:
                candidates.append(item)
        if not candidates:
            return self._abstain(
                request, f"nenhum registro da base {list(entry.basis)} encontrado em evidence/"
            )

        chosen = select_basis(candidates, self.base)
        path = _relative(chosen.path, self.base)
        problem = self._unusable(chosen, request, levels_file.protocol_id)
        if problem is not None or chosen.envelope is None:
            return self._abstain(request, f"registro {path} ({chosen.record.id}): {problem}")
        points = tuple(
            point.model_copy(
                update={
                    "evidence_ref": f"{path}#report.{point.role}",
                    "source": f"{point.source} · registro {chosen.record.id}",
                }
            )
            for role in _PROTOCOL_ROLES
            for point in chosen.envelope.operating_points
            if point.role == role
        )
        return PointsResolution(points=points)

    def _unusable(
        self, item: LoadedEvidence, request: PointsRequest, protocol_id: str
    ) -> str | None:
        """Motivo pelo qual o registro escolhido não pode decidir; ``None`` se pode."""
        rec, env = item.record, item.envelope
        if env is None:
            return "registro publicado/diretriz não traz pontos de operação ajustados no protocolo"
        if rec.passes_protocol is not True:
            return "registro não aprovado no protocolo"
        if rec.provenance is None or rec.provenance.protocol_id != protocol_id:
            got = rec.provenance.protocol_id if rec.provenance else None
            return f"protocolo do registro {got!r} difere do de levels.json {protocol_id!r}"
        measured = set(rec.provenance.weights_sha256.values())
        if not measured <= request.weights_sha256:
            return "pesos medidos no registro não são os pesos declarados no manifesto"
        report: Mapping[str, Any] = env.report
        if report.get("protocol_id") != protocol_id:
            return f"relatório sob protocolo {report.get('protocol_id')!r}, não {protocol_id!r}"
        scale = report.get("score_scale")
        expected_on = _SCALE_OF_REPORT.get(scale) if isinstance(scale, str) else None
        if expected_on is None:
            return f"escala do relatório {scale!r} desconhecida"
        points = [p for p in env.operating_points if p.role in _PROTOCOL_ROLES]
        if not points:
            return "registro sem pontos de operação rule_out/rule_in"
        if len({p.role for p in points}) != len(points):
            return "registro com papel de ponto de operação repetido"
        for point in points:
            if point.on != expected_on:
                return (
                    f"ponto {point.role} na escala {point.on!r}, mas o relatório foi ajustado "
                    f"em {scale!r}"
                )
            if point.on != request.score_kind:
                return (
                    f"ponto {point.role} na escala {point.on!r}, mas o adaptador emite "
                    f"{request.score_kind!r}; sem conversão exata, não há decisão"
                )
            fitted = report.get(point.role)
            if not isinstance(fitted, Mapping):
                return f"relatório sem o ajuste de {point.role}"
            threshold = fitted.get("threshold")
            if (
                not isinstance(threshold, int | float)
                or not math.isfinite(threshold)
                or float(threshold) != point.threshold
                or fitted.get("score_scale") != scale
                or fitted.get("role") != point.role
            ):
                return f"limiar de {point.role} não confere com o relatório do próprio registro"
        return None
