"""Verificação de integridade de splits.

Vazamento por paciente é a forma mais comum de um número publicado de IA
radiológica estar errado: a mesma pessoa aparece em treino e teste, e o modelo
memoriza anatomia em vez de patologia. É erro fatal, nunca aviso.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from radiologyai.errors import PatientLeakageError


def assert_patient_disjoint(splits: Mapping[str, Iterable[str]]) -> None:
    """Verifica que nenhum ``patient_id`` aparece em mais de um split.

    Args:
        splits: mapa ``nome_do_split -> ids de paciente``.

    Raises:
        PatientLeakageError: qualquer interseção não vazia.
    """
    as_sets = {name: set(ids) for name, ids in splits.items()}
    names = sorted(as_sets)

    violations: list[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = as_sets[a] & as_sets[b]
            if shared:
                sample = sorted(shared)[:5]
                violations.append(f"{a} ∩ {b}: {len(shared)} pacientes em comum (ex.: {sample})")

    if violations:
        raise PatientLeakageError("vazamento por paciente entre splits: " + "; ".join(violations))
