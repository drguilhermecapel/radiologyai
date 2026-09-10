"""Trilha de auditoria append-only, encadeada por hash.

Requisito de projeto derivado de REG-01 §4 e do perigo H-07. Todo resultado
precisa ser reconstruível: qual entrada, qual modelo, qual versão dos pesos,
qual saída, e o que o médico decidiu.

O encadeamento por hash (cada registro carrega o hash do anterior) faz com que
alterar ou remover um registro passado invalide todos os seguintes. Não impede
adulteração por quem controla o banco, mas a torna **detectável** — que é o que
uma trilha de auditoria precisa entregar.

Armazenamento: JSON Lines. Simples, append-only por construção, legível sem
ferramenta, e diffável. Postgres entra quando houver concorrência real.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from radiologyai.errors import RadiologyAIError

GENESIS_HASH = "0" * 64


class AuditError(RadiologyAIError):
    """Falha de integridade ou de escrita na trilha."""


class PhysicianDecision(StrEnum):
    """O que o médico fez com a saída da IA.

    Registrar isto é o que torna o "médico no circuito" verificável em vez de
    declarado. Sem esse campo não há como demonstrar que a revisão ocorreu.
    """

    PENDING = "pendente"
    ACCEPTED = "aceito"
    EDITED = "editado"
    REJECTED = "rejeitado"


@dataclass(frozen=True)
class AuditRecord:
    """Um evento na trilha.

    Args:
        prev_hash: hash do registro anterior. Encadeia a trilha.
        input_sha256: hash do arquivo de entrada. Não é o arquivo — a trilha
            não guarda dado de paciente.
        pseudo_patient_id: pseudo-ID determinístico, quando disponível.
    """

    timestamp_utc: str
    event: str
    prev_hash: str
    input_sha256: str
    card_id: str
    weights_sha256: str
    code_version: str
    modality: str
    user_id: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = False
    physician_decision: str = PhysicianDecision.PENDING
    pseudo_patient_id: str | None = None
    note: str | None = None

    def payload(self) -> dict[str, Any]:
        """Conteúdo canônico usado no cálculo do hash."""
        return asdict(self)

    def compute_hash(self) -> str:
        """Hash deste registro, incluindo ``prev_hash``.

        Serialização canônica: chaves ordenadas, sem espaços. Duas máquinas
        chegam ao mesmo hash para o mesmo conteúdo.
        """
        canonical = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditTrail:
    """Trilha em JSON Lines, encadeada por hash.

    Args:
        path: arquivo .jsonl. Criado se não existir.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return sum(1 for _ in self.read())

    def last_hash(self) -> str:
        """Hash do último registro, ou o genesis quando a trilha está vazia."""
        last = GENESIS_HASH
        for entry in self.read():
            last = entry["record_hash"]
        return last

    def append(self, record: AuditRecord) -> str:
        """Acrescenta um registro. Devolve o hash calculado.

        Raises:
            AuditError: ``prev_hash`` não corresponde ao fim da trilha, o que
                indica escrita concorrente ou registro fora de ordem.
        """
        expected = self.last_hash()
        if record.prev_hash != expected:
            raise AuditError(
                f"prev_hash não confere: registro traz {record.prev_hash[:12]}…, "
                f"a trilha termina em {expected[:12]}…"
            )

        record_hash = record.compute_hash()
        line = json.dumps(
            {**record.payload(), "record_hash": record_hash},
            sort_keys=True,
            ensure_ascii=False,
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return record_hash

    def record(self, **kwargs: Any) -> str:
        """Cria e acrescenta um registro, encadeando automaticamente."""
        return self.append(
            AuditRecord(
                timestamp_utc=datetime.now(UTC).isoformat(),
                prev_hash=self.last_hash(),
                **kwargs,
            )
        )

    def read(self) -> list[dict[str, Any]]:
        """Lê todos os registros na ordem de escrita."""
        if not self._path.is_file():
            return []
        entries = []
        for n, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AuditError(f"linha {n} corrompida em {self._path}: {exc}") from exc
        return entries

    def verify(self) -> None:
        """Verifica a integridade de toda a cadeia.

        Raises:
            AuditError: elo quebrado ou registro adulterado, identificando a
                posição exata.
        """
        prev = GENESIS_HASH
        for i, entry in enumerate(self.read()):
            stored = entry.get("record_hash")
            if stored is None:
                raise AuditError(f"registro {i} sem record_hash")

            if entry.get("prev_hash") != prev:
                raise AuditError(
                    f"elo quebrado no registro {i}: prev_hash "
                    f"{str(entry.get('prev_hash'))[:12]}… não corresponde a {prev[:12]}…"
                )

            payload = {k: v for k, v in entry.items() if k != "record_hash"}
            recomputed = hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest()
            if recomputed != stored:
                raise AuditError(
                    f"registro {i} adulterado: hash gravado {stored[:12]}…, "
                    f"recalculado {recomputed[:12]}…"
                )
            prev = stored

    def reconstruct(self, input_sha256: str) -> list[dict[str, Any]]:
        """Todos os eventos de uma entrada, para reconstruir a decisão."""
        return [e for e in self.read() if e.get("input_sha256") == input_sha256]
