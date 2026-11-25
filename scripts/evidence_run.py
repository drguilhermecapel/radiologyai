#!/usr/bin/env python3
"""Mede um (modelo, dataset, rótulo) e grava o registro de evidência.

Entrada: parquet de predições já inferidas (ver ``radiologyai.evidence.predictions``):

* por rótulo — ``<preds-root>/<modelo>/<dataset>/<rótulo>.parquet`` com colunas
  ``patient_id, image_id, y_true, y_score[, sex, age, view]``;
* multirrótulo — um arquivo com ``patient_id, image_id[, sex, age, view]`` e, por
  rótulo ``L``, ``y_true__L`` e ``y_score__L``.

Saída: ``evidence/measured/<dataset>/<modelo>/<rótulo>.json`` (envelope com o
``EvidenceRecord``, o relatório completo e os pontos de operação).

Desfecho: discriminação (desenho pré-registrado: metade de calibração / metade de
teste por paciente). Concordância, segmentação e detecção em regra fixa usam a
API Python (``radiologyai.evidence.measure``).

Exemplos::

    python scripts/evidence_run.py --model-id xrv-densenet121-pc --dataset-id nih-cxr14 \\
        --label Effusion --condition Effusion=cxr.pleural_effusion \\
        --training-data padchest --weights /caminho/pc-densenet121.pt \\
        --reference-standard report_nlp

    python scripts/evidence_run.py --predictions preds.parquet --all-labels ...

Códigos de saída: 0 sucesso; 2 erro de entrada/proveniência; 3 algum rótulo não
pôde ser avaliado (os demais foram gravados).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_PREDS_ROOT = Path("/root/medai-cache/eval-preds")


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"ERRO: {what} deve ser NOME=VALOR; recebido {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--dataset-id", required=True, help="chave em evidence.leakage.DATASETS")
    p.add_argument("--predictions", type=Path, help="parquet de predições (senão, --preds-root)")
    p.add_argument("--preds-root", type=Path, default=DEFAULT_PREDS_ROOT)
    p.add_argument("--label", action="append", default=[], help="rótulo (repetível)")
    p.add_argument("--all-labels", action="store_true", help="todos os rótulos do parquet")
    p.add_argument(
        "--condition",
        action="append",
        default=[],
        help="ROTULO=condition_id (repetível); obrigatório para cada rótulo medido",
    )
    p.add_argument(
        "--training-data",
        action="append",
        default=[],
        help="chave de dataset de treino do modelo (repetível); controla o vazamento",
    )
    p.add_argument("--no-training-data-declared", action="store_true")
    p.add_argument("--weights", action="append", type=Path, default=[], help="arquivo de pesos")
    p.add_argument("--weights-sha256", action="append", default=[], help="NOME=sha256")
    p.add_argument(
        "--reference-standard",
        choices=[
            "radiologist_adjudicated",
            "radiologist_single_read",
            "report_nlp",
            "histopathology",
            "expert_segmentation",
            "clinical_reference_measurement",
            "other",
        ],
    )
    p.add_argument("--population")
    p.add_argument("--score-scale", choices=["probability", "logit"], default="probability")
    p.add_argument("--internal-holdout", action="store_true")
    p.add_argument("--caveat", action="append", default=[])
    p.add_argument("--out", type=Path, help="raiz de evidence/ (padrão: a do repositório)")
    p.add_argument(
        "--n-bootstrap",
        type=int,
        help="sobrescreve o protocolo; só permitido com --out (nunca na evidência do repo)",
    )
    p.add_argument("--allow-dirty", action="store_true", help="aceita árvore git modificada")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from radiologyai.catalog import load_protocol
    from radiologyai.errors import RadiologyAIError
    from radiologyai.evidence.leakage import lineage
    from radiologyai.evidence.measure import MeasurementContext, measure_binary
    from radiologyai.evidence.predictions import labels_in, read_label
    from radiologyai.evidence.provenance import git_state, sha256_file
    from radiologyai.evidence.store import EVIDENCE_ROOT, measured_path, write_measured

    command = shlex.join([Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])])
    try:
        protocol = load_protocol()
        lineage(args.dataset_id)
        for key in args.training_data:
            lineage(key)
        if not args.training_data and not args.no_training_data_declared:
            raise SystemExit(
                "ERRO: declare --training-data (chaves) ou --no-training-data-declared "
                "(o registro sairá com leakage 'possible')"
            )
        if args.n_bootstrap is not None and args.out is None:
            raise SystemExit(
                "ERRO: --n-bootstrap altera o procedimento do protocolo; só com --out "
                "fora de evidence/ do repositório"
            )
        root = args.out or EVIDENCE_ROOT

        sha, dirty = git_state()
        if dirty and not args.allow_dirty:
            raise SystemExit(
                "ERRO: árvore git com modificações rastreadas não commitadas; a medição não "
                "seria reproduzível a partir do sha. Faça commit ou use --allow-dirty "
                "(o registro sai com aviso e não deve ser commitado)."
            )
        weights = _pairs(args.weights_sha256, "--weights-sha256")
        for w in args.weights:
            weights[w.name] = sha256_file(w)
        if not weights:
            raise SystemExit("ERRO: informe --weights (arquivo) ou --weights-sha256")
        conditions = _pairs(args.condition, "--condition")

        jobs: list[tuple[Path, str, str]] = []  # (arquivo, coluna_rótulo, rótulo)
        if args.predictions:
            available = labels_in(args.predictions)
            wanted = available if args.all_labels else args.label
            if available == [""]:
                if len(wanted) != 1:
                    raise SystemExit("ERRO: parquet por rótulo exige exatamente um --label")
                jobs.append((args.predictions, "", wanted[0]))
            else:
                unknown = sorted(set(wanted) - set(available))
                if unknown or not wanted:
                    raise SystemExit(f"ERRO: rótulos ausentes do parquet: {unknown or wanted}")
                jobs += [(args.predictions, lab, lab) for lab in wanted]
        else:
            if not args.label:
                raise SystemExit("ERRO: informe --label (ou --predictions com --all-labels)")
            for lab in args.label:
                path = args.preds_root / args.model_id / args.dataset_id / f"{lab}.parquet"
                jobs.append((path, "", lab))

        failures = 0
        for path, column, label in jobs:
            if label not in conditions:
                raise SystemExit(f"ERRO: rótulo {label!r} sem --condition {label}=<condition_id>")
            preds = read_label(path, column)
            ctx = MeasurementContext(
                model_id=args.model_id,
                label=label,
                condition_id=conditions[label],
                dataset_id=args.dataset_id,
                training_data=tuple(args.training_data),
                internal_holdout=args.internal_holdout,
                reference_standard=args.reference_standard,
                population=args.population,
                weights_sha256=weights,
                git_sha=sha,
                git_dirty=dirty,
                command=command,
                dataset_sha256=preds.dataset_sha256,
                created_at=datetime.now(UTC),
                caveats=tuple(args.caveat),
            )
            out_path = measured_path(root, args.dataset_id, args.model_id, label)
            try:
                rel = out_path.relative_to(root.parent).as_posix()
            except ValueError:
                rel = out_path.as_posix()
            try:
                evidence = measure_binary(
                    ctx,
                    protocol,
                    patient_ids=preds.patient_ids,
                    y_true=preds.y_true,
                    y_score=preds.y_score,
                    score_scale=args.score_scale,
                    subgroup_columns=preds.subgroups,
                    n_bootstrap=args.n_bootstrap,
                    inputs={
                        "predictions_file": path.name,
                        "predictions_sha256": preds.predictions_sha256,
                        "n_missing_label_excluded": preds.n_missing_label,
                        "score_column": f"y_score__{column}" if column else "y_score",
                    },
                    evidence_ref=f"{rel}#report",
                )
            except RadiologyAIError as exc:
                failures += 1
                print(f"[{label}] NÃO AVALIADO: {exc}", file=sys.stderr)
                continue
            written = write_measured(evidence, root)
            rec = evidence.record
            verdict = {True: "APROVADO", False: "REPROVADO", None: "NÃO ELEGÍVEL"}[
                rec.passes_protocol
            ]
            print(f"[{label}] {verdict} ({rec.kind}, leakage={rec.leakage}) -> {written}")
            for reason in rec.failure_reasons:
                print(f"    - {reason}")
        return 3 if failures else 0
    except RadiologyAIError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
