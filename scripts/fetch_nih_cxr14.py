#!/usr/bin/env python3
"""Baixa o NIH ChestX-ray14 dos links oficiais do NIH (Box).

Alternativa à rota do Kaggle, para quem não quer criar conta. Mais lenta.

Os 12 arquivos somam ~42 GB comprimidos. O download é resumível: arquivos já
presentes e completos são pulados.

AVISO SOBRE OS LINKS: as URLs abaixo vêm do `batch_download_zips.py` publicado
pelo NIH junto ao dataset. São links do Box e podem mudar. Se algum falhar,
confira a lista atual na página do dataset:
https://nihcc.app.box.com/v/ChestXray-NIHCC

Uso:
    python scripts/fetch_nih_cxr14.py --out /content/nih
    python scripts/fetch_nih_cxr14.py --out ./data/nih --metadata-only
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

# Arquivos de imagem (12 tarballs, ~42 GB no total).
IMAGE_ARCHIVES: list[tuple[str, str]] = [
    (
        "images_001.tar.gz",
        "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    ),
    (
        "images_002.tar.gz",
        "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    ),
    (
        "images_003.tar.gz",
        "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    ),
    (
        "images_004.tar.gz",
        "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    ),
    (
        "images_005.tar.gz",
        "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    ),
    (
        "images_006.tar.gz",
        "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    ),
    (
        "images_007.tar.gz",
        "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    ),
    (
        "images_008.tar.gz",
        "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    ),
    (
        "images_009.tar.gz",
        "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    ),
    (
        "images_010.tar.gz",
        "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    ),
    (
        "images_011.tar.gz",
        "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    ),
    (
        "images_012.tar.gz",
        "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
    ),
]

# Metadados e splits oficiais. O split de teste do NIH já é disjunto por
# paciente — é por isso que usamos o oficial em vez de construir o nosso.
#
# Servidos pelo espelho do Hugging Face: os links diretos do Box para estes três
# arquivos mudaram e retornam 404. Os arquivos são idênticos aos publicados pelo
# NIH; o CSV é verificado abaixo pelo cabeçalho esperado. Os tarballs de imagem
# continuam vindo do Box, cujos links foram verificados e funcionam.
HF_MIRROR = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/"

METADATA: list[tuple[str, str]] = [
    ("Data_Entry_2017_v2020.csv", HF_MIRROR + "Data_Entry_2017_v2020.csv"),
    ("test_list.txt", HF_MIRROR + "test_list.txt"),
    ("train_val_list.txt", HF_MIRROR + "train_val_list.txt"),
]

EXPECTED_CSV_HEADER = "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age"

# Contagens oficiais publicadas pelo NIH. Servem de verificação de sanidade:
# um download truncado é detectado aqui, não depois de horas de inferência.
EXPECTED_COUNTS = {
    "Data_Entry_2017_v2020.csv": 112_120,  # linhas de dados
    "test_list.txt": 25_596,
    "train_val_list.txt": 86_524,
}


def verify_metadata(out: Path) -> bool:
    """Confere cabeçalho e contagem de linhas dos arquivos oficiais."""
    ok = True
    for name, expected in EXPECTED_COUNTS.items():
        path = out / name
        if not path.is_file():
            print(f"  FALTA {name}", file=sys.stderr)
            ok = False
            continue

        lines = path.read_text(encoding="utf-8").splitlines()
        if name.endswith(".csv"):
            if not lines[0].startswith(EXPECTED_CSV_HEADER):
                print(f"  {name}: cabeçalho inesperado -> {lines[0][:70]}", file=sys.stderr)
                ok = False
            count = len(lines) - 1
        else:
            count = len([ln for ln in lines if ln.strip()])

        status = "OK " if count == expected else "!! "
        print(f"  {status}{name}: {count} (esperado {expected})")
        if count != expected:
            ok = False
    return ok


def download(url: str, dest: Path) -> bool:
    """Baixa um arquivo, pulando se já existir com tamanho não-nulo."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  já presente: {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        return True

    print(f"  baixando {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            while chunk := response.read(1 << 20):
                fh.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100 * read / total
                    print(f"\r    {pct:5.1f}%  {read / 1e9:.2f} GB", end="", flush=True)
        print()
        tmp.rename(dest)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  FALHOU {dest.name}: {exc}", file=sys.stderr)
        print(
            "  Se o link estiver morto, confira a lista atual em "
            "https://nihcc.app.box.com/v/ChestXray-NIHCC",
            file=sys.stderr,
        )
        tmp.unlink(missing_ok=True)
        return False
    return True


def extract(archive: Path, out: Path) -> None:
    """Extrai um tarball para ``out/<nome>/``."""
    target = out / archive.name.replace(".tar.gz", "")
    if target.is_dir() and any(target.rglob("*.png")):
        print(f"  já extraído: {target.name}")
        return
    print(f"  extraindo {archive.name} ...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(target, filter="data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="diretório de destino")
    parser.add_argument(
        "--metadata-only", action="store_true", help="baixa só CSV e splits (~10 MB)"
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="não apaga os .tar.gz após extrair"
    )
    parser.add_argument(
        "--max-archives",
        type=int,
        default=len(IMAGE_ARCHIVES),
        help="limita o número de tarballs (para teste)",
    )
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Destino: {out}\n\nMetadados e splits oficiais:")
    if not all(download(url, out / name) for name, url in METADATA):
        return 1

    print("\nVerificação de integridade:")
    if not verify_metadata(out):
        print(
            "\nERRO: os arquivos oficiais não conferem. Download truncado ou fonte "
            "alterada. Não prossiga: uma avaliação sobre metadados incompletos "
            "produz um número sem sentido.",
            file=sys.stderr,
        )
        return 1

    if args.metadata_only:
        print("\nSomente metadados. Use sem --metadata-only para as imagens.")
        return 0

    print(f"\nImagens ({args.max_archives} arquivos, ~42 GB no total):")
    for name, url in IMAGE_ARCHIVES[: args.max_archives]:
        archive = out / name
        if not download(url, archive):
            return 1
        extract(archive, out)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
            print(f"  removido {name} (libera espaço; extraído já está em disco)")

    n = sum(1 for _ in out.rglob("*.png"))
    print(f"\nPronto. {n} imagens em {out}")
    print("Próximo passo: radiologyai manifest", out, "--split test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
