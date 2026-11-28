"""Documentos de estado acompanham ``levels.json`` e o ledger de medições (REQ-190, REQ-191).

O revisor encontrou HONEST_STATUS, README e ROADMAP dizendo que o catálogo estava vazio e
que nenhum registro medido existia, enquanto ``levels.json`` e ``evidence/measured/`` diziam
o contrário. Estes testes tornam a divergência uma falha de CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
LEVELS = REPO / "src" / "radiologyai" / "catalog" / "levels.json"
LEDGER = REPO / "evidence" / "measured" / "ledger.yaml"
KPI_DOCS = ("ROADMAP.md", "HONEST_STATUS.md", "README.md")
BEGIN = "<!-- kpi-levels:inicio (gerado de src/radiologyai/catalog/levels.json) -->"
END = "<!-- kpi-levels:fim -->"
ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}

# Frases que o revisor encontrou e que contradizem levels.json / evidence/measured.
STALE_PHRASES = (
    "catálogo de condições está vazio",
    "nenhum registro medido",
    "duas medições reais",
    "existem hoje **duas**",
    "nenhuma condição catalogada ainda",
    "nenhum dado de paciente brasileiro foi analisado",
    "ainda **vazio**",
    "nenhuma dessas sondas foi avaliada sob o protocolo",
    "sem medição sob o protocolo",
    "avaliação sob o protocolo pendente",
)


def render_kpi(levels: dict) -> str:
    """Tabela do KPI a partir de levels.json, contando só entradas ``distributed``.

    Nível de uma condição = melhor nível entre suas entradas distribuídas; refutada =
    alguma entrada distribuída refutada por medição própria. Modalidade = prefixo do id.
    """
    best: dict[str, str] = {}
    refuted: set[str] = set()
    for e in levels["entries"]:
        if not e.get("distributed"):
            continue
        cid = e["condition_id"]
        if cid not in best or ORDER[e["level"]] < ORDER[best[cid]]:
            best[cid] = e["level"]
        if e.get("refuted_by_measurement"):
            refuted.add(cid)

    def row(name: str, ids: list[str]) -> str:
        counts = [sum(1 for c in ids if best[c] == lv) for lv in "ABCD"]
        n_ref = sum(1 for c in ids if c in refuted)
        return "| " + " | ".join([name, *map(str, counts), str(n_ref)]) + " |"

    modalities = sorted({c.split(".")[0] for c in best})
    lines = [
        "| Modalidade (prefixo) | A | B | C | D | Refutadas |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [row(f"`{m}`", [c for c in best if c.split(".")[0] == m]) for m in modalities]
    lines.append(row("**Todas**", list(best)))
    return "\n".join(lines)


def _kpi_block(text: str) -> str:
    assert BEGIN in text, "marcador de início do KPI ausente"
    assert END in text, "marcador de fim do KPI ausente"
    inner = text.split(BEGIN, 1)[1].split(END, 1)[0]
    return "\n".join(ln for ln in inner.strip().splitlines() if ln.startswith("|"))


@pytest.fixture(scope="module")
def levels() -> dict:
    return json.loads(LEVELS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ledger() -> dict:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


@pytest.mark.requirement("REQ-190")
@pytest.mark.parametrize("doc", KPI_DOCS)
def test_kpi_table_matches_levels_json(doc, levels):
    expected = render_kpi(levels)
    got = _kpi_block((REPO / doc).read_text(encoding="utf-8"))
    assert got == expected, (
        f"{doc}: KPI diverge de levels.json. Substitua o bloco entre os marcadores por:\n{expected}"
    )


@pytest.mark.requirement("REQ-190")
@pytest.mark.parametrize("doc", [*KPI_DOCS, "docs/protocolo-evidencia.md"])
def test_no_stale_status_phrases(doc):
    text = (REPO / doc).read_text(encoding="utf-8").lower()
    found = [p for p in STALE_PHRASES if p.lower() in text]
    assert not found, f"{doc} ainda contém frases desmentidas por levels.json: {found}"


@pytest.mark.requirement("REQ-190")
def test_honest_status_lists_every_ledger_outcome(ledger):
    """Toda medição com registro, e todo item pendente, aparece no HONEST_STATUS."""
    text = (REPO / "HONEST_STATUS.md").read_text(encoding="utf-8")
    missing = []
    for e in ledger["entries"]:
        if e.get("record"):
            if f"evidence/{e['record']}" not in text:
                missing.append(e["record"])
        elif e["outcome"] == "pendente" and e["job"] not in text:
            missing.append(e["job"])
    assert not missing, f"HONEST_STATUS.md omite: {missing}"


@pytest.mark.requirement("REQ-190")
@pytest.mark.parametrize("doc", [*KPI_DOCS, "MODEL_LICENSES.md", "docs/protocolo-evidencia.md"])
def test_cited_evidence_paths_exist(doc):
    text = (REPO / doc).read_text(encoding="utf-8")
    paths = set(re.findall(r"evidence/(?:measured|published|selection)/[\w./@:-]+", text))
    missing = sorted(p.rstrip(".:") for p in paths if not (REPO / p.rstrip(".:")).exists())
    assert not missing, f"{doc} cita caminhos de evidência inexistentes: {missing}"


@pytest.mark.requirement("REQ-190")
def test_risk_file_requirement_column_matches_frontmatter():
    by_risk: dict[str, set[str]] = {}
    for path in (REPO / "docs" / "regulatory" / "06-requirements").glob("REQ-*.md"):
        front = yaml.safe_load(path.read_text(encoding="utf-8").split("---")[1])
        for risk in front.get("risk_controls") or []:
            by_risk.setdefault(risk, set()).add(front["id"])
    table = (REPO / "docs" / "regulatory" / "04-risk-file.md").read_text(encoding="utf-8")
    problems = []
    for line in table.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8 or not re.fullmatch(r"RISK-\d{3}", cells[1]) or "REQ-" not in cells[6]:
            continue
        listed = set(re.findall(r"REQ-\d{3}", cells[6]))
        want = by_risk.get(cells[1], set())
        if listed != want:
            problems.append(
                f"{cells[1]}: faltam {sorted(want - listed)}, sobram {sorted(listed - want)}"
            )
    assert not problems, "\n".join(problems)


# --- REQ-191: redação honesta dos níveis -------------------------------------------------


def _ts_messages(lang: str) -> dict[str, str]:
    """Extrai as mensagens ``level.*.meaning`` de messages.ts (objeto ``pt`` ou ``en``)."""
    src = (REPO / "frontend" / "src" / "i18n" / "messages.ts").read_text(encoding="utf-8")
    start = src.index(f"export const {lang}")
    end = src.find("\nexport ", start + 1)
    end = len(src) if end < 0 else end
    body = src[start:end]
    out = {}
    for m in re.finditer(r'"level\.([abcd])\.meaning":\s*"((?:[^"\\]|\\.)*)"', body):
        out[m.group(1).upper()] = m.group(2)
    return out


@pytest.mark.requirement("REQ-191")
def test_frontend_pt_level_meaning_mirrors_catalog():
    from radiologyai.catalog import LEVEL_MEANING_PT

    pt = _ts_messages("pt")
    assert set(pt) == {"A", "B", "C", "D"}
    for level, meaning in LEVEL_MEANING_PT.items():
        rest = meaning.split(" — ", 1)[1]
        canonical = rest[0].upper() + rest[1:]
        assert pt[level].startswith(canonical), (
            f"nível {level}: o frontend deve começar com o texto do catálogo:\n{canonical}"
        )


@pytest.mark.requirement("REQ-191")
@pytest.mark.parametrize(("lang", "banned"), [("pt", r"validad|validaç"), ("en", r"validat")])
def test_level_wording_never_says_validated(lang, banned):
    for level, text in _ts_messages(lang).items():
        assert not re.search(banned, text, re.I), f"{lang} nível {level}: {text}"


@pytest.mark.requirement("REQ-191")
@pytest.mark.parametrize(
    "doc",
    [
        "README.md",
        "ROADMAP.md",
        "docs/regulatory/01-intended-use.md",
        "docs/regulatory/12-labeling-ifu.md",
    ],
)
def test_docs_level_tables_never_say_validated(doc):
    """As tabelas de níveis dos documentos não usam 'validado' (linguagem vedada, §3.1)."""
    for line in (REPO / doc).read_text(encoding="utf-8").splitlines():
        if re.match(r"\|\s*\*\*[ABCD]\*\*\s*\|", line):
            assert not re.search(r"validad", line, re.I), f"{doc}: {line}"


# --- REQ-190: inventário de licenças acompanha os manifestos ------------------------------

_CU_WORDS = (("não", "no"), ("nao", "no"), ("incerto", "unclear"), ("sim", "yes"))


def _expand(token: str) -> list[str]:
    m = re.fullmatch(r"(.*)\{([^}]*)\}(.*)", token)
    if not m:
        return [token]
    return [f"{m.group(1)}{part}{m.group(3)}" for part in m.group(2).split(",")]


@pytest.mark.requirement("REQ-190")
def test_model_licenses_commercial_use_matches_manifests():
    manifests = {}
    for path in (REPO / "src" / "radiologyai" / "analysis" / "manifests").glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifests[path.stem] = str(data["license"]["commercial_use"])
    lines = (REPO / "MODEL_LICENSES.md").read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    problems = []
    col_m = col_cu = None
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.lstrip().startswith("|"):
            col_m = col_cu = None
            continue
        if "Uso comercial" in cells and any(c.startswith("Manifesto") for c in cells):
            col_m = next(i for i, c in enumerate(cells) if c.startswith("Manifesto"))
            col_cu = cells.index("Uso comercial")
            continue
        if col_m is None or col_cu is None or set(line.strip()) <= set("|-: "):
            continue
        ids = [x for tok in re.findall(r"`([^`]+)`", cells[col_m]) for x in _expand(tok)]
        word = re.sub(r"[*_]", "", cells[col_cu]).strip().lower()
        declared = next((v for k, v in _CU_WORDS if word.startswith(k)), None)
        for mid in ids:
            matched = [
                k for k in manifests if k == mid or (mid.endswith("*") and k.startswith(mid[:-1]))
            ]
            if not matched:
                problems.append(f"{mid}: manifesto inexistente")
            for k in matched:
                seen.add(k)
                if declared != manifests[k]:
                    problems.append(
                        f"{k}: tabela diz {cells[col_cu]!r}, manifesto {manifests[k]!r}"
                    )
    problems += [f"{k}: manifesto ausente da tabela" for k in sorted(set(manifests) - seen)]
    assert not problems, "\n".join(problems)
