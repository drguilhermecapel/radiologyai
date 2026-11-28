"""Métricas de desempenho — medidas, nunca geradas.

O legado tinha 164 chamadas a ``np.random`` em ``src/``, incluindo
``cv_scores = np.random.normal(base_accuracy, 0.02, n_folds)`` em
``medai_advanced_clinical_validation.py:357``: cross-validation **sintetizada**.

Aqui não há nenhuma fonte de aleatoriedade exceto os reamostradores de
bootstrap, que recebem seed explícita e são usados para estimar incerteza sobre
dados reais. Um verificador de CI (``scripts/check_honesty.py``) impede a
reintrodução de ``np.random`` fora deste arquivo — por isso **todo** gerador
aleatório do pacote de avaliação vive aqui, e os demais módulos estatísticos
recebem pesos de reamostragem prontos de :class:`ClusterBootstrap`.

Dois bootstraps convivem:

* :func:`auroc_with_ci` — por imagem, estratificado por classe. Mantido por
  compatibilidade com os artefatos já commitados (2026-09). **Subestima** a
  incerteza quando um paciente contribui várias imagens (o teste oficial do NIH
  tem até 184 imagens por paciente).
* :class:`ClusterBootstrap` — por paciente (``bootstrap_cluster: patient`` no
  protocolo ``2026-11-v1``), estratificado pela presença de algum positivo no
  paciente. É o exigido pelo protocolo.

As estatísticas ponderadas (``weighted_*``) recebem uma matriz de pesos
``B x n`` (multiplicidade de cada imagem em cada réplica) e calculam todas as
réplicas de uma vez. Com pesos unitários reproduzem exatamente as definições
usuais (Mann-Whitney com empates = ``sklearn.metrics.roc_auc_score``; precisão
média = ``sklearn.metrics.average_precision_score``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from radiologyai.errors import EvaluationError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.floating[Any]]
    IntArray = npt.NDArray[np.integer[Any]]

BOOTSTRAP_CHUNK = 64
"""Réplicas processadas por bloco. Não afeta o resultado (os sorteios são feitos
réplica a réplica, na mesma ordem), apenas a memória: ``64 x n`` pesos."""


class AUROCResult(NamedTuple):
    """AUROC com intervalo de confiança e contagens de suporte."""

    auroc: float
    ci95_low: float
    ci95_high: float
    n_pos: int
    n_neg: int
    ci_method: str


class OperatingPoint(NamedTuple):
    """Ponto de operação num alvo de sensibilidade."""

    target_sensitivity: float
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float


def _validate_binary(y_true: IntArray, y_score: FloatArray) -> tuple[int, int]:
    import numpy as np

    if y_true.shape != y_score.shape:
        raise EvaluationError(
            f"shapes incompatíveis: y_true {y_true.shape} vs y_score {y_score.shape}"
        )
    if y_true.size == 0:
        raise EvaluationError("conjunto vazio")
    if not np.all(np.isfinite(np.asarray(y_score, dtype=np.float64))):
        raise EvaluationError("y_score contém valores não finitos (NaN/inf)")

    unique = set(np.unique(y_true).tolist())
    if not unique <= {0, 1}:
        raise EvaluationError(f"y_true deve ser binário; encontrado {sorted(unique)}")

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise EvaluationError(
            f"AUROC indefinido: {n_pos} positivos, {n_neg} negativos. "
            "Reportar AUROC aqui produziria NaN — foi o que o legado fez "
            "(mean_auc: NaN sobre test_samples: 1)."
        )
    return n_pos, n_neg


def auroc(y_true: IntArray, y_score: FloatArray) -> float:
    """AUROC pela estatística de Mann-Whitney U, com tratamento de empates."""
    import numpy as np
    from scipy.stats import rankdata

    _validate_binary(y_true, y_score)
    ranks = rankdata(np.asarray(y_score, dtype=np.float64), method="average")
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_with_ci(
    y_true: IntArray,
    y_score: FloatArray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260101,
) -> AUROCResult:
    """AUROC com IC95% por bootstrap estratificado **por imagem**.

    Estratificado por classe, para que cada reamostra preserve a prevalência e
    nunca degenere numa amostra de classe única. Mantido para os artefatos de
    2026-09; para o protocolo pré-registrado use :class:`ClusterBootstrap`
    (reamostragem por paciente), porque imagens do mesmo paciente são
    correlacionadas e o IC por imagem sai estreito demais.

    Args:
        seed: obrigatória e explícita. Duas execuções com a mesma seed produzem
            resultado idêntico — requisito de reprodutibilidade para V&V.
    """
    import numpy as np

    n_pos, n_neg = _validate_binary(y_true, y_score)
    point = auroc(y_true, y_score)

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=n_pos, replace=True),
                rng.choice(neg_idx, size=n_neg, replace=True),
            ]
        )
        samples[b] = auroc(y_true[idx], y_score[idx])

    low, high = np.percentile(samples, [2.5, 97.5])
    return AUROCResult(
        auroc=point,
        ci95_low=float(low),
        ci95_high=float(high),
        n_pos=n_pos,
        n_neg=n_neg,
        ci_method=f"bootstrap estratificado (n={n_bootstrap}, seed={seed})",
    )


def _confusion_at(
    y_true: IntArray, y_score: FloatArray, threshold: float
) -> tuple[int, int, int, int]:
    pred = y_score >= threshold
    pos = y_true == 1
    tp = int((pred & pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & ~pos).sum())
    fp = int((pred & ~pos).sum())
    return tp, fn, tn, fp


def threshold_for_sensitivity(y_true: IntArray, y_score: FloatArray, target: float) -> float:
    """**Maior** limiar observado cuja sensibilidade (``escore >= limiar``) atinge ``target``.

    Entre os limiares que satisfazem a restrição de sensibilidade, o maior é o
    que maximiza a especificidade. Os candidatos são os valores de escore
    observados; nunca um limiar fora dos dados.

    Raises:
        EvaluationError: alvo fora de (0, 1] ou dados degenerados.
    """
    import numpy as np

    _validate_binary(y_true, y_score)
    if not 0.0 < target <= 1.0:
        raise EvaluationError(f"alvo de sensibilidade deve estar em (0, 1]; recebido {target}")
    scores = np.asarray(y_score, dtype=np.float64)
    uniq, inverse = np.unique(scores, return_inverse=True)
    pos_per = np.bincount(inverse, weights=(y_true == 1).astype(np.float64), minlength=uniq.size)
    # sensibilidade ao limiar uniq[k] = positivos com escore >= uniq[k]
    sens = np.cumsum(pos_per[::-1])[::-1] / pos_per.sum()
    ok = np.flatnonzero(sens >= target - 1e-12)
    # sens é não crescente em k; ok nunca é vazio porque sens[0] == 1.
    return float(uniq[ok.max()])


def threshold_for_specificity(y_true: IntArray, y_score: FloatArray, target: float) -> float:
    """**Menor** limiar observado cuja especificidade atinge ``target``.

    Negativo é ``escore < limiar``.

    Entre os limiares que satisfazem a restrição de especificidade, o menor é o
    que maximiza a sensibilidade.

    Raises:
        EvaluationError: alvo fora de (0, 1], ou nenhum valor de escore observado
            atinge a especificidade pedida (ex.: empate no topo da distribuição).
    """
    import numpy as np

    _validate_binary(y_true, y_score)
    if not 0.0 < target <= 1.0:
        raise EvaluationError(f"alvo de especificidade deve estar em (0, 1]; recebido {target}")
    scores = np.asarray(y_score, dtype=np.float64)
    uniq, inverse = np.unique(scores, return_inverse=True)
    neg_per = np.bincount(inverse, weights=(y_true == 0).astype(np.float64), minlength=uniq.size)
    # especificidade ao limiar uniq[k] = negativos com escore < uniq[k]
    spec = (np.cumsum(neg_per) - neg_per) / neg_per.sum()
    ok = np.flatnonzero(spec >= target - 1e-12)
    if ok.size == 0:
        raise EvaluationError(
            f"nenhum valor de escore observado atinge especificidade {target}; "
            "o ponto de inclusão não é definido (nenhum limiar inventado fora dos dados)"
        )
    return float(uniq[ok.min()])


def operating_point_at_sensitivity(
    y_true: IntArray, y_score: FloatArray, target_sensitivity: float = 0.90
) -> OperatingPoint:
    """Ponto de operação no **maior** limiar que ainda atinge a sensibilidade-alvo.

    Entre os limiares que garantem a sensibilidade pedida, o maior maximiza a
    especificidade. (A docstring anterior dizia "menor threshold", mas o código
    sempre devolveu o maior — o comportamento correto; a documentação foi
    corrigida.) Sensibilidade é priorizada porque, em triagem radiológica, o
    custo de um falso negativo em achado crítico domina (perigo H-01).

    Atenção: medido no mesmo conjunto em que é ajustado, o ponto é otimista por
    construção. O protocolo pré-registrado ajusta na metade de calibração e
    reporta na metade de teste (:mod:`radiologyai.evaluation.protocol_eval`).
    """
    _validate_binary(y_true, y_score)
    if not 0.0 < target_sensitivity <= 1.0:
        raise EvaluationError(
            f"target_sensitivity deve estar em (0, 1]; recebido {target_sensitivity}"
        )
    t = threshold_for_sensitivity(y_true, y_score, target_sensitivity)
    tp, fn, tn, fp = _confusion_at(y_true, y_score, t)
    return OperatingPoint(
        target_sensitivity=target_sensitivity,
        threshold=t,
        sensitivity=tp / (tp + fn) if (tp + fn) else 0.0,
        specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        ppv=tp / (tp + fp) if (tp + fp) else 0.0,
        npv=tn / (tn + fn) if (tn + fn) else 0.0,
    )


def _ece_bins(y_score: FloatArray, n_bins: int) -> IntArray:
    """Índice do bin de cada escore (largura fixa; o primeiro bin inclui 0)."""
    import numpy as np

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.searchsorted(edges, np.asarray(y_score, dtype=np.float64), side="left") - 1
    return np.clip(idx, 0, n_bins - 1)


def expected_calibration_error(y_true: IntArray, y_score: FloatArray, *, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) com binning de largura fixa.

    Um ECE alto significa que o escore **não** é probabilidade de doença — o que
    deve ser declarado na IFU (perigo H-10).
    """
    import numpy as np

    _validate_binary(y_true, y_score)
    scores = np.asarray(y_score, dtype=np.float64)
    if scores.min() < 0.0 or scores.max() > 1.0:
        raise EvaluationError("ECE exige escore em [0, 1]")
    return float(weighted_ece(y_true, scores, np.ones((1, scores.size)), n_bins=n_bins)[0])


# --------------------------------------------------------------------------- #
# Estatísticas ponderadas (todas as réplicas de bootstrap de uma vez)
# --------------------------------------------------------------------------- #
class _SortedScores(NamedTuple):
    order: IntArray  # permutação que ordena os escores (crescente)
    starts: IntArray  # início de cada grupo de empate na ordem
    pos_sorted: FloatArray  # 1.0 onde positivo, na ordem


def _sorted_groups(y_true: IntArray, y_score: FloatArray) -> _SortedScores:
    import numpy as np

    scores = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    starts = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1]]))
    pos_sorted = (np.asarray(y_true)[order] == 1).astype(np.float64)
    return _SortedScores(order, starts, pos_sorted)


def weighted_auroc(y_true: IntArray, y_score: FloatArray, weights: FloatArray) -> FloatArray:
    """AUROC (Mann-Whitney, empates valem 1/2) para cada linha de ``weights``.

    Args:
        weights: ``B x n`` multiplicidades (ou ``n`` para uma única réplica).

    Returns:
        vetor ``B``; ``NaN`` numa réplica sem positivos ou sem negativos.
    """
    import numpy as np

    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    g = _sorted_groups(y_true, y_score)
    ws = w[:, g.order]
    p_g = np.add.reduceat(ws * g.pos_sorted, g.starts, axis=1)
    n_g = np.add.reduceat(ws * (1.0 - g.pos_sorted), g.starts, axis=1)
    below = np.cumsum(n_g, axis=1) - n_g
    num = (p_g * (below + 0.5 * n_g)).sum(axis=1)
    den = p_g.sum(axis=1) * n_g.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def weighted_average_precision(
    y_true: IntArray, y_score: FloatArray, weights: FloatArray
) -> FloatArray:
    """Precisão média (AUPRC, definição do scikit-learn) para cada linha de ``weights``.

    ``AP = Σ_k (R_k − R_{k−1}) · P_k`` sobre os limiares distintos em ordem
    decrescente, sem interpolação.
    """
    import numpy as np

    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    g = _sorted_groups(y_true, y_score)
    ws = w[:, g.order]
    p_g = np.add.reduceat(ws * g.pos_sorted, g.starts, axis=1)[:, ::-1]
    n_g = np.add.reduceat(ws * (1.0 - g.pos_sorted), g.starts, axis=1)[:, ::-1]
    tp = np.cumsum(p_g, axis=1)
    fp = np.cumsum(n_g, axis=1)
    total_p = tp[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1.0), 0.0)
        ap = (p_g * precision).sum(axis=1) / total_p[:, 0]
    out: FloatArray = np.where(total_p[:, 0] > 0, ap, np.nan)
    return out


def weighted_rates_at(
    y_true: IntArray,
    y_score: FloatArray,
    threshold: float,
    weights: FloatArray,
    *,
    direction: str = "ge",
) -> dict[str, FloatArray]:
    """Sensibilidade, especificidade, VPP e VPN num limiar fixo, por réplica."""
    import numpy as np

    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    scores = np.asarray(y_score, dtype=np.float64)
    pred = (scores >= threshold) if direction == "ge" else (scores <= threshold)
    pos = np.asarray(y_true) == 1
    tp = w @ (pred & pos).astype(np.float64)
    fn = w @ (~pred & pos).astype(np.float64)
    tn = w @ (~pred & ~pos).astype(np.float64)
    fp = w @ (pred & ~pos).astype(np.float64)

    def ratio(a: FloatArray, b: FloatArray) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            r: FloatArray = np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)
        return r

    return {
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
    }


def weighted_ece(
    y_true: IntArray, y_score: FloatArray, weights: FloatArray, *, n_bins: int = 10
) -> FloatArray:
    """ECE de largura fixa para cada linha de ``weights``."""
    import numpy as np

    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    scores = np.asarray(y_score, dtype=np.float64)
    bins = _ece_bins(scores, n_bins)
    onehot = np.zeros((scores.size, n_bins), dtype=np.float64)
    onehot[np.arange(scores.size), bins] = 1.0
    count = w @ onehot
    conf = w @ (onehot * scores[:, None])
    acc = w @ (onehot * (np.asarray(y_true) == 1).astype(np.float64)[:, None])
    total = count.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ece: FloatArray = np.abs(acc - conf).sum(axis=1) / total
    return ece


# --------------------------------------------------------------------------- #
# Bootstrap por paciente (cluster), estratificado
# --------------------------------------------------------------------------- #
class ClusterBootstrap:
    """Reamostrador bootstrap por cluster (paciente), estratificado.

    Cada réplica sorteia, com reposição, tantos pacientes quantos existem em cada
    estrato, e cada imagem herda a multiplicidade do seu paciente. O estrato de
    um paciente é o máximo de ``strata`` entre suas imagens — com ``strata =
    y_true``, "tem algum positivo" vs. "só negativos" — o que garante positivos
    e negativos em toda réplica.

    Os sorteios são feitos réplica a réplica, estrato a estrato, sempre na mesma
    ordem: o resultado depende apenas de ``seed``, nunca do tamanho do bloco.
    """

    def __init__(
        self,
        clusters: Sequence[str] | npt.NDArray[Any],
        *,
        seed: int,
        strata: IntArray | None = None,
    ) -> None:
        import numpy as np

        cl = np.asarray(clusters)
        if cl.ndim != 1 or cl.size == 0:
            raise EvaluationError("bootstrap por cluster exige um vetor não vazio de ids")
        _, inverse = np.unique(cl.astype(str), return_inverse=True)
        self._inverse = inverse.astype(np.int64)
        self.n_rows = int(cl.size)
        self.n_clusters = int(inverse.max()) + 1
        self.seed = int(seed)
        cluster_stratum = np.zeros(self.n_clusters, dtype=np.int64)
        if strata is not None:
            st = np.asarray(strata, dtype=np.int64)
            if st.shape != cl.shape:
                raise EvaluationError("strata deve ter o mesmo tamanho dos ids de cluster")
            np.maximum.at(cluster_stratum, self._inverse, st)
        self._strata_members = [
            np.flatnonzero(cluster_stratum == s) for s in np.unique(cluster_stratum)
        ]

    def weights(self, n_bootstrap: int) -> Iterator[FloatArray]:
        """Blocos ``b x n_rows`` de multiplicidades, somando ``n_bootstrap`` réplicas."""
        import numpy as np

        if n_bootstrap < 1:
            raise EvaluationError("n_bootstrap deve ser >= 1")
        rng = np.random.default_rng(self.seed)
        done = 0
        while done < n_bootstrap:
            b = min(BOOTSTRAP_CHUNK, n_bootstrap - done)
            counts = np.zeros((b, self.n_clusters), dtype=np.float64)
            for r in range(b):
                for members in self._strata_members:
                    draw = members[rng.integers(0, members.size, size=members.size)]
                    counts[r] += np.bincount(draw, minlength=self.n_clusters)
            yield counts[:, self._inverse]
            done += b

    def indices(self, n_bootstrap: int) -> Iterator[IntArray]:
        """Uma réplica por vez como vetor de índices de linha (com repetição)."""
        import numpy as np

        base = np.arange(self.n_rows)
        for block in self.weights(n_bootstrap):
            for row in block:
                yield np.repeat(base, row.astype(np.int64))


MIN_DEFINED_REPLICATE_FRACTION = 0.95
"""Fração mínima de réplicas de bootstrap com estatística definida para reportar o IC.

Réplica em que a estatística é indefinida (NaN: ex. ICC com variância zero) não
entra no percentil; descartá-las em silêncio enviesa o IC para as réplicas
"fáceis" enquanto o método continua declarando B réplicas. Por isso elas são
**contadas** e reportadas, e abaixo desta fração o IC fica indefinido (``None``):
um IC indefinido nunca aprova nada (o julgamento trata limite ausente como
reprovação). É uma salvaguarda que só pode tornar o resultado mais conservador.
"""


class BootstrapInterval(NamedTuple):
    """IC percentílico com a contagem de réplicas definidas."""

    low: float | None  # None: réplicas definidas abaixo do mínimo
    high: float | None
    n_defined: int
    n_replicates: int

    @property
    def n_undefined(self) -> int:
        return self.n_replicates - self.n_defined


def bootstrap_interval(
    samples: FloatArray,
    ci_level: float = 0.95,
    *,
    min_defined_fraction: float = MIN_DEFINED_REPLICATE_FRACTION,
) -> BootstrapInterval:
    """IC percentílico que conta as réplicas indefinidas (NaN/inf) em vez de escondê-las.

    Raises:
        EvaluationError: nenhuma réplica, ou nenhuma réplica definida.
    """
    import numpy as np

    arr = np.asarray(samples, dtype=np.float64).ravel()
    total = int(arr.size)
    if total == 0:
        raise EvaluationError("bootstrap sem réplicas")
    defined = arr[np.isfinite(arr)]
    if defined.size == 0:
        raise EvaluationError("nenhuma réplica de bootstrap produziu estatística definida")
    if defined.size < min_defined_fraction * total:
        return BootstrapInterval(None, None, int(defined.size), total)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(defined, [100 * alpha, 100 * (1 - alpha)])
    return BootstrapInterval(float(lo), float(hi), int(defined.size), total)


def percentile_ci(samples: FloatArray, ci_level: float = 0.95) -> tuple[float, float]:
    """IC percentílico para quem não reporta contagem de réplicas.

    Réplicas indefinidas são toleradas só até :data:`MIN_DEFINED_REPLICATE_FRACTION`;
    abaixo disso é erro explícito, nunca um IC calculado sobre o subconjunto
    "fácil". Quem precisa reportar a contagem usa :func:`bootstrap_interval`.

    Raises:
        EvaluationError: nenhuma réplica definida, ou réplicas definidas abaixo do
            mínimo.
    """
    ci = bootstrap_interval(samples, ci_level)
    if ci.low is None or ci.high is None:
        raise EvaluationError(
            f"só {ci.n_defined} de {ci.n_replicates} réplicas de bootstrap com estatística "
            f"definida (< {MIN_DEFINED_REPLICATE_FRACTION:.0%}); IC não reportável"
        )
    return ci.low, ci.high


def cluster_bootstrap_samples(
    resampler: ClusterBootstrap,
    statistic: Any,
    n_bootstrap: int,
) -> FloatArray:
    """Aplica ``statistic(pesos_bloco) -> vetor`` a todas as réplicas e concatena."""
    import numpy as np

    parts = [
        np.asarray(statistic(block), dtype=np.float64) for block in resampler.weights(n_bootstrap)
    ]
    out: FloatArray = np.concatenate(parts)
    return out


def bootstrap_method(n_bootstrap: int, seed: int, ci_level: float = 0.95) -> str:
    """Descrição textual do método de IC, gravada em todo resultado."""
    return (
        f"IC{round(ci_level * 100)}% bootstrap percentílico por paciente (cluster), "
        f"estratificado por presença de positivo; B={n_bootstrap}, seed={seed}"
    )
