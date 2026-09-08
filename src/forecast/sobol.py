"""Sobol variance decomposition, to replace the univariate tornado.

A tornado moves one driver at a time and reports the swing. That answers "how much does
this driver move the answer on its own", which is only the whole story if the model is
additive. This one is not: revenue is a chain of products with clamps in it, so a driver
can matter far more in combination than alone. Sobol splits the output variance into the
part each driver explains by itself and the part it explains only jointly:

    S1  first-order   variance explained by this driver alone
    ST  total-order   variance explained by this driver plus every interaction it is in
    ST - S1           the interaction share, which a tornado cannot see at all

Grouping
--------
One Sobol variable per **register driver**, applied to every indication at once, rather
than one per sampled quantity. Two reasons:

* It matches what the tornado in docs/MODEL_DECISIONS.md section 3 actually did -- it
  swung prevalence for the model, not for one indication -- so the comparison is like
  for like.
* Sampling each indication's copy independently splits a driver's influence in half and
  lets the halves partly cancel, which depresses its index for a reason that has nothing
  to do with how much the driver matters.

The consequence is that indication-level drivers move together here, which is the
opposite of mc.py's default of independence. Neither is obviously right; the truth is a
cross-indication correlation somewhere between them.

Independence
------------
**Sobol indices assume independent inputs, so this analysis deliberately ignores the
copula in params/correlations.yaml.** The Saltelli estimator builds its sample by
swapping columns between two independent matrices; with correlated inputs those hybrid
rows fall outside the joint distribution and the indices stop being a variance
decomposition of the actual model.

That matters here because the register's strongest documented correlation, prevalence
against diagnosis rate at -0.5, is exactly the kind that changes apportionment: the two
partly cancel, so their joint contribution is smaller than the sum of their independent
ones. Read these indices as "how the model apportions variance among drivers treated as
independent", and treat the correlated case as a separate question -- Shapley effects or
Kucherenko indices are the tools for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from SALib.analyze import sobol as _sobol_analyze
from SALib.sample import sobol as _sobol_sample

from .engine import compute
from .mc import build_variables
from .params import Molecule
from .priors import Priors, load_priors
from .mc import _apply as _apply_draw

__all__ = ["analyze", "SobolResult", "TORNADO_RANKING"]

DEFAULT_N = 1024

# The range-based tornado from docs/MODEL_DECISIONS.md section 3, keyed by the register's
# "Model parameter". Rank, swing in cumulative net revenue, and the range swung. Measured
# on the single-indication defaults against a base of $3,285M cumulative.
TORNADO_RANKING = {
    "prevalence": (1, 1_925e6, "110-195 /100K"),
    "eligiblePct": (2, 1_433e6, "42-66%"),
    "classCapture": (3, 1_165e6, "50-72%"),
    "diagnosedPct": (4, 1_111e6, "55-78%"),
    "yearsToPeak": (5, 1_049e6, "3.5-6.5 yr"),
    "segments.discount": (14, 515e6, "x0.90-x1.15"),
}
TORNADO_BASE = 3_285e6


@dataclass(frozen=True)
class SobolResult:
    """First-order, total-order and (optionally) second-order Sobol indices."""

    names: list[str]
    S1: np.ndarray
    S1_conf: np.ndarray
    ST: np.ndarray
    ST_conf: np.ndarray
    S2: np.ndarray | None
    S2_conf: np.ndarray | None
    Y: np.ndarray
    metric: str
    n_samples: int
    n_runs: int

    # A driver below this share of output variance is not worth acting on, whatever its
    # confidence interval says. Bootstrap error bars shrink with the estimate, so a
    # driver explaining 0.002% of variance can be "statistically resolvable" and still
    # be irrelevant.
    MATERIAL = 0.01

    def table(self) -> pd.DataFrame:
        """Indices by driver, most influential first, against the section 3 tornado."""
        rows = []
        for i, name in enumerate(self.names):
            s1, st = float(self.S1[i]), float(self.ST[i])
            tornado = TORNADO_RANKING.get(name)
            rows.append(
                {
                    "driver": name,
                    "S1": s1,
                    "S1_conf": float(self.S1_conf[i]),
                    "ST": st,
                    "ST_conf": float(self.ST_conf[i]),
                    "interaction": st - s1,
                    # Share of this driver's influence that only appears in combination.
                    "interaction_share": (st - s1) / st if st > self.MATERIAL else np.nan,
                    # Worth acting on: explains at least MATERIAL of output variance.
                    "material": st >= self.MATERIAL,
                    # Separately, distinguishable from zero at the estimator's own error
                    # bar. Necessary but not sufficient -- see MATERIAL.
                    "above_noise": abs(st) > float(self.ST_conf[i]),
                    "tornado_rank": tornado[0] if tornado else None,
                    "tornado_swing": tornado[1] if tornado else None,
                }
            )
        df = pd.DataFrame(rows).sort_values("ST", ascending=False).reset_index(drop=True)
        df.insert(0, "sobol_rank", df.index + 1)
        return df

    def interactions(self, top: int = 10) -> pd.DataFrame:
        """Second-order pairs, each with its own error bar.

        ``resolvable`` is the column that matters. Second-order indices need far more
        samples than first-order ones, and on this model the total interaction variance
        is small enough to be spread thinly across hundreds of pairs -- so at ordinary
        sample sizes almost every pair estimate is smaller than its own confidence
        interval and the apparent ranking is noise. Check ``resolvable`` before reading
        anything into the order.
        """
        if self.S2 is None:
            raise ValueError("run analyze(..., second_order=True) to get pair indices")
        rows = []
        for i in range(len(self.names)):
            for j in range(i + 1, len(self.names)):
                v = self.S2[i, j]
                if np.isfinite(v):
                    conf = float(self.S2_conf[i, j])
                    rows.append(
                        {
                            "a": self.names[i],
                            "b": self.names[j],
                            "S2": float(v),
                            "S2_conf": conf,
                            "resolvable": abs(float(v)) > conf,
                        }
                    )
        df = pd.DataFrame(rows)
        return (
            df.reindex(df["S2"].abs().sort_values(ascending=False).index)
            .head(top)
            .reset_index(drop=True)
        )

    def resolvable_pairs(self) -> int:
        """How many second-order estimates exceed their own confidence interval."""
        if self.S2 is None:
            return 0
        finite = np.isfinite(self.S2)
        return int((np.abs(self.S2[finite]) > self.S2_conf[finite]).sum())

    def additivity(self) -> float:
        """Sum of first-order indices. 1.0 means a purely additive model."""
        return float(self.S1.sum())


def _metric(result, metric: str) -> float:
    if metric == "cum_net":
        return result.cum_net
    if metric == "peak_net":
        return float(result.peak["net"])
    if metric == "npv":
        return result.npv
    raise ValueError(
        f"unknown metric {metric!r}; use cum_net, peak_net or npv. "
        "rNPV is excluded because its Bernoulli component is not a continuous "
        "function of the sampled parameters."
    )


def analyze(
    mol: Molecule,
    *,
    n: int = DEFAULT_N,
    metric: str = "cum_net",
    priors: Priors | None = None,
    second_order: bool = True,
    seed: int = 0,
) -> SobolResult:
    """Decompose the variance of ``metric`` across the register's drivers.

    ``n`` is the Saltelli base sample; the model is evaluated ``n * (2D + 2)`` times with
    second-order indices and ``n * (D + 2)`` without, for D drivers.
    """
    if priors is None:
        priors = load_priors(require_coverage=False)

    variables = build_variables(mol, priors)

    # One Sobol column per register driver; every variable sharing that driver is
    # transformed from the same uniform, so the indications move together.
    groups: dict[str, list[int]] = {}
    for i, v in enumerate(variables):
        groups.setdefault(v.param, []).append(i)
    names = sorted(groups)

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [[0.0, 1.0]] * len(names),
    }
    unit = _sobol_sample.sample(
        problem, n, calc_second_order=second_order, seed=seed
    )

    Y = np.empty(len(unit))
    values = np.empty(len(variables))
    for r, row in enumerate(unit):
        for g, name in enumerate(names):
            u = row[g]
            for k in groups[name]:
                var = variables[k]
                # pos_outcome carries the uniform itself; it is compared against the
                # sampled probability inside _apply.
                values[k] = u if var.kind == "pos_outcome" else var.dist.ppf(u)
        perturbed, _, _ = _apply_draw(mol, variables, values)
        Y[r] = _metric(compute(perturbed), metric)

    Si = _sobol_analyze.analyze(
        problem, Y, calc_second_order=second_order, seed=seed, print_to_console=False
    )

    return SobolResult(
        names=names,
        S1=np.asarray(Si["S1"]),
        S1_conf=np.asarray(Si["S1_conf"]),
        ST=np.asarray(Si["ST"]),
        ST_conf=np.asarray(Si["ST_conf"]),
        S2=np.asarray(Si["S2"]) if second_order else None,
        S2_conf=np.asarray(Si["S2_conf"]) if second_order else None,
        Y=Y,
        metric=metric,
        n_samples=n,
        n_runs=len(unit),
    )
