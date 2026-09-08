"""Monte Carlo over the register's three-point estimates, coupled by a Gaussian copula.

This replaces the deterministic scenario set, whose multiplier bundles are perfectly
correlated and carry no attached probability (docs/MODEL_DECISIONS.md, known gaps).

How a draw is built
-------------------
1. One correlated standard normal vector per draw, from the correlation matrix in
   params/correlations.yaml. Independent sampling misstates variance in both directions,
   which is the whole reason for the copula.
2. Normals to uniforms through the normal CDF.
3. Uniforms to parameter values through each variable's own inverse CDF -- the modified
   PERT from priors.py for continuous drivers, a rounded triangular for competitor entry
   years, and a Bernoulli for probability of success.

Sampling every driver from its own register-derived distribution is the point. A uniform
+-20% swing makes every linear driver tie for first, which measures the analyst's choice
of swing rather than the forecast (MODEL_DECISIONS section 3), and independent sampling
reproduces that same error in a different form.

Relative shocks, not absolute values
------------------------------------
The register describes a related but different asset: wac base 9800 against 6800 in
params/example_ibd.yaml, prevalence 145 against 290 and 215. Sampling its absolute
numbers would forecast the register's asset rather than this one, and the median run
would not resemble the deterministic base case.

So each register row supplies *relative* uncertainty. A draw becomes a shock,
``shock = draw / register_base``, applied to whatever that field holds in the parameter
file. The register supplies the spread; the parameter file supplies the level. At the
median draw the shock is about 1 and the run reproduces the base case, which is what
makes the golden master a usable reference for the Monte Carlo.

Scope
-----
The register is single-indication and predates the multi-indication model, so scope has
to be bridged:

* Molecule-level rows apply once.
* Indication-level rows become one variable per indication, so UC and Crohn's get
  separate draws. Correlate them through ``cross_indication`` in the YAML; they are
  independent unless named there.
* Payer-segment rows are pooled to one shock per parameter applied to every segment,
  because the register's segmentation (5 mix rows, 4 access, 6 discount) is a different
  cut from the model's six segments and mapping row i to segment i would misalign them.
  A pooled shock on ``segments.mix`` is a no-op by construction -- the engine normalises
  mix by its own sum -- so payer mix carries no uncertainty here. ``Diagnostics`` says so
  rather than leaving it implicit.
* Competitor entry years are sampled only for competitors entering at or after the base
  year. In the example parameters every competitor entered between 2012 and 2024, so
  nothing is sampled and ``Diagnostics.entry_year_variables`` is zero.

Probability of success
----------------------
The deterministic engine applies PoS as a haircut, ``rnpv = npv * pos/100``. Here it is
hierarchical: the probability itself is drawn from its register row, then success is a
Bernoulli draw at that probability, per indication. The expectation is unchanged, but
rNPV becomes a distribution with a real left tail including the case where an indication
contributes nothing at all -- which is the point of sampling PoS rather than haircutting
(MODEL_DECISIONS, next steps item 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import annotated_types
import numpy as np
import yaml
from scipy import stats

from .engine import compute
from .params import Competitor, Indication, Molecule, Segment
from .priors import Prior, Priors, load_priors
from .priors import pert as _pert

__all__ = [
    "run",
    "McResult",
    "Diagnostics",
    "Variable",
    "DEFAULT_CORRELATIONS",
]

DEFAULT_CORRELATIONS = (
    Path(__file__).resolve().parents[2] / "params" / "correlations.yaml"
)

DEFAULT_DRAWS = 10_000

# Register parameters that are pooled across payer segments rather than sampled per
# segment. See the module docstring.
SEGMENT_PARAMS = {
    "segments.mix": "mix",
    "segments.access": "access",
    "segments.discount": "discount",
}

_MODELS = {
    "Molecule": Molecule,
    "Indication": Indication,
    "Segment": Segment,
    "Competitor": Competitor,
}


# ---------------------------------------------------------------------------------
# variables
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Variable:
    """One sampled quantity: a marginal, and where its draw is applied."""

    name: str  # e.g. "wac", "prevalence@Crohn's disease"
    param: str  # the register's "Model parameter"
    kind: str  # relative | pos_probability | pos_outcome | entry_year
    dist: Any  # frozen scipy distribution, or None for a plain uniform
    base: float  # register base, the denominator of a relative shock
    target: tuple  # ("molecule", field) | ("indication", idx, field)
    #                 | ("segments", field) | ("competitor", idx, cidx)


def _field_bounds(model_name: str, fname: str) -> tuple[float | None, float | None]:
    """Validation bounds declared on a params.py field, so a shock cannot escape them."""
    lo = hi = None
    for meta in _MODELS[model_name].model_fields[fname].metadata:
        if isinstance(meta, annotated_types.Ge):
            lo = meta.ge
        elif isinstance(meta, annotated_types.Gt):
            lo = meta.gt
        elif isinstance(meta, annotated_types.Le):
            hi = meta.le
        elif isinstance(meta, annotated_types.Lt):
            hi = meta.lt
    return lo, hi


def build_variables(mol: Molecule, priors: Priors) -> list[Variable]:
    """Expand the register's rows into one variable per sampled quantity."""
    variables: list[Variable] = []

    for param, group in sorted(priors.by_param.items()):
        first = group[0]
        model_name, _, fname = first.field.partition(".")

        if param in SEGMENT_PARAMS:
            # Pooled across segments. Average the rows' *levels* and take the ratio,
            # rather than averaging each row's ratio: a segment with a small base has a
            # very wide relative range -- Medicare Part B discounts run 5/8/12, a ratio
            # span of 0.63-1.50 -- and averaging ratios lets it dominate a shock that is
            # meant to describe the blend. Pooling on levels reproduces the ranges the
            # model is documented against: gross-to-net depth x0.89-x1.16 against the
            # x0.90-x1.15 in docs/MODEL_DECISIONS.md section 3, and formulary coverage
            # x0.86-x1.10 against the x0.85-x1.10 in the JSX's own sensitivity drivers.
            lo = sum(p.low for p in group) / len(group)
            mid = sum(p.base for p in group) / len(group)
            hi = sum(p.high for p in group) / len(group)
            lam = min(p.lam for p in group)  # widest, if the rows disagree

            variables.append(
                Variable(
                    name=param,
                    param=param,
                    kind="relative",
                    dist=_pert(lo, mid, hi, lam),
                    base=mid,
                    target=("segments", SEGMENT_PARAMS[param]),
                )
            )
            continue

        if param == "competitors.entry":
            continue  # handled separately, against the model's own competitors

        if model_name == "Molecule":
            variables.append(
                Variable(param, param, "relative", first.dist, first.base,
                         ("molecule", fname))
            )
            continue

        if model_name == "Indication":
            for idx, ind in enumerate(mol.indications):
                if fname == "pos":
                    variables.append(
                        Variable(f"{param}@{ind.name}", param, "pos_probability",
                                 first.dist, first.base, ("indication", idx, "pos"))
                    )
                else:
                    variables.append(
                        Variable(f"{param}@{ind.name}", param, "relative",
                                 first.dist, first.base, ("indication", idx, fname))
                    )
            continue

    # Competitor entry years: triangular on the register's range, rounded to a year.
    # Only competitors that have not already entered by the base year are in play.
    entry_rows = priors.by_param.get("competitors.entry", [])
    if entry_rows:
        row = entry_rows[0]
        span = row.high - row.low
        for idx, ind in enumerate(mol.indications):
            for cidx, comp in enumerate(ind.competitors):
                if comp.entry < mol.base_year:
                    continue
                c = (row.base - row.low) / span if span else 0.5
                variables.append(
                    Variable(
                        name=f"competitors.entry@{ind.name}#{cidx}",
                        param="competitors.entry",
                        kind="entry_year",
                        dist=stats.triang(c, loc=row.low, scale=span),
                        base=row.base,
                        target=("competitor", idx, cidx),
                    )
                )

    # Bernoulli outcome per indication, drawn at the sampled probability.
    for idx, ind in enumerate(mol.indications):
        variables.append(
            Variable(f"pos_outcome@{ind.name}", "pos", "pos_outcome", None, 0.0,
                     ("indication", idx, "pos"))
        )

    return variables


# ---------------------------------------------------------------------------------
# correlation matrix
# ---------------------------------------------------------------------------------


def _nearest_psd(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Clip negative eigenvalues and restore a unit diagonal.

    A hand-edited correlation file can easily specify something no joint distribution
    can satisfy. Returns the repaired matrix and how far it moved.
    """
    sym = (matrix + matrix.T) / 2
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= 0:
        return sym, 0.0
    repaired = vecs @ np.diag(np.clip(vals, 1e-10, None)) @ vecs.T
    d = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(d, d)
    np.fill_diagonal(repaired, 1.0)
    return repaired, float(np.abs(repaired - sym).max())


def build_correlation(
    variables: Sequence[Variable],
    mol: Molecule,
    spec: dict,
) -> tuple[np.ndarray, list[str], float]:
    """Assemble the correlation matrix from the YAML specification."""
    index = {v.name: i for i, v in enumerate(variables)}
    n = len(variables)
    R = np.eye(n)
    applied: list[str] = []

    def put(a: str, b: str, rho: float, label: str) -> None:
        if a not in index or b not in index:
            return
        i, j = index[a], index[b]
        if i == j:
            return
        R[i, j] = R[j, i] = rho
        applied.append(f"{label}: {a} <-> {b} = {rho:+.2f}")

    for pair in spec.get("pairs") or []:
        a, b, rho = pair["a"], pair["b"], float(pair["rho"])
        scope = pair.get("scope", "explicit")
        if scope == "within_indication":
            for ind in mol.indications:
                put(f"{a}@{ind.name}", f"{b}@{ind.name}", rho, "within-indication")
        else:
            put(a, b, rho, scope)

    for param, rho in (spec.get("cross_indication") or {}).items():
        rho = float(rho)
        names = [f"{param}@{ind.name}" for ind in mol.indications]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                put(names[i], names[j], rho, "cross-indication")

    outcomes = spec.get("pos_outcomes") or {}
    if "rho" in outcomes:
        rho = float(outcomes["rho"])
        names = [f"pos_outcome@{ind.name}" for ind in mol.indications]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                put(names[i], names[j], rho, "pos-outcome")

    R, moved = _nearest_psd(R)
    return R, applied, moved


# ---------------------------------------------------------------------------------
# applying a draw
# ---------------------------------------------------------------------------------


def _apply(mol: Molecule, variables: Sequence[Variable], values: np.ndarray) -> tuple[Molecule, list[bool], int]:
    """Build the perturbed Molecule for one draw.

    Returns the molecule, the per-indication success flags, and how many shocked values
    had to be clamped back inside their declared bounds.
    """
    data = mol.model_dump()
    clamped = 0
    successes = [True] * len(mol.indications)

    for var, value in zip(variables, values):
        kind = var.kind

        if kind == "pos_outcome":
            continue  # resolved below, once the probability is known

        if kind == "entry_year":
            _, idx, cidx = var.target
            data["indications"][idx]["competitors"][cidx]["entry"] = int(round(value))
            continue

        shock = value / var.base if var.base else 1.0

        if var.target[0] == "molecule":
            fname = var.target[1]
            lo, hi = _field_bounds("Molecule", fname)
            new, hit = _clamp(data[fname] * shock, lo, hi)
            data[fname] = new
            clamped += hit
        elif var.target[0] == "segments":
            fname = var.target[1]
            lo, hi = _field_bounds("Segment", fname)
            for seg in data["segments"]:
                new, hit = _clamp(seg[fname] * shock, lo, hi)
                seg[fname] = new
                clamped += hit
        else:
            _, idx, fname = var.target
            lo, hi = _field_bounds("Indication", fname)
            new, hit = _clamp(data["indications"][idx][fname] * shock, lo, hi)
            data["indications"][idx][fname] = new
            clamped += hit

    # Bernoulli success, at each indication's freshly sampled probability.
    for var, value in zip(variables, values):
        if var.kind == "pos_outcome":
            _, idx, _ = var.target
            p = data["indications"][idx]["pos"] / 100
            successes[idx] = bool(value < p)

    return Molecule.model_validate(data), successes, clamped


def _clamp(value: float, lo: float | None, hi: float | None) -> tuple[float, int]:
    out = value
    if lo is not None:
        out = max(out, lo)
    if hi is not None:
        out = min(out, hi)
    return out, int(out != value)


# ---------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostics:
    """What the run actually sampled, and what it could not."""

    n_variables: int
    correlations_applied: list[str]
    psd_repair: float
    entry_year_variables: int
    pooled_segment_params: list[str]
    inert_variables: list[str]
    clamped_values: int
    unsampled_parameters: dict[str, str]
    failures: int


@dataclass(frozen=True)
class McResult:
    """Distributions from a Monte Carlo run. All money figures are dollars."""

    peak_net: np.ndarray  # unconditional, comparable to the deterministic peak
    cum_net: np.ndarray  # unconditional
    npv: np.ndarray  # unconditional
    rnpv: np.ndarray  # Bernoulli-adjusted: failed indications contribute nothing
    cum_net_realised: np.ndarray  # revenue from indications that succeeded
    successes: np.ndarray  # bool, one column per indication
    indication_names: list[str]
    n_draws: int
    seed: int
    base: Any  # the deterministic ForecastResult, for reference
    diagnostics: Diagnostics

    def quantiles(self, metric: str, qs: Iterable[float] = (10, 50, 90)) -> dict[float, float]:
        values = getattr(self, metric)
        return {q: float(np.percentile(values, q)) for q in qs}

    def p_below(self, threshold: float, metric: str = "rnpv") -> float:
        """Probability the metric falls below a threshold."""
        return float((getattr(self, metric) < threshold).mean())

    def summary(self) -> dict:
        """P10/P50/P90 on the headline metrics, plus the downside probability."""
        out = {
            m: self.quantiles(m) for m in ("peak_net", "cum_net", "npv", "rnpv")
        }
        out["mean_rnpv"] = float(self.rnpv.mean())
        out["p_rnpv_below_zero"] = self.p_below(0.0)
        out["p_all_indications_fail"] = float((~self.successes.any(axis=1)).mean())
        out["p_success"] = {
            name: float(self.successes[:, i].mean())
            for i, name in enumerate(self.indication_names)
        }
        return out


# ---------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------


def run(
    mol: Molecule,
    *,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    priors: Priors | None = None,
    correlations: str | Path | dict | None = None,
) -> McResult:
    """Run the Monte Carlo.

    ``priors`` defaults to the register with coverage enforcement relaxed -- the register
    does not yet cover every parameter, and ``load_priors()`` reports which. Parameters
    with no register row stay at their value in ``mol`` and contribute no uncertainty,
    which ``Diagnostics.unsampled_parameters`` records.
    """
    if priors is None:
        priors = load_priors(require_coverage=False)

    if isinstance(correlations, dict):
        spec = correlations
    else:
        path = Path(correlations) if correlations is not None else DEFAULT_CORRELATIONS
        spec = yaml.safe_load(path.read_text()) or {} if path.exists() else {}

    variables = build_variables(mol, priors)
    R, applied, moved = build_correlation(variables, mol, spec)

    # Correlated normals -> uniforms -> marginals.
    rng = np.random.default_rng(seed)
    normals = rng.multivariate_normal(np.zeros(len(variables)), R, size=n_draws,
                                      method="cholesky")
    uniforms = stats.norm.cdf(normals)

    values = np.empty_like(uniforms)
    for k, var in enumerate(variables):
        if var.kind == "pos_outcome":
            values[:, k] = uniforms[:, k]  # compared against the sampled probability
        else:
            values[:, k] = var.dist.ppf(uniforms[:, k])

    base = compute(mol)
    n_ind = len(mol.indications)

    peak = np.empty(n_draws)
    cum = np.empty(n_draws)
    npv = np.empty(n_draws)
    rnpv = np.empty(n_draws)
    cum_realised = np.empty(n_draws)
    successes = np.empty((n_draws, n_ind), dtype=bool)
    clamped = 0
    failures = 0

    for d in range(n_draws):
        try:
            perturbed, success, hits = _apply(mol, variables, values[d])
            result = compute(perturbed)
        except Exception:
            # A draw that cannot be built or evaluated is recorded, not silently
            # replaced with a plausible number.
            failures += 1
            peak[d] = cum[d] = npv[d] = rnpv[d] = cum_realised[d] = np.nan
            successes[d] = False
            continue

        clamped += hits
        successes[d] = success
        peak[d] = float(result.peak["net"])
        cum[d] = result.cum_net
        npv[d] = result.npv

        per_npv = result.summary["npv"].to_numpy()
        per_cum = result.summary["cum"].to_numpy()
        mask = np.array(success, dtype=float)
        rnpv[d] = float((per_npv * mask).sum())
        cum_realised[d] = float((per_cum * mask).sum())

    inert = _inert_variables(mol, variables)
    diagnostics = Diagnostics(
        n_variables=len(variables),
        correlations_applied=applied,
        psd_repair=moved,
        entry_year_variables=sum(1 for v in variables if v.kind == "entry_year"),
        pooled_segment_params=sorted(SEGMENT_PARAMS),
        inert_variables=inert,
        clamped_values=clamped,
        unsampled_parameters=dict(priors.missing),
        failures=failures,
    )

    return McResult(
        peak_net=peak,
        cum_net=cum,
        npv=npv,
        rnpv=rnpv,
        cum_net_realised=cum_realised,
        successes=successes,
        indication_names=[i.name for i in mol.indications],
        n_draws=n_draws,
        seed=seed,
        base=base,
        diagnostics=diagnostics,
    )


def _inert_variables(mol: Molecule, variables: Sequence[Variable]) -> list[str]:
    """Variables that are sampled but cannot move the answer as configured.

    Worth naming: an inert driver looks identical to an unimportant one in the output,
    and the difference matters when reading a variance decomposition.
    """
    inert = []
    for v in variables:
        if v.name == "segments.mix":
            inert.append("segments.mix (engine normalises mix, so a pooled shock cancels)")
        elif v.target[0] == "indication":
            idx, fname = v.target[1], v.target[2]
            ind = mol.indications[idx]
            if fname in {"incidence", "start_prevalent", "mortality", "remission"} \
                    and ind.epi_mode != "incidence":
                inert.append(f"{v.name} (indication is in prevalence mode)")
            elif fname == "prevalence" and ind.epi_mode != "prevalence":
                inert.append(f"{v.name} (indication is in incidence mode)")
    return inert
