# Pharma forecast workbench

US pharmaceutical revenue forecasting for a multi-indication asset. Demand is built from
epidemiology upward:

```
patient pool -> clinical funnel -> competitive ceiling -> diffusion -> access gate
             -> units -> gross-to-net -> net revenue -> risk-adjusted NPV
```

Everything above units forks per indication. Everything below it is molecule level: one
price, one gross-to-net waterfall, one valuation. Net revenue is attributed back to each
indication on unit share so that risk adjustment applies per indication before summing.

`docs/MODEL_DECISIONS.md` explains why the model is built this way. Several choices that
look arbitrary are not; read it before changing engine logic.

## Install

Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

A clone runs 34 tests and skips 70. That is expected -- see
[The assumptions register](#the-assumptions-register).

## Use

### A deterministic forecast

```python
from forecast import Molecule, compute

mol = Molecule.from_yaml("params/example_ibd.yaml")
f = compute(mol)

f.peak["net"]     # 650_558_...  peak net sales, dollars
f.peak["year"]    # 2037
f.cum_net         # cumulative net revenue over the horizon
f.rnpv            # risk-adjusted NPV, summed over indications
f.blended_gtn     # 48.3, percent of gross
f.summary         # one row per indication
f.molecule        # one row per year
```

`f.summary` carries the per-indication build, including the halo and share-of-voice
adjustments:

```
              name  peak_year         npv  pos        rnpv  t_star  ramp_years
Ulcerative colitis       2037  1.0425e+09   85  8.8615e+08   5.551       2.776
   Crohn's disease       2037  1.0918e+09   65  7.0967e+08   3.221       0.842
```

Ulcerative colitis launches first, so it gets no halo and its time to peak *stretches*
from 5.0 to 5.55 years -- it runs at 0.77x analog promotional support because field
capacity is finite and split across indications. Crohn's launches second into a
prescriber base that already knows the molecule, compressing 4.5 years to 3.22 and its
access ramp from 2.5 years to 0.84.

### Priors from the assumptions register

Each register row's Low/Base/High becomes a modified PERT, `mu = (Low + λ·Base + High) /
(λ + 2)`, with λ = 4, or λ = 2 for rows sourced by expert elicitation.

```python
from forecast import load_priors, MissingRegisterRows

try:
    priors = load_priors()
except MissingRegisterRows as e:
    print(f"{len(e.missing)} parameters have no register row")
    priors = e.priors          # the priors that do exist

p = priors.one("prevalence")   # 110.0 / 145.0 / 195.0, λ=4, mean 147.5
p.dist.rvs(1000)               # a frozen scipy distribution
```

`load_priors()` raises by default on any model parameter with no register row. That is
deliberate: a parameter sitting at its default reports zero uncertainty, which is
indistinguishable in the output from a driver that does not matter. Fifteen parameters
are currently uncovered, including the entire promotional and halo block that
`MODEL_DECISIONS.md` measures at 11.9% of cumulative net revenue.

The register stores percentages as fractions (`0.28`) while the model stores them as
`28`. `priors.py` reconciles this and records the factor on every row as
`Prior.scaled_by`.

### Monte Carlo

```python
from forecast import Molecule, run

mol = Molecule.from_yaml("params/example_ibd.yaml")
mc = run(mol, n_draws=10_000, seed=0)

mc.summary()                  # P10/P50/P90 per metric, success rates
mc.quantiles("cum_net")
mc.p_below(1_000e6)           # probability rNPV falls below a threshold
mc.diagnostics                # what was sampled, and what could not be
```

```
peak_net  P10 $473M    P50 $602M    P90 $751M
cum_net   P10 $3,103M  P50 $3,913M  P90 $4,889M
rnpv      P10 $706M    P50 $1,504M  P90 $2,302M
```

Drivers are drawn through a Gaussian copula, not independently -- independent sampling
misstates variance in both directions. Correlations live in `params/correlations.yaml`
and are meant to be edited. An inconsistent matrix is repaired to the nearest positive
semi-definite one, and the repair is reported rather than silently applied.

Register rows supply *relative* uncertainty: a draw becomes `shock = draw / register_base`
applied to whatever the parameter file holds. The register describes a related but
different asset, so sampling its absolute values would forecast that asset instead of
this one. The register supplies spread; `params/example_ibd.yaml` supplies level.

Probability of success is a Bernoulli per indication rather than a haircut, which gives
rNPV a real left tail: on the example parameters both indications fail together in about
6% of draws.

Two things to know when reading the output:

* **rNPV cannot go below zero in this model.** Net sales are floored at zero and there
  are no costs -- no COGS, no R&D, no SG&A, no milestones -- so `p_rnpv_below_zero` is
  structurally 0.0, not an estimate. Use `p_below(threshold)` against a real hurdle.
* **The base case is not a P50.** It sits around P66 on revenue, because most of the
  register's ranges are skewed against it.

`mc.diagnostics` names what the run could not do: parameters with no register row,
variables that are inert as configured, and whether competitor entry years were sampled
at all.

## The assumptions register

`reference/assumptions_register.xlsx` is **not committed**. It cites Komodo,
Merative/MarketScan, IQVIA, Symphony, HealthVerity, Truveta and Datavant as the sources
behind its values. Those licences prohibit redistribution and git history is permanent,
so the file stays local (see the data handling section of `CLAUDE.md`).

`priors.py` and `mc.py` read it at runtime, so without it:

| | tests |
|---|---|
| clone, no register | 34 passed, 70 skipped |
| with `reference/` present | 104 passed |

The engine and its golden master are fully covered either way. To restore the rest, put
the register at `reference/assumptions_register.xlsx`, or point `priors.load_priors()` at
your own copy.

Confidential parameters belong in `params/real/`, which is gitignored. Only synthetic or
public examples go in `params/`.

## Layout

| Path | |
|---|---|
| `src/forecast/params.py` | pydantic models: `Segment`, `Competitor`, `Indication`, `Molecule` |
| `src/forecast/engine.py` | the forecast. Pure: parameters in, dataframes out, no I/O |
| `src/forecast/priors.py` | register rows to PERT distributions; the I/O boundary |
| `src/forecast/mc.py` | Gaussian copula, discrete drivers, output distributions |
| `params/example_ibd.yaml` | example parameters; the golden master is defined against these |
| `params/correlations.yaml` | correlation structure, meant to be edited |
| `docs/MODEL_DECISIONS.md` | why the model is built this way |

`engine.py` does no file reads, no printing, and holds no module-level state, so the
Monte Carlo can call it in a tight loop. The indication/molecule boundary is enforced by
function signature: `_indication_demand` is never passed price, gross-to-net, channel or
discount rate, so computing a rebate per indication is a `NameError` rather than a
plausible-looking wrong number.

## Verification

The Python port reproduces the original JSX engine bit-for-bit. `tests/test_golden.py`
checks the nine headline figures from `CLAUDE.md`; `tests/test_jsx_trace.py` checks the
whole build year by year and column by column against a fixture generated from the JSX
itself, at 1e-9 relative tolerance.

That second test is the one that matters. The golden master is nine rounded aggregates,
and compensating errors can satisfy all of them: injecting a deliberate 0.1% error into
the units calculation passes all 12 golden tests and fails 7 trace tests.

Beyond the defaults, the port was diffed against the JSX across twelve parameter variants
covering in-line mode, incidence-flow epidemiology, loss of exclusivity, competition off,
launch-year ties and three indications. Worst relative deviation across every one:
2.9e-16, a single float ULP.
