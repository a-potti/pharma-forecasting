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

A clone runs the full suite: 131 passed, 1 skipped. The single skip re-verifies the
Python port against the original JSX engine, which is not distributed -- the committed
fixture still checks the port itself.

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
peak_net  P10 $471M    P50 $618M    P90 $807M
cum_net   P10 $3,088M  P50 $4,001M  P90 $5,192M
rnpv      P10 $725M    P50 $1,534M  P90 $2,408M
```

Drivers are drawn through a Gaussian copula, not independently -- independent sampling
misstates variance in both directions. Correlations live in `params/correlations.yaml`
and are meant to be edited. An inconsistent matrix is repaired to the nearest positive
semi-definite one, and the repair is reported rather than silently applied.

Indication-level parameters are drawn once per indication, so `cross_indication` matters
more than it looks. Left independent, the two draws partly cancel and every epidemiology
and funnel driver is understated -- prevalence ranks 8th by influence on cumulative net
revenue. With the shared measurement error seeded (the two indications are one disease
area counted from one claims database under one case definition), it ranks 2nd.

Specifying that correctly needs one more step. If prevalence and diagnosis rate correlate
-0.5 within an indication, and each correlates 0.8 across indications, then prevalence in
one against diagnosis rate in the other cannot be zero -- setting it so describes no joint
distribution at all, and puts the smallest eigenvalue at -0.30. `mc.py` derives the
implied term as the product, restoring the smallest eigenvalue to +0.10, and lists it as
`implied` in `diagnostics.correlations_applied`.

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
* **The base case is not a P50.** It sits around P59 on revenue, because most of the
  register's ranges are skewed against it.

`mc.diagnostics` names what the run could not do: parameters with no register row,
variables that are inert as configured, and whether competitor entry years were sampled
at all.

### Sensitivity: Sobol variance decomposition

```python
from forecast import Molecule, analyze

mol = Molecule.from_yaml("params/example_ibd.yaml")
s = analyze(mol, n=1024, metric="cum_net")   # ~76,000 model evaluations, ~3 min

s.table()          # S1, ST, interaction, ranked, against the section 3 tornado
s.additivity()     # sum of first-order indices; 1.0 means a purely additive model
s.interactions()   # second-order pairs, each with its own error bar
```

`S1` is the variance a driver explains alone, `ST` the variance it explains including
every interaction it takes part in. `ST - S1` is the interaction share, which a tornado
cannot see.

On the example parameters, `sum(S1) = 0.972`: the model is **97% additive** over the
register's ranges, so interactions explain about 4% of output variance and no ranking
difference is caused by them. That is not a contradiction of the multiplicative
structure -- for a product of factors each varying by 10-20%, `log Y` is very nearly a
sum, and interaction terms are second-order small. Interactions would matter if the
ranges were much wider, or if a clamp were being hit regularly.

| | Sobol ST | Sobol rank | section 3 rank |
|---|---|---|---|
| prevalence | 0.201 | 1 | 1 |
| **orderPenalty** | 0.133 | **2** | **never tested** |
| eligiblePct | 0.121 | 3 | 2 |
| brandAttr | 0.086 | 4 | -- |
| wac | 0.080 | 5 | -- |
| classCapture | 0.080 | 6 | 3 |
| diagnosedPct | 0.069 | 8 | 4 |
| segments.discount | 0.022 | 12 | 14 |
| **yearsToPeak** | 0.020 | **13** | **5** |

Two differences are worth knowing about, and neither is an interaction effect:

* **`orderPenalty` ranks second and the original tornado never tested it.** It is not in
  the JSX's driver list at all. It compounds once per competitor already established at
  launch, so with four prior entrants the register's 6-18% range moves brand
  attractiveness by a factor of 0.78 down to 0.45 -- a bigger lever than most
  epidemiology.
* **`yearsToPeak` falls from 5th to 12th because of the horizon, not the method.** Over
  12 years both indications sit at plateau for about five of them, so cumulative revenue
  is dominated by the plateau rather than the ramp. Shorten the horizon and it returns:
  its swing relative to prevalence goes 0.30x at 12 years, 0.74x at 8, and 1.38x at 6.
  The halo compresses it further, taking Crohn's entered 3.5-6.5 years down to an
  effective 2.25-4.19.

Running the section 3 tornado's own method against the *current* multi-indication model
reproduces the Sobol ranking almost exactly, which is what isolates the cause: the
ranking moved because the model became multi-indication, not because the method changed.

Two cautions the module enforces rather than leaves to the reader:

* **Sobol assumes independent inputs, so this ignores the copula.** The Saltelli
  estimator swaps columns between independent matrices; with correlated inputs those
  hybrid rows fall outside the joint distribution. The register's -0.5 prevalence /
  diagnosis-rate correlation is exactly the kind that changes apportionment, so read
  these as "variance apportioned among drivers treated as independent". Shapley effects
  are the tool for the correlated case.
* **Second-order indices are not resolvable at this sample size.** No pair of the 630
  exceeds its own confidence interval, because 4% of interaction variance spread over
  630 pairs leaves each one far below the noise floor. `interactions()` returns the
  error bars and a `resolvable` flag so the apparent ranking is not mistaken for signal.

`pos` and `discountRate` score exactly zero against cumulative net revenue, which is
correct -- it is undiscounted and unconditional on success -- and serves as a null
control on the sampling.

## The app

An interactive workbench over the same engine:

```bash
.venv/bin/pip install -e ".[app]"
streamlit run app/workbench.py
```

Parameters are on the left, results on the right across seven tabs — forecast,
by indication, gross-to-net, sensitivity, Monte Carlo, Sobol, year table. The
deterministic forecast is about a millisecond, so it recomputes on every edit;
the Monte Carlo and the Sobol decomposition are expensive and sit behind buttons
with cached results.

A banner reports whether the current parameters still reproduce the golden
master, so it is obvious when you have wandered off the reference case.

`engine.py` stays pure — the app imports it and draws what it returns. Nothing in
`app/` is imported by the library.

## The assumptions register

`params/example_register.xlsx` is a **synthetic** register, committed so the repository is
self-contained. Every value in it was authored for this repository; it is not a redacted
real register and describes no real programme. Regenerate it with
`python tools/make_example_register.py`, which is where those values live and can be
edited.

Two properties are deliberate:

* The six drivers tabulated in `docs/MODEL_DECISIONS.md` section 3 carry exactly the
  ranges that document publishes, so the Sobol-versus-tornado comparison above stays
  like for like.
* Molecule-level base values match `params/example_ibd.yaml`, so the register and the
  example parameters describe one asset.

It also preserves the awkward cases a loader has to survive: a `Derived` row with no
estimates, an `Elicited` row nobody filled in, a row for a mechanism that is documented
but not implemented, and the same fifteen-parameter coverage gap.

**A real register is not distributed.** Registers cite the data vendors behind their
values; those licences prohibit redistribution and git history is permanent, so
`reference/` is gitignored (see the data handling section of `CLAUDE.md`). To use one:

```python
load_priors(path="reference/assumptions_register.xlsx")
```

Confidential parameters belong in `params/real/`, also gitignored. Only synthetic or
public examples go in `params/`.

## Layout

| Path | |
|---|---|
| `src/forecast/params.py` | pydantic models: `Segment`, `Competitor`, `Indication`, `Molecule` |
| `src/forecast/engine.py` | the forecast. Pure: parameters in, dataframes out, no I/O |
| `src/forecast/priors.py` | register rows to PERT distributions; the I/O boundary |
| `src/forecast/mc.py` | Gaussian copula, discrete drivers, output distributions |
| `src/forecast/sobol.py` | Sobol variance decomposition, compared to the section 3 tornado |
| `params/example_ibd.yaml` | example parameters; the golden master is defined against these |
| `params/correlations.yaml` | correlation structure, meant to be edited |
| `params/example_register.xlsx` | synthetic assumptions register, so the repo is self-contained |
| `tools/make_example_register.py` | authors that register; every value is visible here |
| `app/workbench.py` | Streamlit UI over the engine; presentation only |
| `app/charts.py`, `app/theme.py` | chart builders and the validated palette |
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
