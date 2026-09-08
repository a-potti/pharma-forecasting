# Pharma forecast workbench

US pharmaceutical revenue forecasting for a multi-indication asset. Builds demand from
epidemiology upward: patient pool → clinical funnel → competitive ceiling → diffusion →
access gate → units → gross-to-net → net revenue → risk-adjusted NPV.

Current goal: port the JavaScript engine to Python, then add a Monte Carlo layer driven
by the three-point estimates already recorded in the assumptions register.

## Source material in this repo

| File | What it is |
|---|---|
| `reference/pharma_forecast_workbench.jsx` | The working engine. Treat as the specification to port, not as code to maintain. |
| `reference/assumptions_register.xlsx` | 53 assumptions with Low/Base/High, source, owner, confidence, sensitivity rank. The source of truth for parameter values. |
| `reference/forecast_model_documentation.docx` | Full mathematics. **Out of date** — documents the single-indication architecture, superseded by the multi-indication JSX. |
| `docs/MODEL_DECISIONS.md` | Why the model is built the way it is. Read this before changing engine logic. |

## Conventions

- Percentages are stored as `28`, not `0.28`. Convert at point of use only.
- All rates are annual. The engine steps in whole years.
- The register is the source of truth for parameter values. **Never hardcode an
  assumption in engine code.** New parameter → register row first.
- `engine.py` stays pure: parameters in, arrays out. No file reads, no printing, no globals.
  The Monte Carlo depends on this.
- Indication-level vs molecule-level is a hard boundary. Everything above units forks per
  indication (epidemiology, competition, uptake, access, dosing, persistence, PoS).
  Everything below units is molecule-level (price, gross-to-net, channel, valuation).
  Do not compute gross-to-net per indication — rebates are contracted at molecule level.
- Tests pass before committing.

## Golden master

The Python port must reproduce the JSX defaults exactly. Write this test first:

| Output | Expected |
|---|---|
| Peak net sales | $651M in 2037 |
| Cumulative net (12 yr) | $4,203M |
| Blended gross-to-net | 48.3% of gross |
| Risk-adjusted NPV | $1,596M total |
| — UC | $886M (NPV $1,043M × 85% PoS) |
| — Crohn's | $710M (NPV $1,092M × 65% PoS) |
| UC effective time to peak | 5.55 yr (entered 5.0, stretched by 0.77× SOV) |
| Crohn's effective time to peak | 3.22 yr (entered 4.5, compressed by halo) |
| Crohn's effective access ramp | 0.84 yr (entered 2.5) |

If these don't match, the port is wrong. Do not proceed to the Monte Carlo until they do.

## Working style

- Use Plan mode for anything touching engine math. A silent sign error is expensive here.
- Prefer small, verifiable commits over large refactors.
- When a modelling choice is ambiguous, check `docs/MODEL_DECISIONS.md` before inventing one.
- Flag anything that would change a golden-master number, before changing it.

## Data handling

Licensed vendor data (IQVIA, Komodo, MarketScan, Symphony/HealthVerity) must never be
committed — those licences prohibit redistribution and git history is permanent. Real asset
assumptions are confidential. Keep both in `params/real/`, which is gitignored. Only
synthetic or public example parameters go in `params/`.
