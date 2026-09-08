# Model decisions

Why the model is built the way it is. Each entry records a decision, the reasoning, and —
where it was measured — the size of the effect. Read before changing engine logic; several
of these look like arbitrary choices and are not.

---

## 1. Competition derives share; share is not an input

Peak share is an output of an attribute-weighted share of preference, not a typed
assumption:

```
b    = brandAttr × (1 − orderPenalty)^k        k = competitors established before launch
C(t) = classCapture × b / (b + Σ active competitor attractiveness)
```

The denominator expands when a competitor enters, so the ceiling steps down in the entry
year with no manual haircut. This is why competitor entry dates are forecast drivers in
their own right.

Attractiveness scores are meant to come from a target-product-profile conjoint, rescaled to
0–100. Only relative position matters — the absolute scale cancels. **Rebase every
competitor score together whenever the TPP changes.**

## 2. Three effects are kept multiplicative and separable

```
realised share = competitive ceiling × diffusion fraction × access index
```

A forecast that misses should be diagnosable to one of the three. Do not collapse them.
On the single-indication defaults this read 13.1% × 64% = 8.4% — the answer to "why did a
13% share assumption produce an 8% forecast."

## 3. Uniform ±20% tornado was wrong and was replaced

The original sensitivity test swung every driver ±20%. Five drivers returned a swing of
**exactly $1,314M** — class capture, price, units per patient-year, prevalence, diagnosis
rate. They tie because each enters revenue as a plain multiplier, so the test measures the
analyst's choice of swing, not uncertainty in the driver.

Replaced with per-driver plausible ranges taken from the register. The ranking changed
completely. Single-indication results, base $3,285M cumulative:

| Rank | Driver | Range | Swing |
|---|---|---|---|
| 1 | Prevalence | 110–195 /100K | $1,925M (58.6%) |
| 2 | Label-eligible | 42–66% | $1,433M |
| 3 | Class capture | 50–72% | $1,165M |
| 4 | Diagnosis rate | 55–78% | $1,111M |
| 5 | Time to peak | 3.5–6.5 yr | $1,049M |
| 14 | Gross-to-net depth | ×0.90–×1.15 | $515M |

Units per patient-year was tied for third under the uniform test and falls to fifteenth on
a realistic range — nobody is uncertain about dosing by ±20%.

**Implication for the Monte Carlo:** uniform or independent sampling reproduces this same
error. Sample each driver from its own register-derived distribution.

## 4. Promotional spend acts on time to peak, not on p

Bass `p` is by definition the external-influence coefficient, so promotion belongs there
conceptually. In this implementation it does not work, because the curve is normalised:

```
R(τ) = F(τ) / F(T*)
```

Change `p` and both numerator and denominator move; the normalisation forces the curve to
hit the ceiling at `T*` regardless. Measured, at half analog support:

- Editing `p` only: **−1.8%** on cumulative revenue
- Letting support move `T*`: **−13.6%**

Same economic assumption, 7× difference, purely from insertion point. **Model support
through `T*`**, using `T* ∝ SOV^(−θ)` with θ ≈ 0.4 as a concave default. Calibrate θ by
regressing analog time-to-peak against relative share of voice if that data exists.

Never let promotional support move the ceiling. Promotion pulls revenue forward; it does
not raise the plateau. The arithmetic supports this — `t* = ln(q/p)/(p+q)` puts `p` inside
a logarithm, so doubling it moves the inflection about six months.

Keep `q` fixed unless the *composition* of spend changes. Speaker programmes, KOL
development and publications buy contagion and do move `q`; field detailing does not.
Because `q` appears in both numerator and exponent, a `q` cut is far more damaging than a
`p` cut — peer-influence spend is structurally the highest-leverage line in the mix.

## 5. Multi-indication: fork above units, merge below

Indication-specific: epidemiology, funnel, competitive set and attractiveness, launch year,
diffusion, access ramp, PA abrasion, persistence, dosing, probability of success.

Molecule-level: price, gross-to-net waterfall, payer mix, formulary coverage, channel,
distribution fees, discount rate.

Rebates are contracted at molecule level and cannot be computed per indication. Units are
summed first, one waterfall is applied, then net revenue is attributed back to each
indication **on unit share** so risk adjustment can be applied per indication before
summing. Applying one blended PoS to the molecule is wrong.

Formulary coverage is molecule-level — payers list the molecule. Utilisation management is
written per indication, so abrasion and the coverage ramp are not shared.

## 6. Halo effect for later indications

A second indication does not launch into a naive prescriber population. Modelled through
prescriber overlap φ:

- `T*` compressed by up to 45% × φ
- Access ramp compressed by up to 80% × φ — the molecule is already on formulary and faces
  only new UM criteria, not a listing decision
- Diffusion curve seeded at `F(0) = 15% × φ` rather than zero

Measured on the IBD defaults with 90% overlap: Crohn's runs at 3.22 years to peak against
4.5 entered, and a 0.84-year ramp against 2.5. **Turning the halo off costs 11.9% of
cumulative net revenue** — larger than most epidemiology assumptions, and invisible in any
one-indication-per-model setup.

Halo scales with prescriber overlap, so an expansion from IBD into dermatology gets almost
none and behaves like a fresh launch.

## 7. Field capacity is finite

Promotional intensity is total investment as a multiple of what *one* analog launch
receives, split across indications by share of field effort. Two indications at 1.0× total
means each gets roughly half of analog support; holding both at parity requires 2.0×.

On the defaults (1.4× total) UC runs at 0.77× and its time to peak *stretches* from 5.0 to
5.55 years. This trade-off is the reason promotional intensity is molecule-level and share
of effort is per indication.

## 8. Persistence is derived from a survival curve

Evidence arrives as a curve, not as a mean. Under an exponential discontinuation hazard:

```
h  = −ln(r₁₂) / 12
m̄  = (1 − r₁₂) / h        capped at 12
```

r₁₂ = 62% implies 9.5 months. This is the mean for an *incident cohort*; a steady-state
population containing established patients shows a higher mean, so the model takes a
conservative view of a mature brand.

Use closed claims, never open. In open claims a patient who leaves the dataset is
indistinguishable from one who stops therapy, which overstates discontinuation and drags
revenue down 15–20% with no visible cause.

## 9. Diagnosis rate is the weakest parameter in the model

It is not measured. It is claims-derived diagnosed prevalence over epidemiological
prevalence — a ratio of two sources measured by different methods, inheriting both errors.
Claims cannot see undiagnosed patients: a patient with no code and a person without the
disease produce identical records.

Rules: numerator and denominator must share a case definition, a year, and a population
basis. Where they cannot, forecast from diagnosed prevalence directly and omit the
diagnosis rate.

A diagnosis rate held constant is arithmetically indistinguishable from a scaling of
prevalence — it multiplies the pool and does nothing else. It earns status as a separate
parameter only when it is expected to *move*, which requires a mechanism: a companion
diagnostic, a screening guideline change, reflex laboratory reporting, an awareness
programme.

Where testing is incidental (eGFR on a metabolic panel, LDL on a lipid panel) the
"undiagnosed but tested" population is very large — 62–64% of US stage 3 CKD patients lack
a code despite lab evidence. Where testing is deliberate (biopsy, germline panel) it is
small. Two populations sit inside it and behave differently: **known but uncoded**, who are
already addressable because the physician knows, and **genuinely missed**, who require a
detection intervention. Separable in claims by looking for disease-directed management.

## 10. Prevalence must record its case definition

The case definition — ICD code set, minimum encounters and spacing, provider specialty,
exclusion of rule-out coding, continuous-enrolment window — is the single line that
explains most disagreement between two forecasts of the same market. It lives in the
register's method note and must travel with the number.

Demand the algorithm's PPV and sensitivity against chart review before trusting it. At 0.80
sensitivity the prevalence is 20% low before any other error. At low prevalence,
specificity governs positive predictive value: 2% prevalence with a 95%-specific rule gives
a PPV near 27%.

---

# Known gaps

Carried forward deliberately. Each biases output in a knowable direction.

**Structural**
- No lines of therapy — eligibility collapses 1L/2L/3L into one fraction
- Static preference share: competitors do not respond, reposition, or reprice
- Share taken proportionally from all competitors rather than from the one displaced
- No switching costs; an established patient is as winnable as a new one
- Persistence is a single exponential; real discontinuation is front-loaded
- No channel inventory — launch stocking and destocking absent
- No ASP feedback loop for buy-and-bill units
- IRA maximum fair price documented but not implemented; it is a step change in net price
  on a known date, not a deeper rebate rate
- Free goods never convert to paid

**Method**
- Single methodology. The epidemiological funnel, preference share and diffusion are one
  chain, not independent methods. No time-series calibration, no prescriber-based bottom-up,
  no standalone analog forecast to triangulate against. Every sensitivity bar is conditional
  on the same funnel being right.
- Deterministic. Scenarios are perfectly correlated multiplier sets with no attached
  probability. This is what the Monte Carlo is meant to fix.

# Next steps

1. Port to Python; pass the golden master in `CLAUDE.md`
2. `priors.py` — register Low/Base/High → modified PERT (λ = 4; lower λ for Elicited rows)
3. `mc.py` — Gaussian copula over the marginals. Prevalence and diagnosis rate are
   **negatively** correlated (shared denominator); class capture and attractiveness
   positively. Independent sampling misstates variance in both directions.
4. Sample discrete drivers too — competitor entry years, probability of success. Sampling
   PoS is what turns rNPV from a haircut into a distribution with a real left tail.
5. Sobol variance decomposition to replace the univariate tornado
6. Reconciliation view: analog peak share, prescriber-based bottom-up, consensus, plotted
   against the model's own output, with the gap explained
7. Update the Word documentation to the multi-indication architecture
8. Add indication and scope (molecule vs indication) columns to the register
