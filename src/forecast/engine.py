"""The forecast engine, ported from reference/pharma_forecast_workbench.jsx.

Pure: parameters in, dataframes out. No file reads, no printing, no globals, no mutable
module state. The Monte Carlo depends on this.

The chain is epidemiology -> clinical funnel -> competitive ceiling -> diffusion ->
access gate -> units -> gross-to-net -> net revenue -> risk-adjusted NPV, and it forks
and merges once:

    _indication_demand()   everything ABOVE units, once per indication
    _molecule_waterfall()  everything BELOW units, once for the molecule

That boundary is enforced by the function signatures rather than by convention.
``_indication_demand`` is not given price, gross-to-net, channel or discount-rate
arguments at all, so computing a rebate per indication is a NameError instead of a
plausible-looking wrong number. Rebates are contracted at molecule level: units are
summed first, one waterfall is applied, and net revenue is attributed back to each
indication on unit share so that risk adjustment can be applied per indication before
summing (docs/MODEL_DECISIONS.md section 5).

Input clamping is kept here, matching the JSX, rather than being pushed into pydantic
bounds -- a value the JSX would accept and clamp must not become a validation error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .params import Indication, Molecule

__all__ = ["compute", "ForecastResult", "competitive_ceiling"]


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


def _or_default(v: float, fallback: float) -> float:
    """Port of the JSX ``x || fallback`` idiom, which fires when x is 0.

    The JSX also relies on this to swallow NaN, but pydantic validation means NaN cannot
    reach the engine, so only the zero case needs handling.
    """
    return v if v else fallback


def _bass_cum(t: float, p: float, q: float) -> float:
    """Cumulative Bass adoption fraction at time t.

    ``p`` is guaranteed strictly positive by the Indication model, so the q/p term is
    always finite.
    """
    if t <= 0:
        return 0.0
    e = math.exp(-(p + q) * t)
    return (1 - e) / (1 + (q / p) * e)


def _eff_months(ind: Indication) -> float:
    """Mean months on therapy per patient-year.

    In curve mode this is derived from the 12-month persistence rate under an
    exponential discontinuation hazard, h = -ln(r12)/12, mean = (1 - r12)/h. That is the
    mean for an incident cohort, so a steady-state population containing established
    patients would show a higher mean -- the model takes the conservative view
    (docs/MODEL_DECISIONS.md section 8).
    """
    if ind.persist_mode == "curve":
        r = _clamp(ind.persist12 / 100, 0.01, 0.999)
        return _clamp((1 - r) / (-math.log(r) / 12), 0, 12)
    return _clamp(ind.persist_months, 0, 12)


def competitive_ceiling(ind: Indication, year: int) -> float:
    """Peak-share ceiling as an attribute-weighted share of preference, in percent.

    Share is an output here, not an input. The denominator expands when a competitor
    enters, so the ceiling steps down in the entry year with no manual haircut
    (docs/MODEL_DECISIONS.md section 1).
    """
    brand_entry = ind.launch_year if ind.mode == "pipeline" else 0
    if ind.mode == "pipeline" and year < brand_entry:
        return 0.0

    active = [c for c in ind.competitors if year >= c.entry]
    priors = sum(1 for c in ind.competitors if c.entry < brand_entry)

    # Order-of-entry penalty, compounded once per competitor already established at
    # launch. Clamped at 90, not 100, so a late entrant never scores exactly zero.
    bonus = math.pow(1 - _clamp(ind.order_penalty, 0, 90) / 100, priors)
    brand = ind.brand_attr * bonus

    total = sum(c.attr for c in active) + brand
    if total <= 0:
        return 0.0
    return (ind.class_capture / 100) * (brand / total) * 100


def _launch_order(indications: list[Indication]) -> dict[int, int]:
    """Rank indications by launch year; rank 0 is first to market and gets no halo.

    In-line indications sort ahead of everything. Python's sort is stable, as is the
    JSX's, so ties break on the order the indications are declared.
    """
    keyed = [
        (i, -1e6 if ind.mode == "inline" else float(ind.launch_year))
        for i, ind in enumerate(indications)
    ]
    return {i: rank for rank, (i, _) in enumerate(sorted(keyed, key=lambda o: o[1]))}


# ---------------------------------------------------------------------------------
# above units: forks per indication
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class IndicationDemand:
    """One indication's demand build, everything above units."""

    name: str
    frame: pd.DataFrame
    t_star: float  # effective years to peak, after halo and share of voice
    ramp_years: float  # effective access ramp, after halo and share of voice
    seed: float  # starting point on the diffusion curve
    sov: float  # share of voice, x a single analog launch
    months: float  # mean months on therapy
    phi: float  # prescriber overlap actually applied


def _indication_demand(
    ind: Indication,
    *,
    base_year: int,
    horizon: int,
    population_m: float,
    pop_growth: float,
    cov_idx: float,
    phi: float,
    sov: float,
    sov_theta: float,
    halo_t: float,
    halo_a: float,
    halo_seed: float,
) -> IndicationDemand:
    """Build one indication from epidemiology up to units.

    Deliberately has no access to price, gross-to-net, channel or discount rate.

    ``cov_idx`` is the one molecule-level value crossing downward, and legitimately so:
    payers list the molecule, but utilisation management is written per indication, so
    the ramp and abrasion applied to it here are indication-specific
    (docs/MODEL_DECISIONS.md section 5).
    """
    # Promotional support acts on time to peak, never on the ceiling -- promotion pulls
    # revenue forward, it does not raise the plateau (MODEL_DECISIONS section 4). Halo
    # compresses both time to peak and the access ramp in proportion to prescriber
    # overlap, and seeds the diffusion curve above zero (section 6).
    sov_factor = math.pow(sov, -_clamp(sov_theta, 0, 1))
    t_star = max(0.5, ind.years_to_peak * (1 - phi * halo_t / 100) * sov_factor)
    ramp_years = max(0.15, ind.access_ramp_years * (1 - phi * halo_a / 100) * sov_factor)
    seed = phi * _clamp(halo_seed / 100, 0, 0.6)

    months = _eff_months(ind)
    anchor_ceil = _or_default(
        competitive_ceiling(
            ind, ind.launch_year if ind.mode == "pipeline" else base_year
        ),
        1.0,
    )

    pool = 0.0
    rows = []
    for t in range(horizon):
        year = base_year + t
        pop_n = population_m * 1e6 * math.pow(1 + pop_growth / 100, t)

        # -- patient pool --
        if ind.epi_mode == "incidence":
            exit_rate = _clamp((ind.mortality + ind.remission) / 100, 0, 1)
            pool = (
                pop_n * (ind.start_prevalent / 1e5)
                if t == 0
                else pool * (1 - exit_rate) + pop_n * (ind.incidence / 1e5)
            )
            prevalent = pool
        else:
            prevalent = pop_n * (ind.prevalence / 1e5)

        # -- clinical funnel, then de-duplicate against other indications' pools --
        addr_gross = (
            prevalent
            * (ind.diagnosed_pct / 100)
            * (ind.treated_pct / 100)
            * (ind.eligible_pct / 100)
        )
        addressable = addr_gross * (1 - _clamp(ind.patient_overlap / 100, 0, 0.9))

        # -- access gate. tau is 1 in the launch year, so launch gets a full year --
        tau = year - ind.launch_year + 1
        ramp = _clamp(tau / ramp_years, 0, 1) if ind.mode == "pipeline" else 1.0
        access_idx = cov_idx * ramp * (1 - _clamp(ind.abrasion / 100, 0, 0.95))

        # -- competitive ceiling --
        if ind.comp_on:
            ceiling = competitive_ceiling(ind, year)
        elif ind.mode == "pipeline":
            ceiling = ind.brand_attr
        else:
            ceiling = ind.current_share + ind.share_delta * t

        # -- diffusion --
        share_raw = 0.0
        if ind.mode == "pipeline":
            if tau > 0:
                # The curve is normalised to hit the ceiling at t_star, which is why
                # promotional support is modelled through t_star and not through p.
                fp = _or_default(_bass_cum(t_star, ind.bass_p, ind.bass_q), 1.0)
                r = _clamp(
                    seed
                    + (1 - seed) * _bass_cum(tau, ind.bass_p, ind.bass_q) / fp,
                    0,
                    1.02,
                )
                share_raw = ceiling * r
        elif ind.comp_on:
            share_raw = ind.current_share * (ceiling / anchor_ceil) + ind.share_delta * t
        else:
            share_raw = ceiling

        # -- loss of exclusivity, compounding from the LOE year itself --
        y_post = year - ind.loe_year
        if ind.loe_on and y_post >= 0:
            share_raw *= math.pow(1 - ind.loe_erosion / 100, y_post + 1)

        # -- realised share = ceiling x diffusion x access, kept separable --
        eff_share = _clamp(share_raw * access_idx, 0, 100)
        patients = addressable * eff_share / 100
        patient_years = patients * _clamp(months / 12, 0, 1) * (ind.compliance / 100)
        units = patient_years * ind.units_per_year

        rows.append(
            {
                "year": year,
                "prevalent": prevalent,
                "addressable": addressable,
                "ceiling": ceiling,
                "access_idx": access_idx * 100,  # reported as a percent
                "share_raw": share_raw,
                "eff_share": eff_share,
                "patients": patients,
                "patient_years": patient_years,
                "units": units,
            }
        )

    return IndicationDemand(
        name=ind.name,
        frame=pd.DataFrame(rows),
        t_star=t_star,
        ramp_years=ramp_years,
        seed=seed,
        sov=sov,
        months=months,
        phi=phi,
    )


# ---------------------------------------------------------------------------------
# below units: one molecule, one price, one gross-to-net
# ---------------------------------------------------------------------------------


def _molecule_waterfall(
    demands: list[IndicationDemand],
    mol: Molecule,
    *,
    dist_fee: float,
    seg_disc: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum units across indications, apply one waterfall, attribute net back.

    Returns the molecule frame and the attribution frame, the latter with one column per
    indication *position* (not name -- two indications sharing a name would otherwise
    overwrite each other's revenue).
    """
    rows = []
    attribution = []
    for t in range(mol.horizon):
        unit_split = [d.frame["units"].iloc[t] for d in demands]
        units = sum(unit_split)

        price = mol.wac * math.pow(1 + mol.price_growth / 100, t)
        gross = units * price

        ded_dist = gross * dist_fee
        ded_segment = gross * seg_disc
        ded_copay = gross * (mol.copay_pct / 100)
        ded_free = gross * (mol.free_goods_pct / 100)
        ded_returns = gross * (mol.returns_pct / 100)
        ded_total = ded_dist + ded_segment + ded_copay + ded_free + ded_returns

        net = max(gross - ded_total, 0)

        rows.append(
            {
                "year": mol.base_year + t,
                "units": units,
                "price": price,
                "gross": gross,
                "ded_dist": ded_dist,
                "ded_segment": ded_segment,
                "ded_copay": ded_copay,
                "ded_free": ded_free,
                "ded_returns": ded_returns,
                "ded_total": ded_total,
                "net": net,
                "gtn_pct": (ded_total / gross) * 100 if gross > 0 else 0.0,
                "disc": 1 / math.pow(1 + mol.discount_rate / 100, t),
            }
        )
        attribution.append(
            [net * (u / units) if units > 0 else 0.0 for u in unit_split]
        )

    molecule = pd.DataFrame(rows)
    attrib = pd.DataFrame(
        attribution, columns=[f"ind_{i}" for i in range(len(demands))]
    )
    attrib.insert(0, "year", molecule["year"])
    return molecule, attrib


# ---------------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastResult:
    """Output of one forecast run."""

    per_indication: list[IndicationDemand]
    molecule: pd.DataFrame
    attribution: pd.DataFrame  # net revenue per indication, on unit share
    summary: pd.DataFrame  # one row per indication
    cum_net: float
    npv: float
    rnpv: float  # sum of per-indication risk-adjusted NPVs
    peak: pd.Series  # the molecule row with the highest net sales
    blended_gtn: float  # total deductions / total gross across the horizon, percent
    seg_disc: float
    dist_fee: float
    cov_idx: float


def compute(mol: Molecule) -> ForecastResult:
    """Run the forecast.

    Pure. Same parameters in, same dataframes out, every time.
    """
    # -- molecule-level rates, computed once --
    mix_sum = _or_default(sum(s.mix for s in mol.segments), 100.0)
    dist_fee = (
        mol.ch_retail * mol.fee_wholesale
        + mol.ch_specialty * mol.fee_sp
        + mol.ch_buy_bill * mol.fee_wholesale
    ) / 10000
    seg_disc = sum((s.mix / mix_sum) * (s.discount / 100) for s in mol.segments)
    cov_idx = sum((s.mix / mix_sum) * (s.access / 100) for s in mol.segments)

    # -- promotional support is finite and split across indications --
    sov_sum = _or_default(sum(i.sov_share for i in mol.indications), 100.0)
    order = _launch_order(mol.indications)

    demands = []
    for idx, ind in enumerate(mol.indications):
        # The first indication to launch inherits nothing; there is no earlier
        # prescriber base to overlap with.
        phi = 0.0 if order[idx] == 0 else _clamp(ind.prescriber_overlap / 100, 0, 1)
        sov = _clamp((mol.promo_intensity * ind.sov_share) / sov_sum, 0.05, 5)
        demands.append(
            _indication_demand(
                ind,
                base_year=mol.base_year,
                horizon=mol.horizon,
                population_m=mol.population_m,
                pop_growth=mol.pop_growth,
                cov_idx=cov_idx,
                phi=phi,
                sov=sov,
                sov_theta=mol.sov_theta,
                halo_t=mol.halo_t,
                halo_a=mol.halo_a,
                halo_seed=mol.halo_seed,
            )
        )

    molecule, attribution = _molecule_waterfall(
        demands, mol, dist_fee=dist_fee, seg_disc=seg_disc
    )

    # -- per-indication summary. Risk adjustment is applied here, per indication,
    #    before summing -- one blended PoS on the molecule would be wrong. --
    disc = molecule["disc"]
    summary_rows = []
    for idx, (ind, demand) in enumerate(zip(mol.indications, demands)):
        net_by_year = attribution[f"ind_{idx}"]
        npv = float((net_by_year * disc).sum())
        pk = int(net_by_year.idxmax())  # first max wins, as in the JSX
        peak_units = molecule["units"].iloc[pk]
        summary_rows.append(
            {
                "name": ind.name,
                "pos": ind.pos,
                "cum": float(net_by_year.sum()),
                "npv": npv,
                "rnpv": npv * ind.pos / 100,
                "peak": float(net_by_year.iloc[pk]),
                "peak_year": int(molecule["year"].iloc[pk]),
                "t_star": demand.t_star,
                "sov": demand.sov,
                "ramp_years": demand.ramp_years,
                "phi": demand.phi,
                "months": demand.months,
                "ceiling": float(demand.frame["ceiling"].iloc[pk]),
                "eff_share": float(demand.frame["eff_share"].iloc[pk]),
                "patients": float(demand.frame["patients"].iloc[pk]),
                "addressable": float(demand.frame["addressable"].iloc[pk]),
                "unit_share": (
                    float(demand.frame["units"].iloc[pk] / peak_units)
                    if peak_units > 0
                    else 0.0
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    total_gross = float(molecule["gross"].sum())
    return ForecastResult(
        per_indication=demands,
        molecule=molecule,
        attribution=attribution,
        summary=summary,
        cum_net=float(molecule["net"].sum()),
        npv=float((molecule["net"] * disc).sum()),
        rnpv=float(summary["rnpv"].sum()),
        peak=molecule.iloc[int(molecule["net"].idxmax())],
        blended_gtn=(
            float(molecule["ded_total"].sum()) / total_gross * 100
            if total_gross > 0
            else 0.0
        ),
        seg_disc=seg_disc,
        dist_fee=dist_fee,
        cov_idx=cov_idx,
    )
