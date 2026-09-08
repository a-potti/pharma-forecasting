"""Parameter models for the forecast engine.

Ported from the DEFAULTS / IND / SEGMENTS shapes in
reference/pharma_forecast_workbench.jsx.

Two conventions from CLAUDE.md are load-bearing here:

* Percentages are stored as ``28``, not ``0.28``. Every field below whose unit is a
  percent holds the 0-100 form; the engine converts at point of use.
* Indication level and molecule level are a hard boundary. ``Indication`` carries
  everything above units -- epidemiology, funnel, competition, uptake, access, dosing,
  persistence, probability of success. ``Molecule`` carries everything below -- price,
  gross-to-net, channel, valuation -- plus the horizon and the promotional budget that
  is split across indications.

Bounds here are deliberately loose: physical sanity only. The JSX clamps several inputs
(abrasion to 0-95, patient overlap to 0-0.9, share of voice to 0.05-5) and those clamps
stay in the engine where the JSX put them, so that a value the JSX would accept and clamp
is not rejected outright here. The one exception is ``bass_p``, which must be strictly
positive -- see the note on that field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Segment(_Base):
    """One payer segment. Molecule level -- payers contract on the molecule."""

    name: str
    mix: NonNegativeFloat  # % of units; normalised across segments by the engine
    access: float = Field(ge=0, le=100)  # % formulary coverage
    discount: float = Field(ge=0, le=100)  # % gross-to-net discount


class Competitor(_Base):
    """A competing product in one indication's class.

    Only the relative position of ``attr`` matters -- the absolute scale cancels in the
    share-of-preference denominator. Rebase every competitor score together whenever the
    target product profile changes (docs/MODEL_DECISIONS.md section 1).
    """

    name: str
    entry: int  # calendar year of entry
    attr: NonNegativeFloat  # attractiveness index, nominally 0-100


class Indication(_Base):
    """Everything above units forks per indication."""

    name: str
    mode: Literal["pipeline", "inline"] = "pipeline"

    # -- epidemiology --
    epi_mode: Literal["prevalence", "incidence"] = "prevalence"
    prevalence: NonNegativeFloat = 250  # per 100K
    incidence: NonNegativeFloat = 14  # per 100K per year, incidence mode only
    start_prevalent: NonNegativeFloat = 250  # per 100K, incidence mode only
    mortality: float = Field(default=1.2, ge=0, le=100)  # % of pool per year
    remission: float = Field(default=2.0, ge=0, le=100)  # % of pool per year

    # -- clinical funnel --
    diagnosed_pct: float = Field(default=85, ge=0, le=100)
    treated_pct: float = Field(default=78, ge=0, le=100)
    eligible_pct: float = Field(default=38, ge=0, le=100)

    # -- diffusion --
    launch_year: int = 2028
    years_to_peak: NonNegativeFloat = 5  # before halo and share-of-voice adjustment
    # Bass innovation coefficient. Strictly positive: the cumulative Bass function
    # divides by p, and at p = 0 the JSX silently returns 0 (JS yields Infinity, not an
    # error) so uptake collapses to the halo seed with no visible cause. Rejecting it
    # here keeps that degenerate draw out of the engine once the Monte Carlo is sampling.
    bass_p: float = Field(default=0.04, gt=0, le=1)
    bass_q: NonNegativeFloat = 0.6  # imitation coefficient

    # -- in-line mode only --
    current_share: float = Field(default=10, ge=0, le=100)
    share_delta: float = 0.6  # percentage points per year

    # -- loss of exclusivity --
    loe_on: bool = False
    loe_year: int = 2036
    loe_erosion: float = Field(default=55, ge=0, le=100)  # % per year, compounding

    # -- competition --
    comp_on: bool = True  # derive the ceiling from share of preference
    class_capture: float = Field(default=58, ge=0, le=100)
    brand_attr: NonNegativeFloat = 76  # this molecule's attractiveness in THIS indication
    order_penalty: float = Field(default=12, ge=0, le=100)  # % per prior entrant
    competitors: list[Competitor] = Field(default_factory=list)

    # -- access --
    access_ramp_years: NonNegativeFloat = 2.5  # before halo
    abrasion: float = Field(default=14, ge=0, le=100)  # % lost to PA / step edits

    # -- persistence and dosing --
    persist_mode: Literal["curve", "months"] = "curve"
    persist12: float = Field(default=62, ge=0, le=100)  # % still on therapy at 12 months
    persist_months: NonNegativeFloat = 8.5  # used when persist_mode is "months"
    compliance: float = Field(default=87, ge=0, le=100)
    units_per_year: NonNegativeFloat = 9  # induction plus maintenance, this indication

    # -- risk --
    pos: float = Field(default=85, ge=0, le=100)  # probability of success

    # -- multi-indication couplings --
    prescriber_overlap: float = Field(default=0, ge=0, le=100)  # drives the halo
    patient_overlap: float = Field(default=0, ge=0, le=100)  # de-duplicates the pool
    sov_share: NonNegativeFloat = 50  # % of total field effort


class Molecule(_Base):
    """Everything below units is molecule level, plus horizon and promotional budget."""

    asset: str = "Asset"

    # -- horizon and population --
    base_year: int = 2026
    horizon: int = Field(default=12, ge=1)
    population_m: NonNegativeFloat = 340
    pop_growth: float = 0.5  # % per year

    # -- payer mix --
    segments: list[Segment] = Field(default_factory=list)

    # -- price and channel --
    wac: NonNegativeFloat = 6800  # $ per unit
    price_growth: float = 1.5  # % per year
    ch_retail: NonNegativeFloat = 8  # % of units
    ch_specialty: NonNegativeFloat = 70
    ch_buy_bill: NonNegativeFloat = 22
    fee_wholesale: float = Field(default=4.5, ge=0, le=100)
    fee_sp: float = Field(default=2.5, ge=0, le=100)

    # -- gross-to-net --
    copay_pct: float = Field(default=5.0, ge=0, le=100)
    free_goods_pct: float = Field(default=3.0, ge=0, le=100)
    returns_pct: float = Field(default=1.2, ge=0, le=100)

    # -- valuation --
    discount_rate: float = Field(default=9, gt=-100)

    # -- promotion and halo --
    # Total field investment as a multiple of what ONE analog launch receives, split
    # across indications by share of effort. Support acts on time to peak, never on the
    # ceiling (docs/MODEL_DECISIONS.md section 4).
    promo_intensity: NonNegativeFloat = 1.4
    sov_theta: float = Field(default=0.4, ge=0, le=1)  # time to peak ~ sov ** -theta
    halo_t: float = Field(default=45, ge=0, le=100)  # max % compression of time to peak
    halo_a: float = Field(default=80, ge=0, le=100)  # max % compression of access ramp
    halo_seed: float = Field(default=15, ge=0, le=100)  # max % seeded adoption

    indications: list[Indication] = Field(default_factory=list, min_length=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Molecule":
        """Load a parameter set from YAML.

        Lives here rather than in engine.py, which stays free of I/O so the Monte Carlo
        can call it in a tight loop.
        """
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))
