"""Golden master: the Python port must reproduce the JSX defaults.

Every expected value below is taken from the golden master table in CLAUDE.md, which is
itself the output of reference/pharma_forecast_workbench.jsx run on its own DEFAULTS.

Tolerance: 0.5% on money figures, 0.05 years on the time-to-peak values.

If these fail, the port is wrong -- fix the port, not the test. Anything that would move
one of these numbers gets flagged before it is changed (CLAUDE.md, working style).
"""

from pathlib import Path

import pytest

from forecast.engine import compute
from forecast.params import Molecule

PARAMS = Path(__file__).resolve().parents[1] / "params" / "example_ibd.yaml"

MONEY_TOL = 0.005  # 0.5%, relative
YEARS_TOL = 0.05  # absolute, years

M = 1e6


@pytest.fixture(scope="module")
def result():
    return compute(Molecule.from_yaml(PARAMS))


def approx_money(expected):
    return pytest.approx(expected, rel=MONEY_TOL)


def approx_years(expected):
    return pytest.approx(expected, abs=YEARS_TOL)


def indication(result, name):
    row = result.summary[result.summary["name"] == name]
    assert len(row) == 1, f"expected exactly one {name!r} in the summary"
    return row.iloc[0]


# ---- molecule level ----------------------------------------------------------------


def test_peak_net_sales(result):
    assert result.peak["net"] == approx_money(651 * M)


def test_peak_year(result):
    assert int(result.peak["year"]) == 2037


def test_cumulative_net_over_horizon(result):
    assert result.cum_net == approx_money(4203 * M)


def test_blended_gross_to_net(result):
    # 48.3% of gross, cumulative-weighted across the horizon.
    assert result.blended_gtn == pytest.approx(48.3, abs=0.05)


def test_risk_adjusted_npv_total(result):
    assert result.rnpv == approx_money(1596 * M)


# ---- indication level --------------------------------------------------------------


def test_uc_npv_and_rnpv(result):
    uc = indication(result, "Ulcerative colitis")
    assert uc["npv"] == approx_money(1043 * M)
    assert uc["pos"] == 85
    assert uc["rnpv"] == approx_money(886 * M)


def test_crohns_npv_and_rnpv(result):
    cd = indication(result, "Crohn's disease")
    assert cd["npv"] == approx_money(1092 * M)
    assert cd["pos"] == 65
    assert cd["rnpv"] == approx_money(710 * M)


def test_rnpv_is_the_sum_of_indication_rnpvs(result):
    # Risk adjustment is applied per indication before summing, never as one blended
    # PoS on the molecule (docs/MODEL_DECISIONS.md section 5).
    assert result.rnpv == approx_money(result.summary["rnpv"].sum())


# ---- halo and share of voice -------------------------------------------------------


def test_uc_effective_time_to_peak_is_stretched_by_share_of_voice(result):
    # Entered 5.0, stretched to 5.55 because UC runs at 0.77x analog support.
    uc = indication(result, "Ulcerative colitis")
    assert uc["t_star"] == approx_years(5.55)
    assert uc["sov"] == pytest.approx(0.77, abs=0.005)


def test_crohns_effective_time_to_peak_is_compressed_by_halo(result):
    # Entered 4.5, compressed to 3.22 by 90% prescriber overlap.
    cd = indication(result, "Crohn's disease")
    assert cd["t_star"] == approx_years(3.22)


def test_crohns_effective_access_ramp_is_compressed_by_halo(result):
    # Entered 2.5, compressed to 0.84 -- the molecule is already on formulary.
    cd = indication(result, "Crohn's disease")
    assert cd["ramp_years"] == approx_years(0.84)


def test_first_indication_to_launch_gets_no_halo(result):
    uc = indication(result, "Ulcerative colitis")
    assert uc["phi"] == 0.0
