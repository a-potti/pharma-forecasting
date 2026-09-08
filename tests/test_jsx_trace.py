"""Full-trace regression guard against the JSX engine.

tests/test_golden.py checks the nine headline numbers in CLAUDE.md. Those are rounded
aggregates -- three compensating errors in the chain can satisfy every one of them. This
module diffs the *whole* build, year by year and column by column, against values taken
from the JSX itself: patient pool, funnel, competitive ceiling, access index, raw and
realised share, patients, patient-years, units, price, gross, deductions, net,
discount factor, and the per-indication attribution.

The fixture is generated output, not a hand-typed expectation. It comes from
reference/pharma_forecast_workbench.jsx, whose pure-JS section (lines 114-327: helpers,
DEFAULTS, competitiveCeiling, computeAll) runs under node with no React dependency:

    sed -n '114,327p' reference/pharma_forecast_workbench.jsx > /tmp/gen.mjs
    # append a JSON.stringify of computeAll(DEFAULTS) in the shape of the fixture
    node /tmp/gen.mjs > tests/fixtures/jsx_default_trace.json

Because both engines are float64 and this port keeps the JSX's operation order, they
agree to the last bit, so the tolerance here is 1e-9 relative -- roughly a million times
tighter than the golden master's 0.5%. That is the point: at this tolerance a silent sign
error or a reordered term shows up immediately rather than hiding inside a rounded total.

If the fixture and the port disagree, regenerate the fixture first and confirm the JSX
itself has not been edited -- ``source_md5`` in the fixture pins the file it came from.
"""

import json
from pathlib import Path

import pytest

from forecast.engine import compute
from forecast.params import Molecule

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "jsx_default_trace.json"
PARAMS = ROOT / "params" / "example_ibd.yaml"

REL_TOL = 1e-9

# Columns carried through the fork, in chain order: pool -> funnel -> ceiling ->
# access -> diffusion -> patients -> units.
INDICATION_COLUMNS = [
    "prevalent",
    "addressable",
    "ceiling",
    "access_idx",
    "share_raw",
    "eff_share",
    "patients",
    "patient_years",
    "units",
]

# Columns below the merge, molecule level only.
MOLECULE_COLUMNS = ["units", "price", "gross", "ded_total", "net", "gtn_pct", "disc"]


@pytest.fixture(scope="module")
def expected():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def result():
    return compute(Molecule.from_yaml(PARAMS))


def close(actual, want):
    return actual == pytest.approx(want, rel=REL_TOL)


JSX = ROOT / "reference" / "pharma_forecast_workbench.jsx"


@pytest.mark.skipif(
    not JSX.exists(),
    reason="reference/ is not distributed with the repository; the committed fixture "
    "still checks the port, but its provenance cannot be re-verified here",
)
def test_fixture_matches_the_reference_engine_on_disk():
    """The fixture is only meaningful if the JSX it came from is unchanged."""
    import hashlib

    jsx = JSX
    digest = hashlib.md5(jsx.read_bytes()).hexdigest()
    want = json.loads(FIXTURE.read_text())["_provenance"]["source_md5"]
    assert digest == want, (
        f"{jsx.name} has changed since the fixture was generated "
        f"({digest} != {want}). Regenerate the fixture and re-verify the port."
    )


def test_parameter_file_covers_the_same_indications(expected, result):
    assert list(result.summary["name"]) == expected["indication_names"], (
        "params/example_ibd.yaml has drifted from the JSX DEFAULTS"
    )


# ---- above units: the fork ----------------------------------------------------------


@pytest.mark.parametrize("column", INDICATION_COLUMNS)
def test_indication_trace(result, expected, column):
    for i, want_ind in enumerate(expected["indications"]):
        got = result.per_indication[i].frame
        for t, want_row in enumerate(want_ind["rows"]):
            assert int(got["year"].iloc[t]) == want_row["year"]
            assert close(float(got[column].iloc[t]), want_row[column]), (
                f"{want_ind['name']} {column} in {want_row['year']}"
            )


def test_indication_derived_scalars(result, expected):
    """Halo and share-of-voice adjustments, before any revenue is computed."""
    for i, want_ind in enumerate(expected["indications"]):
        got = result.per_indication[i]
        for attr in ("t_star", "ramp_years", "seed", "sov", "months", "phi"):
            assert close(getattr(got, attr), want_ind[attr]), (
                f"{want_ind['name']} {attr}"
            )


# ---- below units: the merge ---------------------------------------------------------


@pytest.mark.parametrize("column", MOLECULE_COLUMNS)
def test_molecule_trace(result, expected, column):
    for t, want_row in enumerate(expected["molecule"]):
        assert int(result.molecule["year"].iloc[t]) == want_row["year"]
        assert close(float(result.molecule[column].iloc[t]), want_row[column]), (
            f"molecule {column} in {want_row['year']}"
        )


def test_net_revenue_attribution_by_unit_share(result, expected):
    """Net is attributed back to each indication on unit share, after one waterfall."""
    for t, want_row in enumerate(expected["attribution"]):
        for i, want in enumerate(want_row):
            assert close(float(result.attribution[f"ind_{i}"].iloc[t]), want), (
                f"attribution to indication {i} in {expected['molecule'][t]['year']}"
            )


def test_attribution_sums_back_to_molecule_net(result):
    """Nothing is created or lost in the attribution step."""
    columns = [c for c in result.attribution.columns if c.startswith("ind_")]
    assert result.attribution[columns].sum(axis=1).to_numpy() == pytest.approx(
        result.molecule["net"].to_numpy(), rel=REL_TOL
    )


# ---- totals -------------------------------------------------------------------------


def test_totals(result, expected):
    want = expected["totals"]
    assert close(result.cum_net, want["cum_net"])
    assert close(result.npv, want["npv"])
    assert close(result.rnpv, want["rnpv"])
    assert close(float(result.peak["net"]), want["peak_net"])
    assert int(result.peak["year"]) == want["peak_year"]


def test_molecule_level_rates(result, expected):
    want = expected["totals"]
    assert close(result.cov_idx, want["cov_idx"])
    assert close(result.seg_disc, want["seg_disc"])
    assert close(result.dist_fee, want["dist_fee"])
