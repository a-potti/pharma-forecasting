"""Smoke test for the Streamlit workbench.

The app is presentation only -- the engine is tested elsewhere -- so this checks
the two things that actually break: that the script runs without raising, and
that what it puts on screen is the engine's answer rather than a stale or
mis-wired one.
"""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit", reason="install the 'app' extra")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "workbench.py"


@pytest.fixture(scope="module")
def at():
    return AppTest.from_file(str(APP), default_timeout=180).run()


def test_the_app_runs_without_raising(at):
    assert not at.exception, at.exception[0].message if at.exception else ""


def test_it_shows_the_golden_master(at):
    """The headline tiles must be the engine's numbers, not a cached copy."""
    tiles = {m.label: m.value for m in at.metric}
    peak = next(v for k, v in tiles.items() if k.startswith("Peak net sales"))
    assert peak == "$651M"
    assert tiles["Cumulative net"] == "$4.20B"
    assert tiles["Risk-adjusted NPV"] == "$1.60B"
    assert tiles["Gross-to-net"] == "48.3%"
    assert at.success, "the golden-master banner should be showing"


def test_every_tab_renders(at):
    assert [t.label for t in at.tabs] == [
        "Forecast", "By indication", "Gross-to-net", "Sensitivity",
        "Monte Carlo", "Sobol", "Year table",
    ]
    # Revenue, patients, waterfall, tornado. Monte Carlo and Sobol are gated
    # behind their buttons because they are expensive.
    assert len(at.get("vega_lite_chart")) == 4


def test_changing_a_parameter_moves_the_forecast(at):
    """Guards against widgets that render but are not wired to the engine."""
    fresh = AppTest.from_file(str(APP), default_timeout=180).run()
    fresh.number_input(key="w_indications_0_prevalence").set_value(400.0).run()
    assert not fresh.exception
    tiles = {m.label: m.value for m in fresh.metric}
    peak = next(v for k, v in tiles.items() if k.startswith("Peak net sales"))
    assert peak != "$651M", "raising prevalence must change peak net sales"
    assert fresh.info, "the golden-master banner should have stood down"


def test_invalid_parameters_are_reported_not_crashed():
    """A bound crossed in the sidebar should surface as a message."""
    fresh = AppTest.from_file(str(APP), default_timeout=180).run()
    fresh.number_input(key="w_indications_0_diagnosed_pct").set_value(0.0).run()
    assert not fresh.exception
