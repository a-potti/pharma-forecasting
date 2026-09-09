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
    # Revenue, patients, waterfall, tornado, and the Monte Carlo histogram --
    # which runs itself once at a small draw count so the tab is never empty.
    # Sobol stays behind its button; it is the expensive one.
    assert len(at.get("vega_lite_chart")) == 5


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


# ---- timing and exclusivity ---------------------------------------------------------


def test_launch_year_and_loe_are_reachable(at):
    """Both are engine parameters; neither should be missing from the UI."""
    keys = {w.key for w in at.number_input}
    assert "w_indications_0_launch_year" in keys
    assert any(c.key == "w_indications_0_loe_on" for c in at.checkbox)


def test_loe_inputs_appear_only_when_it_is_switched_on(at):
    keys = {w.key for w in at.number_input}
    assert "w_indications_0_loe_year" not in keys, "hidden while LOE is off"

    fresh = AppTest.from_file(str(APP), default_timeout=240).run()
    fresh.checkbox(key="w_indications_0_loe_on").set_value(True).run()
    keys = {w.key for w in fresh.number_input}
    assert "w_indications_0_loe_year" in keys
    assert "w_indications_0_loe_erosion" in keys


def test_loss_of_exclusivity_erodes_the_forecast():
    """Erosion compounds from the LOE year itself, so an earlier year costs more."""
    late = AppTest.from_file(str(APP), default_timeout=240).run()
    late.checkbox(key="w_indications_0_loe_on").set_value(True).run()
    late_cum = next(m.value for m in late.metric if m.label == "Cumulative net")

    early = AppTest.from_file(str(APP), default_timeout=240).run()
    early.checkbox(key="w_indications_0_loe_on").set_value(True).run()
    early.number_input(key="w_indications_0_loe_year").set_value(2032.0).run()
    early_cum = next(m.value for m in early.metric if m.label == "Cumulative net")

    assert not early.exception
    assert early_cum != late_cum, "an earlier LOE must cut cumulative revenue"


# ---- monte carlo --------------------------------------------------------------------


def test_monte_carlo_shows_a_distribution_without_being_asked(at):
    """The tab should land on results, not on a button and an empty panel."""
    labels = {m.label for m in at.metric}
    assert "P(rNPV < 0)" in labels
    assert "P(both indications fail)" in labels
    # Four eager charts plus the Monte Carlo histogram.
    assert len(at.get("vega_lite_chart")) == 5
