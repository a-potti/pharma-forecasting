"""Sobol variance decomposition, and its comparison to the section 3 tornado."""

import numpy as np
import pytest

from forecast import sobol
from forecast.params import Molecule
from forecast.priors import load_priors

PARAMS = "params/example_ibd.yaml"

# Sobol needs n * (2D + 2) evaluations with second-order indices. These runs are
# deliberately tiny: they check structure and the estimator's own bookkeeping, not
# converged index values. The converged analysis is a separate, slow exercise.
N_SMALL = 16  # power of two: Sobol sequence balance requires it


@pytest.fixture(scope="module")
def mol():
    return Molecule.from_yaml(PARAMS)


@pytest.fixture(scope="module")
def result(mol):
    return sobol.analyze(mol, n=N_SMALL, metric="cum_net", second_order=True, seed=0)


# ---- structure ----------------------------------------------------------------------


def test_one_sobol_variable_per_register_driver(mol, result):
    """Drivers are grouped, so both indications move together -- see the module docs."""
    priors = load_priors(require_coverage=False)
    from forecast.mc import build_variables

    variables = build_variables(mol, priors)
    assert len(result.names) == len({v.param for v in variables})
    assert len(result.names) < len(variables)  # grouping actually collapsed something


def test_evaluation_count_matches_the_saltelli_design(result):
    d = len(result.names)
    assert result.n_runs == N_SMALL * (2 * d + 2)


def test_table_is_ordered_by_total_order_index(result):
    t = result.table()
    assert list(t["sobol_rank"]) == list(range(1, len(t) + 1))
    assert t["ST"].is_monotonic_decreasing


def test_table_carries_the_tornado_comparison(result):
    t = result.table().set_index("driver")
    assert t.loc["prevalence", "tornado_rank"] == 1
    assert t.loc["segments.discount", "tornado_rank"] == 14
    assert np.isnan(t.loc["orderPenalty", "tornado_rank"])  # never in the tornado


def test_tornado_reference_names_real_register_parameters():
    """Guards the comparison against a typo silently dropping a driver."""
    priors = load_priors(require_coverage=False)
    for name in sobol.TORNADO_RANKING:
        assert name in priors.by_param, name


# ---- structural zeros: the strongest available correctness check ---------------------


@pytest.mark.parametrize("driver", ["pos", "discountRate"])
def test_drivers_that_cannot_affect_the_metric_score_zero(result, driver):
    """Cumulative net revenue is undiscounted and unconditional on success.

    Probability of success and the discount rate therefore cannot move it at all, so
    both indices must be zero. If either is non-zero the sampling plumbing is wrong.
    """
    i = result.names.index(driver)
    assert abs(result.S1[i]) < 1e-6
    assert abs(result.ST[i]) < 1e-6


@pytest.mark.parametrize("driver", ["incidence", "startPrevalent", "mortality", "remission"])
def test_incidence_mode_drivers_are_inert_in_prevalence_mode(result, driver):
    i = result.names.index(driver)
    assert abs(result.ST[i]) < 1e-6


def test_a_real_driver_is_not_zero(result):
    """Guards the tests above from passing because nothing is being sampled."""
    i = result.names.index("prevalence")
    assert result.ST[i] > 0.01


# ---- interactions -------------------------------------------------------------------


def test_interactions_report_their_own_error_bars(result):
    pairs = result.interactions(5)
    assert {"a", "b", "S2", "S2_conf", "resolvable"} <= set(pairs.columns)


def test_interactions_require_second_order(mol):
    first_only = sobol.analyze(mol, n=N_SMALL, second_order=False, seed=0)
    assert first_only.S2 is None
    assert first_only.resolvable_pairs() == 0
    with pytest.raises(ValueError, match="second_order=True"):
        first_only.interactions()


def test_first_order_only_uses_the_cheaper_design(mol):
    first_only = sobol.analyze(mol, n=N_SMALL, second_order=False, seed=0)
    assert first_only.n_runs == N_SMALL * (len(first_only.names) + 2)


# ---- materiality --------------------------------------------------------------------


def test_materiality_is_separate_from_statistical_resolvability(result):
    """Bootstrap error bars shrink with the estimate, so a driver explaining a
    negligible share of variance can still clear its own confidence interval."""
    t = result.table()
    assert set(t.columns) >= {"material", "above_noise"}
    assert (t["ST"] >= sobol.SobolResult.MATERIAL).equals(t["material"])


def test_interaction_share_is_not_reported_for_immaterial_drivers(result):
    """(ST - S1)/ST is meaningless when ST is near zero."""
    t = result.table()
    assert t.loc[~t["material"], "interaction_share"].isna().all()


# ---- metric selection ---------------------------------------------------------------


def test_rnpv_is_refused_with_a_reason(mol):
    with pytest.raises(ValueError, match="Bernoulli"):
        sobol.analyze(mol, n=8, metric="rnpv", second_order=False)


def test_unknown_metric_is_refused(mol):
    with pytest.raises(ValueError, match="unknown metric"):
        sobol.analyze(mol, n=8, metric="profit", second_order=False)


def test_is_reproducible(mol):
    a = sobol.analyze(mol, n=N_SMALL, second_order=False, seed=3)
    b = sobol.analyze(mol, n=N_SMALL, second_order=False, seed=3)
    assert np.array_equal(a.Y, b.Y)
    assert np.allclose(a.ST, b.ST)
