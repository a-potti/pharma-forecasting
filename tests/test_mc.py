"""Monte Carlo: the copula, the discrete drivers, and what the run reports."""

import numpy as np
import pytest
from scipy import stats

from forecast import mc
from forecast.engine import compute
from forecast.params import Molecule
from forecast.priors import DEFAULT_REGISTER, load_priors

# mc.py draws its marginals from the assumptions register. It is not distributed with
# the repository (see .gitignore), so these tests skip rather than fail on a clone that
# does not have it.
pytestmark = pytest.mark.skipif(
    not DEFAULT_REGISTER.exists(),
    reason=f"assumptions register not present at {DEFAULT_REGISTER}",
)

PARAMS = "params/example_ibd.yaml"

# Engine-backed runs are the slow part, so the statistical tests below sample the copula
# directly at high volume and the end-to-end tests use a smaller draw count.
DRAWS = 1_500
COPULA_DRAWS = 100_000


@pytest.fixture(scope="module")
def mol():
    return Molecule.from_yaml(PARAMS)


@pytest.fixture(scope="module")
def priors():
    return load_priors(require_coverage=False)


@pytest.fixture(scope="module")
def variables(mol, priors):
    return mc.build_variables(mol, priors)


@pytest.fixture(scope="module")
def result(mol):
    return mc.run(mol, n_draws=DRAWS, seed=0)


@pytest.fixture(scope="module")
def sampled(mol, variables):
    """Marginals drawn straight from the copula, without running the engine."""
    import yaml

    spec = yaml.safe_load(open("params/correlations.yaml"))
    R, _, _ = mc.build_correlation(variables, mol, spec)
    rng = np.random.default_rng(11)
    z = rng.multivariate_normal(np.zeros(len(variables)), R, size=COPULA_DRAWS,
                                method="cholesky")
    u = stats.norm.cdf(z)
    x = np.column_stack([
        v.dist.ppf(u[:, k]) if v.kind != "pos_outcome" else u[:, k]
        for k, v in enumerate(variables)
    ])
    return {v.name: x[:, k] for k, v in enumerate(variables)}


# ---- the copula ---------------------------------------------------------------------


def _gaussian_to_spearman(rho: float) -> float:
    """Rank correlation implied by a Gaussian copula correlation."""
    return 6 / np.pi * np.arcsin(rho / 2)


@pytest.mark.parametrize(
    "a,b,rho",
    [
        ("prevalence@Ulcerative colitis", "diagnosedPct@Ulcerative colitis", -0.5),
        ("prevalence@Crohn's disease", "diagnosedPct@Crohn's disease", -0.5),
        ("classCapture@Ulcerative colitis", "brandAttr@Ulcerative colitis", 0.4),
        ("classCapture@Crohn's disease", "brandAttr@Crohn's disease", 0.4),
        ("persist12@Ulcerative colitis", "persist12@Crohn's disease", 0.3),
    ],
)
def test_requested_correlations_are_realised(sampled, a, b, rho):
    """The YAML rho is the copula correlation; rank correlation is attenuated by a
    known transform, so compare against that rather than against rho itself."""
    got = stats.spearmanr(sampled[a], sampled[b]).statistic
    assert got == pytest.approx(_gaussian_to_spearman(rho), abs=0.02)


def test_correlations_are_not_independent_draws(sampled):
    """Guards the test above from passing on noise."""
    got = stats.spearmanr(
        sampled["prevalence@Ulcerative colitis"],
        sampled["diagnosedPct@Ulcerative colitis"],
    ).statistic
    assert got < -0.4


def test_unnamed_pairs_stay_independent(sampled):
    got = stats.spearmanr(sampled["wac"], sampled["discountRate"]).statistic
    assert abs(got) < 0.02


def test_correlation_is_applied_within_each_indication_separately(mol, variables):
    import yaml

    spec = yaml.safe_load(open("params/correlations.yaml"))
    _, applied, _ = mc.build_correlation(variables, mol, spec)
    within = [a for a in applied if a.startswith("within-indication")]
    assert len(within) == 2 * len(mol.indications)  # two pairs, per indication


def test_marginals_survive_the_copula(sampled, priors):
    """A copula must not move the marginals -- draws stay inside the register range."""
    for name, draws in sampled.items():
        if name.startswith("pos_outcome"):
            continue
        param = name.split("@")[0]
        if param in mc.SEGMENT_PARAMS or param == "competitors.entry":
            continue
        row = priors.by_param[param][0]
        assert draws.min() >= row.low - 1e-9, name
        assert draws.max() <= row.high + 1e-9, name


def test_inconsistent_correlations_are_repaired_not_accepted(mol, variables):
    """A hand-edited file can ask for something no joint distribution satisfies."""
    impossible = {
        "pairs": [
            {"a": "wac", "b": "discountRate", "rho": 0.99, "scope": "explicit"},
            {"a": "wac", "b": "copayPct", "rho": -0.99, "scope": "explicit"},
            {"a": "discountRate", "b": "copayPct", "rho": 0.99, "scope": "explicit"},
        ]
    }
    R, applied, moved = mc.build_correlation(variables, mol, impossible)
    assert len(applied) == 3
    assert moved > 0, "an impossible matrix should register a repair"
    assert np.linalg.eigvalsh(R).min() > -1e-8, "repaired matrix must be PSD"
    assert np.allclose(np.diag(R), 1.0)


def test_a_consistent_matrix_is_left_alone(mol, variables):
    import yaml

    spec = yaml.safe_load(open("params/correlations.yaml"))
    _, _, moved = mc.build_correlation(variables, mol, spec)
    assert moved == 0.0


# ---- discrete drivers ---------------------------------------------------------------


def test_probability_of_success_is_bernoulli_not_a_haircut(result):
    """Each draw either books an indication's revenue or none of it."""
    assert result.successes.dtype == bool
    assert set(np.unique(result.successes)) <= {True, False}
    assert 0 < result.successes.mean() < 1


def test_success_rate_tracks_the_probability_of_success(result, mol):
    rates = result.summary()["p_success"]
    for ind in mol.indications:
        # The probability is itself sampled, so allow for that plus sampling noise.
        assert rates[ind.name] == pytest.approx(ind.pos / 100, abs=0.05)


def test_bernoulli_preserves_the_expected_value(result):
    """Sampling PoS changes the shape of rNPV, not its mean.

    E[rnpv] should sit near the mean unconditional NPV weighted by success rate.
    """
    per_success = result.successes.mean(axis=0)
    assert result.rnpv.mean() > 0
    assert result.rnpv.mean() < result.npv.mean()
    assert result.rnpv.mean() == pytest.approx(
        result.npv.mean() * per_success.mean(), rel=0.15
    )


def test_entry_years_are_sampled_only_for_future_entrants(mol, priors):
    """Every competitor in the example parameters entered before the base year."""
    variables = mc.build_variables(mol, priors)
    assert sum(1 for v in variables if v.kind == "entry_year") == 0


def test_entry_years_are_sampled_when_a_future_entrant_exists(mol, priors):
    data = mol.model_dump()
    data["indications"][0]["competitors"].append(
        {"name": "Future oral", "entry": 2031, "attr": 72}
    )
    future = Molecule.model_validate(data)
    variables = mc.build_variables(future, priors)
    entry = [v for v in variables if v.kind == "entry_year"]
    assert len(entry) == 1
    draws = entry[0].dist.rvs(500, random_state=1)
    assert draws.min() >= 2028 and draws.max() <= 2031  # register row 21's range


def test_entry_years_are_whole_years(mol, priors):
    data = mol.model_dump()
    data["indications"][0]["competitors"].append(
        {"name": "Future oral", "entry": 2031, "attr": 72}
    )
    future = Molecule.model_validate(data)
    variables = mc.build_variables(future, priors)
    values = np.array([v.dist.ppf(0.37) if v.dist is not None else 0.5 for v in variables])
    perturbed, _, _ = mc._apply(future, variables, values)
    entries = [c.entry for c in perturbed.indications[0].competitors]
    assert all(isinstance(e, int) for e in entries)


# ---- the run ------------------------------------------------------------------------


def test_run_is_reproducible(mol):
    a = mc.run(mol, n_draws=200, seed=42)
    b = mc.run(mol, n_draws=200, seed=42)
    assert np.array_equal(a.cum_net, b.cum_net)
    assert np.array_equal(a.rnpv, b.rnpv)


def test_different_seeds_give_different_draws(mol):
    a = mc.run(mol, n_draws=200, seed=1)
    b = mc.run(mol, n_draws=200, seed=2)
    assert not np.array_equal(a.cum_net, b.cum_net)


def test_no_draw_fails(result):
    assert result.diagnostics.failures == 0
    assert not np.isnan(result.cum_net).any()


def test_quantiles_are_ordered(result):
    for metric in ("peak_net", "cum_net", "npv", "rnpv"):
        q = result.quantiles(metric)
        assert q[10] < q[50] < q[90], metric


def test_summary_reports_the_requested_figures(result):
    s = result.summary()
    for metric in ("peak_net", "cum_net", "rnpv"):
        assert set(s[metric]) == {10, 50, 90}
    assert "p_rnpv_below_zero" in s


def test_rnpv_cannot_go_below_zero_in_this_model(result):
    """Net sales are floored at zero and there are no costs, so rNPV >= 0 always.

    The reported probability is therefore structurally zero, not an estimate. Use
    p_below(threshold) against a real hurdle rate for a meaningful downside question.
    """
    assert result.summary()["p_rnpv_below_zero"] == 0.0
    assert (result.rnpv >= 0).all()


def test_p_below_answers_a_meaningful_threshold(result):
    median = float(np.percentile(result.rnpv, 50))
    assert result.p_below(median) == pytest.approx(0.5, abs=0.03)
    assert result.p_below(0.0) == 0.0


def test_the_base_case_is_carried_for_reference(result, mol):
    assert result.base.cum_net == compute(mol).cum_net


def test_spread_is_wide_enough_to_be_informative(result):
    q = result.quantiles("cum_net")
    assert (q[90] - q[10]) / q[50] > 0.2


# ---- diagnostics --------------------------------------------------------------------


def test_diagnostics_name_the_inert_variables(result):
    """An inert driver looks identical to an unimportant one in the output."""
    inert = " ".join(result.diagnostics.inert_variables)
    assert "segments.mix" in inert  # cancels under the engine's mix normalisation
    assert "prevalence mode" in inert  # incidence-flow parameters are unused


def test_diagnostics_report_unsampled_parameters(result):
    """The register's coverage gap has to survive into the Monte Carlo output."""
    unsampled = result.diagnostics.unsampled_parameters
    assert "Molecule.promo_intensity" in unsampled
    assert "Indication.prescriber_overlap" in unsampled


def test_diagnostics_record_pooling_and_entry_years(result):
    assert result.diagnostics.entry_year_variables == 0
    assert "segments.discount" in result.diagnostics.pooled_segment_params


def test_every_variable_is_accounted_for(result, variables):
    assert result.diagnostics.n_variables == len(variables)
