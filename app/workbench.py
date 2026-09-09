"""Pharma forecast workbench -- an interface onto the Python engine.

Run with:  streamlit run app/workbench.py

engine.py stays pure; everything here is presentation. Parameters go in, the
engine returns dataframes, this module draws them. The deterministic forecast is
about a millisecond, so it recomputes on every widget change; the Monte Carlo and
the Sobol decomposition are far more expensive and are button-triggered and
cached.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import numpy as np
import pandas as pd
import streamlit as st

import charts
from forecast import Molecule, analyze, compute, load_priors, run
from forecast.engine import compute as _compute
from forecast.mc import _apply, build_variables
from forecast.priors import MissingRegisterRows
from theme import count, money, palette

PARAMS = ROOT / "params" / "example_ibd.yaml"

# The golden master from CLAUDE.md: what the JSX defaults must produce.
GOLDEN = {"peak": 651e6, "cum": 4203e6, "gtn": 48.3, "rnpv": 1596e6}

st.set_page_config(page_title="Pharma forecast workbench", layout="wide",
                   page_icon="R")


# ---------------------------------------------------------------- state


@st.cache_data(show_spinner=False)
def _defaults_json() -> str:
    return Molecule.from_yaml(PARAMS).model_dump_json()


def base_molecule() -> Molecule:
    return Molecule.model_validate_json(_defaults_json())


if "params" not in st.session_state:
    st.session_state["params"] = base_molecule().model_dump()


def reset():
    st.session_state["params"] = base_molecule().model_dump()
    for k in list(st.session_state):
        if k.startswith("w_"):
            del st.session_state[k]


P = st.session_state["params"]


def num(label, container, path, *, step, fmt="%.2f", minv=None, maxv=None, help=None):
    """A number input bound to a path into the parameter dict."""
    node = P
    for k in path[:-1]:
        node = node[k]
    key = "w_" + "_".join(str(x) for x in path)
    node[path[-1]] = container.number_input(
        label, value=float(node[path[-1]]), step=step, format=fmt,
        min_value=minv, max_value=maxv, key=key, help=help)


def check(label, container, path, *, help=None):
    """A checkbox bound to a path into the parameter dict."""
    node = P
    for k in path[:-1]:
        node = node[k]
    key = "w_" + "_".join(str(x) for x in path)
    node[path[-1]] = container.checkbox(
        label, value=bool(node[path[-1]]), key=key, help=help)


# ---------------------------------------------------------------- sidebar

sb = st.sidebar
sb.markdown("### Parameters")
dark = sb.toggle("Dark charts", value=False,
                 help="Dark is a selected set of colour steps, not an inverted palette")
pal = palette(dark)

names = [i["name"] for i in P["indications"]]
sel = sb.selectbox("Indication", range(len(names)), format_func=lambda i: names[i])
ind = P["indications"][sel]

with sb.expander("Timing & exclusivity", expanded=True):
    c = st.container()
    num("Launch year", c, ("indications", sel, "launch_year"), step=1.0, fmt="%.0f",
        help="The launch year itself gets a full year of diffusion")
    check("Loss of exclusivity", c, ("indications", sel, "loe_on"),
          help="Erosion compounds from the LOE year itself")
    if ind["loe_on"]:
        num("LOE year", c, ("indications", sel, "loe_year"), step=1.0, fmt="%.0f")
        num("Erosion per year %", c, ("indications", sel, "loe_erosion"), step=5.0,
            fmt="%.0f", minv=0.0, maxv=100.0,
            help="Share of remaining brand share lost each year, compounding")
    else:
        st.caption("Off — no erosion is applied. Switch on to model a "
                   "post-exclusivity decline for this indication.")

with sb.expander("Epidemiology funnel", expanded=True):
    c = st.container()
    num("Prevalence (per 100K)", c, ("indications", sel, "prevalence"), step=5.0,
        fmt="%.0f", minv=0.0)
    num("Diagnosed %", c, ("indications", sel, "diagnosed_pct"), step=1.0, fmt="%.0f",
        minv=0.0, maxv=100.0)
    num("Drug-treated %", c, ("indications", sel, "treated_pct"), step=1.0, fmt="%.0f",
        minv=0.0, maxv=100.0)
    num("Biologic-eligible %", c, ("indications", sel, "eligible_pct"), step=1.0,
        fmt="%.0f", minv=0.0, maxv=100.0,
        help="Moderate-to-severe, right line of therapy")

with sb.expander("Uptake & persistence"):
    c = st.container()
    num("Years to peak", c, ("indications", sel, "years_to_peak"), step=0.5,
        fmt="%.1f", minv=0.0, help="Before halo and share-of-voice adjustment")
    num("Bass p", c, ("indications", sel, "bass_p"), step=0.01, fmt="%.3f",
        minv=0.001, maxv=1.0)
    num("Bass q", c, ("indications", sel, "bass_q"), step=0.05, fmt="%.2f", minv=0.0)
    num("Persistence at 12 mo %", c, ("indications", sel, "persist12"), step=1.0,
        fmt="%.0f", minv=1.0, maxv=99.0)
    num("Compliance %", c, ("indications", sel, "compliance"), step=1.0, fmt="%.0f",
        minv=0.0, maxv=100.0)
    num("Units per patient-year", c, ("indications", sel, "units_per_year"), step=0.5,
        fmt="%.1f", minv=0.0)

with sb.expander("Competitive set"):
    c = st.container()
    num("Class capture %", c, ("indications", sel, "class_capture"), step=1.0,
        fmt="%.0f", minv=0.0, maxv=100.0)
    num("Brand attractiveness", c, ("indications", sel, "brand_attr"), step=1.0,
        fmt="%.0f", minv=0.0, help="In this indication. Only relative position matters")
    num("Order-of-entry penalty %", c, ("indications", sel, "order_penalty"), step=1.0,
        fmt="%.0f", minv=0.0, maxv=100.0)
    st.caption("Competitors are edited in the parameter file; share is derived "
               "from them, never typed.")

with sb.expander("Access"):
    c = st.container()
    num("Coverage ramp (yrs)", c, ("indications", sel, "access_ramp_years"), step=0.25,
        fmt="%.2f", minv=0.0, help="Before halo")
    num("PA / step abrasion %", c, ("indications", sel, "abrasion"), step=1.0,
        fmt="%.0f", minv=0.0, maxv=100.0)

with sb.expander("Risk & multi-indication"):
    c = st.container()
    num("Probability of success %", c, ("indications", sel, "pos"), step=5.0,
        fmt="%.0f", minv=0.0, maxv=100.0)
    num("Prescriber overlap %", c, ("indications", sel, "prescriber_overlap"), step=5.0,
        fmt="%.0f", minv=0.0, maxv=100.0, help="Drives the halo. First to launch gets none")
    num("Patient overlap %", c, ("indications", sel, "patient_overlap"), step=1.0,
        fmt="%.0f", minv=0.0, maxv=100.0)
    num("Share of field effort %", c, ("indications", sel, "sov_share"), step=5.0,
        fmt="%.0f", minv=0.0)

with sb.expander("Promotion & halo (molecule)"):
    c = st.container()
    num("Promotional intensity", c, ("promo_intensity",), step=0.1, fmt="%.2f",
        minv=0.0, help="Total field investment, x a single analog launch")
    num("Response exponent theta", c, ("sov_theta",), step=0.05, fmt="%.2f", minv=0.0,
        maxv=1.0)
    num("Halo - time to peak %", c, ("halo_t",), step=5.0, fmt="%.0f", minv=0.0,
        maxv=100.0)
    num("Halo - access ramp %", c, ("halo_a",), step=5.0, fmt="%.0f", minv=0.0,
        maxv=100.0)
    num("Halo - seeded adoption %", c, ("halo_seed",), step=1.0, fmt="%.0f", minv=0.0,
        maxv=100.0)

with sb.expander("Price & gross-to-net (molecule)"):
    c = st.container()
    num("WAC per unit ($)", c, ("wac",), step=100.0, fmt="%.0f", minv=0.0)
    num("Annual price change %", c, ("price_growth",), step=0.5, fmt="%.2f")
    num("Copay assistance %", c, ("copay_pct",), step=0.25, fmt="%.2f", minv=0.0,
        maxv=100.0)
    num("Free goods / PAP %", c, ("free_goods_pct",), step=0.25, fmt="%.2f", minv=0.0,
        maxv=100.0)
    num("Returns & other %", c, ("returns_pct",), step=0.1, fmt="%.2f", minv=0.0,
        maxv=100.0)
    num("Discount rate %", c, ("discount_rate",), step=0.5, fmt="%.2f")

sb.button("Reset to example parameters", on_click=reset, width='stretch')


# ---------------------------------------------------------------- compute

try:
    mol = Molecule.model_validate(P)
except Exception as exc:  # a bound was crossed; say so instead of crashing
    st.error(f"Invalid parameters: {exc}")
    st.stop()

F = compute(mol)
names = list(F.summary["name"])
peak = F.peak

st.markdown(f"## {mol.asset}")
st.caption(f"{len(mol.indications)} indications · US · {mol.base_year}–"
           f"{mol.base_year + mol.horizon - 1} · one molecule, one price, "
           f"one gross-to-net")

# The chain, as stat tiles. A single headline number is not a chart.
pk = int(np.argmax(F.molecule["net"].to_numpy()))
addressable = sum(float(s.frame["addressable"].iloc[pk]) for s in F.per_indication)
patients = sum(float(s.frame["patients"].iloc[pk]) for s in F.per_indication)
tiles = st.columns(5)
tiles[0].metric("Addressable patients", count(addressable), help="De-duplicated across indications")
tiles[1].metric("Patients on brand", count(patients),
                f"{patients / addressable * 100:.1f}% of addressable" if addressable else "")
tiles[2].metric("Units", count(float(peak["units"])))
tiles[3].metric("Gross sales", money(float(peak["gross"])),
                f"less {F.blended_gtn:.0f}% gross-to-net", delta_color="off")
tiles[4].metric(f"Peak net sales · {int(peak['year'])}", money(float(peak["net"])),
                f"{money(F.rnpv)} risk-adjusted NPV", delta_color="off")

# Does this parameter set still reproduce the golden master?
on_golden = (
    abs(float(peak["net"]) - GOLDEN["peak"]) / GOLDEN["peak"] < 0.005
    and abs(F.cum_net - GOLDEN["cum"]) / GOLDEN["cum"] < 0.005
    and abs(F.rnpv - GOLDEN["rnpv"]) / GOLDEN["rnpv"] < 0.005
)
if on_golden:
    st.success(r"Reproduces the golden master in CLAUDE.md: \$651M peak in 2037, "
               r"\$4,203M cumulative, 48.3% gross-to-net, \$1,596M rNPV.")
else:
    st.info("Parameters differ from the example set, so the golden master no longer "
            "applies. Reset in the sidebar to return to it.")

warn = []
mix = sum(s.mix for s in mol.segments)
if abs(mix - 100) > 0.5:
    warn.append(f"Payer mix sums to {mix:.0f}% — normalised, but check the split.")
sov = sum(i.sov_share for i in mol.indications)
if abs(sov - 100) > 0.5:
    warn.append(f"Share of field effort sums to {sov:.0f}% across indications.")
if len(mol.indications) > 1 and all(i.patient_overlap == 0 for i in mol.indications):
    warn.append("No patient overlap set — pools are summed without de-duplication.")
for w in warn:
    st.warning(w)

tabs = st.tabs(["Forecast", "By indication", "Gross-to-net", "Sensitivity",
                "Monte Carlo", "Sobol", "Year table"])

# ---------------------------------------------------------------- forecast
with tabs[0]:
    a, b, c, d = st.columns(4)
    a.metric("Cumulative net", money(F.cum_net))
    b.metric(f"NPV @ {mol.discount_rate:.0f}%", money(F.npv))
    c.metric("Risk-adjusted NPV", money(F.rnpv))
    d.metric("Gross-to-net", f"{F.blended_gtn:.1f}%")
    st.altair_chart(
        charts.revenue_by_year(F.molecule, F.attribution, names, pal),
        width='stretch')
    st.caption("Net revenue is attributed to each indication on unit share, because "
               "gross-to-net is a molecule-level rate.")
    st.altair_chart(charts.patients_by_year(F.per_indication, names, pal),
                    width='stretch')

# ---------------------------------------------------------------- by indication
with tabs[1]:
    s = F.summary.copy()
    s["% of rNPV"] = s["rnpv"] / (F.rnpv or 1) * 100
    show = pd.DataFrame({
        "Indication": s["name"],
        "Launch": [i.launch_year if i.mode == "pipeline" else "in-line"
                   for i in mol.indications],
        "Peak net": s["peak"].map(money),
        "Peak yr": s["peak_year"],
        "Cumulative": s["cum"].map(money),
        "NPV": s["npv"].map(money),
        "PoS": s["pos"].map("{:.0f}%".format),
        "rNPV": s["rnpv"].map(money),
        "% of rNPV": s["% of rNPV"].map("{:.0f}%".format),
    })
    st.dataframe(show, hide_index=True, width='stretch')
    st.caption("Risk adjustment is applied per indication before summing. One blended "
               "probability of success on the molecule would be wrong.")

    build = pd.DataFrame({
        "Indication": s["name"],
        "Addressable @ peak": s["addressable"].map(count),
        "Ceiling": s["ceiling"].map("{:.1f}%".format),
        "Realised share": s["eff_share"].map("{:.1f}%".format),
        "Patients": s["patients"].map(count),
        "SOV": s["sov"].map("{:.2f}x".format),
        "Time to peak": s["t_star"].map("{:.2f} yr".format),
        "Access ramp": s["ramp_years"].map("{:.2f} yr".format),
        "Months on therapy": s["months"].map("{:.1f}".format),
    })
    st.dataframe(build, hide_index=True, width='stretch')
    st.caption("Realised share = competitive ceiling × diffusion × access. The three "
               "are kept separable so a miss is diagnosable to one of them.")

# ---------------------------------------------------------------- gross-to-net
with tabs[2]:
    st.caption(f"Peak year {int(peak['year'])} · one waterfall applied to summed units "
               "across all indications")
    st.altair_chart(charts.gross_to_net(peak, pal), width='stretch')

# ---------------------------------------------------------------- sensitivity
with tabs[3]:
    st.caption("Each driver swung across its own register range, not a uniform "
               "percentage — under a uniform swing every linear driver ties, which "
               "measures the test rather than the forecast. Base "
               f"{money(F.cum_net).replace('$', chr(92) + '$')}.")
    try:
        priors = load_priors(require_coverage=False)
    except Exception as exc:
        st.error(f"Could not read the assumptions register: {exc}")
        priors = None
    if priors is not None:
        variables = build_variables(mol, priors)
        groups = {}
        for i, v in enumerate(variables):
            groups.setdefault(v.param, []).append(i)

        @st.cache_data(show_spinner="Swinging drivers…")
        def swings(params_json: str, n_drivers: int):
            m = Molecule.model_validate_json(params_json)
            vs = build_variables(m, load_priors(require_coverage=False))
            g = {}
            for i, v in enumerate(vs):
                g.setdefault(v.param, []).append(i)

            def at(q):
                vals = np.empty(len(vs))
                for name, idxs in g.items():
                    u = q.get(name, 0.5)
                    for k in idxs:
                        var = vs[k]
                        vals[k] = u if var.kind == "pos_outcome" else var.dist.ppf(u)
                pm, _, _ = _apply(m, vs, vals)
                return _compute(pm).cum_net

            mid = at({})
            out = []
            for name in g:
                lo, hi = at({name: 0.0}), at({name: 1.0})
                out.append({"driver": name, "low": lo - mid, "high": hi - mid,
                            "span": abs(hi - lo)})
            return (pd.DataFrame(out).sort_values("span", ascending=False)
                    .head(n_drivers).reset_index(drop=True))

        top = st.slider("Drivers shown", 5, 20, 12, key="w_torn")
        df = swings(mol.model_dump_json(), top)
        st.altair_chart(charts.tornado(df, pal), width='stretch')

# ---------------------------------------------------------------- monte carlo
with tabs[4]:
    st.caption("Drivers drawn through a Gaussian copula, not independently. "
               "Probability of success is a Bernoulli per indication, so rNPV has a "
               "real left tail rather than a haircut.")
    c1, c2 = st.columns([1, 3])
    draws = c1.select_slider("Draws", [500, 1000, 2000, 5000, 10000], value=2000,
                             key="w_draws")
    if c1.button("Run Monte Carlo", type="primary", width='stretch'):
        st.session_state["mc_key"] = (mol.model_dump_json(), draws)
    elif "mc_key" not in st.session_state:
        # Land on a distribution rather than an empty tab. 500 draws is about half
        # a second; the button re-runs at whatever the slider says.
        st.session_state["mc_key"] = (mol.model_dump_json(), 500)

    @st.cache_data(show_spinner="Sampling…")
    def run_mc(params_json: str, n: int):
        m = Molecule.model_validate_json(params_json)
        r = run(m, n_draws=n, seed=0)
        return (r.peak_net, r.cum_net, r.rnpv, r.summary(),
                r.diagnostics.n_variables,
                list(r.diagnostics.correlations_applied),
                r.diagnostics.psd_repair,
                list(r.diagnostics.inert_variables),
                dict(r.diagnostics.unsampled_parameters))

    if st.session_state.get("mc_key"):
        pj, n = st.session_state["mc_key"]
        pk_net, cum, rnpv, sm, nvar, corrs, repair, inert, unsampled = run_mc(pj, n)
        if pj != mol.model_dump_json():
            st.warning("Parameters have changed since this run. Press Run Monte "
                       "Carlo to resample.")
        st.caption(f"{n:,} draws" + ("" if n >= 2000 else
                   " — raise the slider for a smoother tail"))
        m1, m2, m3 = st.columns(3)
        for col, (arr, base, lab) in zip(
            (m1, m2, m3),
            ((pk_net, float(peak["net"]), "Peak net sales"),
             (cum, F.cum_net, "Cumulative net"),
             (rnpv, F.rnpv, "Risk-adjusted NPV")),
        ):
            q = np.percentile(arr, [10, 50, 90])
            col.markdown(f"**{lab}**")
            esc = lambda v: money(v).replace("$", chr(92) + "$")
            col.markdown(
                f"P10 {esc(q[0])} · **P50 {esc(q[1])}** · P90 {esc(q[2])}  \n"
                f"base {esc(base)} sits at P{(arr < base).mean() * 100:.0f}")
        st.altair_chart(charts.distribution(rnpv, F.rnpv, "Risk-adjusted NPV", pal),
                        width='stretch')
        d1, d2, d3 = st.columns(3)
        d1.metric("P(rNPV < 0)", f"{sm['p_rnpv_below_zero']:.3f}",
                  help="Structurally zero: net sales are floored at zero and the "
                       "model carries no costs. Use a real hurdle instead.")
        d2.metric("P(both indications fail)", f"{sm['p_all_indications_fail']:.3f}")
        d3.metric("Sampled variables", f"{nvar}")
        with st.expander(f"Diagnostics — {len(unsampled)} parameters not sampled, "
                         f"{len(inert)} inert"):
            st.markdown("**Correlations applied**")
            st.code("\n".join(corrs) or "none")
            st.caption(f"PSD repair: {repair} (0.0 means the specification was "
                       "already a valid correlation matrix)")
            st.markdown("**Sampled but inert as configured**")
            st.code("\n".join(inert) or "none")
            st.markdown("**No register row — held at their value, zero variance**")
            st.code("\n".join(sorted(unsampled)) or "none")


# ---------------------------------------------------------------- sobol
with tabs[5]:
    st.caption("Variance decomposition. S1 is what a driver explains alone; ST adds "
               "every interaction it takes part in. The gap between them is what a "
               "tornado cannot see.")
    c1, c2 = st.columns([1, 3])
    n_sob = c1.select_slider("Base sample n", [16, 32, 64, 128, 256], value=64,
                             key="w_sobol")
    c1.caption(f"≈ {n_sob * 38:,} model evaluations")
    if c1.button("Run Sobol", type="primary", width='stretch'):
        st.session_state["sob_key"] = (mol.model_dump_json(), n_sob)

    @st.cache_data(show_spinner="Decomposing variance…")
    def run_sobol(params_json: str, n: int):
        m = Molecule.model_validate_json(params_json)
        r = analyze(m, n=n, metric="cum_net", second_order=False, seed=0)
        return r.table(), r.additivity(), r.n_runs

    if st.session_state.get("sob_key"):
        pj, n = st.session_state["sob_key"]
        table, additivity, runs = run_sobol(pj, n)
        material = table[table["material"]].head(12)
        st.altair_chart(charts.sobol(material, pal), width='stretch')
        st.markdown(
            f"`sum(S1) = {additivity:.3f}` over {runs:,} evaluations. A value near 1 "
            "means the model is additive over these ranges and interactions explain "
            "little — which is the case here, despite the chain being multiplicative: "
            "for factors varying 10–20%, `log Y` is nearly a sum."
        )
        if n < 256:
            st.warning(
                f"At n={n} the first-order estimates are not converged — `sum(S1)` can "
                "exceed 1, which is impossible. The total-order ranking is stable; the "
                "S1 column is not. The published figures used n=1024.")
        zero = list(table[table["ST"] == 0]["driver"])
        if zero:
            st.caption("Exactly zero, as they must be — cumulative net revenue is "
                       "undiscounted and unconditional on success: " + ", ".join(zero))
    else:
        st.info("Press Run Sobol. n=64 takes a few seconds; n=256 about a minute.")

# ---------------------------------------------------------------- year table
with tabs[6]:
    m = F.molecule
    tbl = pd.DataFrame({"Year": m["year"].astype(int)})
    for i, nm in enumerate(names):
        tbl[f"{nm} patients"] = [count(v) for v in F.per_indication[i].frame["patients"]]
    tbl["Units"] = [count(v) for v in m["units"]]
    tbl["Net price"] = [f"${v:,.0f}" for v in m["price"]]
    tbl["Gross"] = [money(v) for v in m["gross"]]
    tbl["GTN"] = [f"{v:.0f}%" for v in m["gtn_pct"]]
    tbl["Net sales"] = [money(v) for v in m["net"]]
    st.dataframe(tbl, hide_index=True, width='stretch')
    st.download_button(
        "Download CSV", F.molecule.to_csv(index=False).encode(),
        file_name="forecast.csv", mime="text/csv")
