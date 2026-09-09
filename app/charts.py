"""Chart builders. One form per data job; colours by role, never by rank.

Every chart carries a hover layer -- an interactive chart that cannot be
interrogated is a picture of a chart. Stacked and adjacent fills carry a 2px
surface-coloured gap so segment boundaries read without relying on hue.
"""

import altair as alt
import pandas as pd

from theme import chart_config

# Vega's SI prefix renders billions as "G", so format explicitly.
_MONEY_LABEL = (
    "abs(datum.value) >= 1e9 ? '$' + format(datum.value/1e9, '.1f') + 'B'"
    " : abs(datum.value) >= 1e6 ? '$' + format(datum.value/1e6, '.0f') + 'M'"
    " : '$' + format(datum.value, ',.0f')"
)
MONEY_AXIS = alt.Axis(labelExpr=_MONEY_LABEL, title=None)


def _series_scale(names, p):
    return alt.Scale(domain=list(names), range=p["series"][: len(names)])


def revenue_by_year(molecule: pd.DataFrame, attribution: pd.DataFrame, names, p):
    """Stacked bars: net revenue per year, split by indication.

    Magnitude over time with a part-to-whole reading, so bars stacked on a
    common baseline rather than lines.
    """
    rows = []
    for i, name in enumerate(names):
        for _, r in attribution.iterrows():
            rows.append({"year": int(r["year"]), "indication": name,
                         "net": float(r[f"ind_{i}"])})
    df = pd.DataFrame(rows)

    base = alt.Chart(df).mark_bar(
        stroke=p["surface"], strokeWidth=2, cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    ).encode(
        x=alt.X("year:O", title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        y=alt.Y("sum(net):Q", axis=MONEY_AXIS, stack="zero"),
        color=alt.Color("indication:N", scale=_series_scale(names, p),
                        legend=alt.Legend(title=None, orient="top")),
        order=alt.Order("indication:N"),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("indication:N", title="Indication"),
            alt.Tooltip("net:Q", title="Net sales", format="$,.0f"),
        ],
    )
    return chart_config(base.properties(height=300), p)


def patients_by_year(per_indication, names, p):
    """Lines: patients on brand. A separate chart, never a second y-axis."""
    rows = []
    for i, name in enumerate(names):
        f = per_indication[i].frame
        for _, r in f.iterrows():
            rows.append({"year": int(r["year"]), "indication": name,
                         "patients": float(r["patients"])})
    df = pd.DataFrame(rows)

    line = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(
        size=42, stroke=p["surface"], strokeWidth=2)).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("patients:Q", axis=alt.Axis(format=",.0f", title=None)),
        color=alt.Color("indication:N", scale=_series_scale(names, p),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("indication:N", title="Indication"),
            alt.Tooltip("patients:Q", title="Patients", format=",.0f"),
        ],
    )
    return chart_config(line.properties(height=220), p)


def gross_to_net(peak, p):
    """Waterfall: one molecule-level waterfall applied to summed units.

    Diverging by role -- totals are the positive pole, deductions the negative
    one -- so the two read as opposites rather than as two categories.
    """
    steps = [
        ("Gross sales", float(peak["gross"]), "total"),
        ("Distribution fees", -float(peak["ded_dist"]), "cost"),
        ("Rebates & chargebacks", -float(peak["ded_segment"]), "cost"),
        ("Copay assistance", -float(peak["ded_copay"]), "cost"),
        ("Free goods / PAP", -float(peak["ded_free"]), "cost"),
        ("Returns & other", -float(peak["ded_returns"]), "cost"),
        ("Net sales", float(peak["net"]), "total"),
    ]
    rows, run = [], 0.0
    for name, v, kind in steps:
        if kind == "total":
            rows.append({"step": name, "lo": 0.0, "hi": v, "amount": v, "kind": kind})
            run = v
        else:
            lo = run + v
            rows.append({"step": name, "lo": lo, "hi": run, "amount": v, "kind": kind})
            run = lo
    df = pd.DataFrame(rows)

    bar = alt.Chart(df).mark_bar(
        stroke=p["surface"], strokeWidth=2, cornerRadius=4,
    ).encode(
        x=alt.X("step:N", sort=None, title=None,
                axis=alt.Axis(labelAngle=-30, labelLimit=160)),
        y=alt.Y("lo:Q", axis=MONEY_AXIS),
        y2="hi:Q",
        color=alt.Color("kind:N",
                        scale=alt.Scale(domain=["total", "cost"],
                                        range=[p["total"], p["neg"]]),
                        legend=alt.Legend(title=None, orient="top",
                                          labelExpr="datum.label == 'total' "
                                                    "? 'Total' : 'Deduction'")),
        tooltip=[alt.Tooltip("step:N", title=""),
                 alt.Tooltip("amount:Q", title="Amount", format="$,.0f")],
    )
    return chart_config(bar.properties(height=300), p)


def tornado(df: pd.DataFrame, p):
    """Diverging horizontal bars: change in cumulative net against the base."""
    long = df.melt(id_vars=["driver"], value_vars=["low", "high"],
                   var_name="side", value_name="delta")
    order = list(df["driver"])
    bar = alt.Chart(long).mark_bar(
        stroke=p["surface"], strokeWidth=2, cornerRadius=3, height=alt.RelativeBandSize(0.78),
    ).encode(
        y=alt.Y("driver:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=190)),
        x=alt.X("delta:Q", axis=MONEY_AXIS),
        color=alt.Color("side:N",
                        scale=alt.Scale(domain=["low", "high"],
                                        range=[p["neg"], p["pos"]]),
                        legend=alt.Legend(title=None, orient="top",
                                          labelExpr="datum.label == 'low' "
                                                    "? 'Low case' : 'High case'")),
        tooltip=[alt.Tooltip("driver:N", title="Driver"),
                 alt.Tooltip("side:N", title="Case"),
                 alt.Tooltip("delta:Q", title="Change", format="$,.0f")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=p["axis"], strokeWidth=1).encode(x="x:Q")
    return chart_config((bar + rule).properties(height=28 * len(order) + 30), p)


def distribution(values, base_value, label, p):
    """Histogram of one metric, with the decision points marked.

    A single series, so no legend -- the title names it. P10/P50/P90 and the
    deterministic base are direct-labelled rules rather than a second series.
    """
    import numpy as np

    df = pd.DataFrame({"value": values})
    hist = alt.Chart(df).mark_bar(
        color=p["pos"], stroke=p["surface"], strokeWidth=1,
        cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
    ).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=44),
                axis=alt.Axis(labelExpr=_MONEY_LABEL, title=label)),
        y=alt.Y("count()", axis=alt.Axis(title="Draws", format=",.0f")),
        tooltip=[alt.Tooltip("count()", title="Draws", format=",.0f"),
                 alt.Tooltip("value:Q", bin=alt.Bin(maxbins=44),
                             title=label, format="$,.0f")],
    )
    marks = pd.DataFrame({
        "x": [float(np.percentile(values, q)) for q in (10, 50, 90)] + [base_value],
        "lab": ["P10", "P50", "P90", "base"],
        "kind": ["pct", "pct", "pct", "base"],
        # The base often lands close to P50, so the two labels are stacked on
        # separate rows rather than left to collide.
        "row": [0, 0, 0, 1],
    })
    rules = alt.Chart(marks).mark_rule(strokeWidth=2).encode(
        x="x:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["pct", "base"],
                                                  range=[p["ink2"], p["neg"]]),
                        legend=None),
        strokeDash=alt.StrokeDash("kind:N", scale=alt.Scale(
            domain=["pct", "base"], range=[[1, 0], [5, 3]]), legend=None),
        tooltip=[alt.Tooltip("lab:N", title=""),
                 alt.Tooltip("x:Q", title="Value", format="$,.0f")],
    )
    # Absolute pixel positions, so the label layer does not join the y scale --
    # giving it its own quantitative y collapses the shared axis and the
    # histogram disappears. The base often lands close to P50, so the two sit on
    # separate rows rather than overprinting.
    def _label(kind, colour, ypx):
        return alt.Chart(marks[marks["kind"] == kind]).mark_text(
            align="left", dx=4, baseline="top", font="system-ui", fontSize=11,
            color=colour,
        ).encode(x="x:Q", y=alt.value(ypx), text="lab:N")

    text = _label("pct", p["ink2"], 4) + _label("base", p["neg"], 20)
    return chart_config((hist + rules + text).properties(height=280), p)


def sobol(df: pd.DataFrame, p):
    """Grouped horizontal bars: first-order against total-order.

    Two measures of the same quantity on one axis, so the gap between them
    reads directly as the interaction share.
    """
    long = df.melt(id_vars=["driver"], value_vars=["S1", "ST"],
                   var_name="index", value_name="value")
    order = list(df["driver"])
    bar = alt.Chart(long).mark_bar(
        stroke=p["surface"], strokeWidth=2, cornerRadiusTopRight=4,
        cornerRadiusBottomRight=4,
    ).encode(
        y=alt.Y("driver:N", sort=order, title=None, axis=alt.Axis(labelLimit=190)),
        yOffset=alt.YOffset("index:N", sort=["S1", "ST"]),
        x=alt.X("value:Q", axis=alt.Axis(format=".0%",
                                         title="Share of output variance")),
        color=alt.Color("index:N", scale=alt.Scale(domain=["S1", "ST"],
                                                   range=p["series"][:2]),
                        legend=alt.Legend(title=None, orient="top",
                                          labelExpr="datum.label == 'S1' "
                                                    "? 'First-order (alone)' "
                                                    ": 'Total (with interactions)'")),
        tooltip=[alt.Tooltip("driver:N", title="Driver"),
                 alt.Tooltip("index:N", title="Index"),
                 alt.Tooltip("value:Q", title="Variance share", format=".1%")],
    )
    return chart_config(bar.properties(height=34 * len(order) + 30), p)
