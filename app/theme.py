"""Palette and chart chrome, from the validated reference instance.

Categorical slots 1 and 2 carry the two indications and are used in that fixed
order in every chart, so an indication keeps its colour wherever it appears.
Validated with the skill's own checker in both modes: worst adjacent CVD Delta E
24.7 light / 26.8 dark against a >= 8 target, normal-vision 33.6 / 31.8 against a
>= 15 floor, all five checks PASS.

Dark is a selected set of steps for the dark surface, not an automatic flip.
"""

LIGHT = {
    "mode": "light",
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    # categorical slots 1, 2 -- the indications
    "series": ["#2a78d6", "#eb6834"],
    # diverging poles for flows and swings: blue <-> red, neutral gray midpoint
    "pos": "#2a78d6",
    "neg": "#e34948",
    "neutral": "#f0efec",
    "total": "#184f95",
}

DARK = {
    "mode": "dark",
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": ["#3987e5", "#d95926"],
    "pos": "#3987e5",
    "neg": "#e66767",
    "neutral": "#383835",
    "total": "#86b6ef",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette(dark: bool) -> dict:
    return DARK if dark else LIGHT


def chart_config(chart, p: dict):
    """Recessive grid and axes, ink-coloured text, transparent surface."""
    return (
        chart.configure_view(strokeWidth=0, fill=None)
        .configure_axis(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=p["muted"],
            titleColor=p["ink2"],
            labelFontSize=11,
            titleFontSize=11,
            gridColor=p["grid"],
            gridWidth=1,
            domainColor=p["axis"],
            tickColor=p["axis"],
            titleFontWeight="normal",
        )
        .configure_legend(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=p["ink2"],
            titleColor=p["ink2"],
            labelFontSize=11,
            titleFontSize=11,
            symbolType="square",
            symbolSize=110,
            titleFontWeight="normal",
        )
        .configure_title(font=FONT, color=p["ink"], fontSize=13, fontWeight=600,
                         anchor="start")
    )


def money(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"${v/1e9:,.2f}B"
    if a >= 1e6:
        return f"${v/1e6:,.0f}M"
    if a >= 1e3:
        return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"


def count(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v/1e6:,.2f}M"
    if a >= 1e3:
        return f"{v/1e3:,.0f}K"
    return f"{v:,.0f}"
