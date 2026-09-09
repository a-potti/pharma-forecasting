"""Generate params/example_register.xlsx, a synthetic assumptions register.

Run: python tools/make_example_register.py

Every value here is authored in this file, not copied from any real register, so the
output is safe to commit. Two properties are deliberate:

* The six drivers tabulated in docs/MODEL_DECISIONS.md section 3 carry exactly the
  ranges that document publishes, so the Sobol-vs-tornado comparison in the README
  remains a like-for-like check. Those ranges are already public in this repository.
* Molecule-level base values match params/example_ibd.yaml, so the register and the
  example parameters describe one asset. Indication-level rows carry a single
  representative base, because the register schema predates the multi-indication model
  and has no indication column.

Structure mirrors a real register: same 19 columns, same sheet name, same header row,
same driver-group banners, and the same awkward cases that the loader has to survive --
a Derived row with no estimates, an Elicited row nobody filled in, and a row for a
mechanism that is documented but not implemented.
"""

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

OUT = Path(__file__).resolve().parents[1] / "params" / "example_register.xlsx"

HEADERS = [
    "ID", "Driver group", "Assumption", "Model parameter", "Unit",
    "Low", "Base", "High", "Range width", "Source type",
    "Primary source", "Triangulation source", "Method / case definition",
    "Owner", "Confidence", "Sens. rank", "Exposure", "Last reviewed", "Review trigger",
]

# Generic source descriptions. Real registers name their data vendors; those licences
# prohibit redistribution, so this synthetic file describes the method instead.
CLAIMS = "National closed-claims database, continuous-enrolment cohort"
LIT = "Published prevalence meta-analysis"
ANALOG = "Analog launch benchmark set (5 immunology launches)"
ELICIT = "Internal expert elicitation panel"
PUBLIC = "Public payer filings and formulary listings"
CENSUS = "National statistical office population projections"

# group, assumption, param, unit, low, base, high, source, primary, method, owner,
# confidence, rank
ROWS = [
    ("Population & epidemiology", [
        ("US population", "populationM", "millions", 338, 340, 342, "Measured", CENSUS,
         "Resident population, all ages, mid-year", "Forecasting", "High", None),
        ("Population growth", "popGrowth", "% per year", 0.003, 0.005, 0.007, "Measured",
         CENSUS, "Compound annual, five-year projection window", "Forecasting", "High", None),
        # Section 3 publishes 110-195 /100K for this driver.
        ("Prevalence", "prevalence", "cases per 100K", 110, 145, 195, "Measured", CLAIMS,
         "CASE DEFINITION: 2+ diagnosis claims 30+ days apart, 12-month lookback. "
         "Restate this line if the definition changes -- it drives most cross-forecast "
         "disagreement.", "Forecasting / Epidemiology", "Medium", 1),
        ("Incidence", "incidence", "new cases per 100K/yr", 18, 24, 30, "Measured", CLAIMS,
         "First diagnosis with 24-month clean period. Incidence-flow mode only",
         "Epidemiology", "Medium", None),
        ("Starting prevalent pool", "startPrevalent", "cases per 100K", 110, 145, 195,
         "Measured", CLAIMS, "Same case definition as prevalence. Incidence-flow mode only",
         "Epidemiology", "Medium", None),
        ("Mortality", "mortality", "% of pool per year", 0.03, 0.045, 0.065, "Benchmarked",
         "Disease-specific survival literature",
         "All-cause, age-standardised. Incidence-flow mode only", "Epidemiology", "Medium", None),
        ("Remission / cure", "remission", "% of pool per year", 0.005, 0.015, 0.03,
         "Elicited", ELICIT, "Sustained steroid-free remission off therapy. "
         "Incidence-flow mode only", "Medical", "Low", None),
        # Section 3 publishes 55-78%.
        ("Diagnosed", "diagnosedPct", "% of prevalent", 0.55, 0.68, 0.78, "Benchmarked",
         CLAIMS, "Claims-derived diagnosed prevalence over epidemiological prevalence. "
         "Numerator and denominator must share a case definition, a year and a "
         "population basis.", "Epidemiology", "Low", 4),
        ("Drug-treated", "treatedPct", "% of diagnosed", 0.64, 0.72, 0.80, "Measured",
         CLAIMS, "Any disease-directed pharmacotherapy within 12 months of diagnosis",
         "Forecasting", "Medium", 11),
        # Section 3 publishes 42-66%.
        ("Biologic-eligible", "eligiblePct", "% of treated", 0.42, 0.55, 0.66, "Measured",
         CLAIMS, "Moderate-to-severe by proxy: prior immunomodulator or steroid burden",
         "Forecasting", "Low", 2),
    ]),
    ("Uptake & demand", [
        ("Peak brand share", "peakShare", "% of addressable", None, None, None, "Derived",
         "--", "Output of the competitive ceiling, not an input. Do not source it -- "
         "source its inputs.", "Forecasting", None, None),
        # Section 3 publishes 3.5-6.5 yr.
        ("Years to peak", "yearsToPeak", "years", 3.5, 5, 6.5, "Benchmarked", ANALOG,
         "Time from launch to plateau, before halo and share-of-voice adjustment",
         "Commercial", "Medium", 5),
        ("Innovation coefficient p", "bassP", "coefficient", 0.02, 0.04, 0.07,
         "Benchmarked", ANALOG, "Fitted to analog launch curves", "Commercial", "Medium", None),
        ("Imitation coefficient q", "bassQ", "coefficient", 0.45, 0.6, 0.8, "Benchmarked",
         ANALOG, "Fitted to analog launch curves. Moves only if spend composition changes",
         "Commercial", "Medium", None),
        ("Persistence at 12 months", "persist12", "% still on therapy", 0.50, 0.62, 0.72,
         "Measured", CLAIMS, "Closed claims only. Open claims cannot distinguish a "
         "patient who stopped from one who left the dataset.", "Forecasting", "Medium", 13),
        ("Compliance", "compliance", "% days covered", 0.82, 0.88, 0.92, "Measured",
         CLAIMS, "Proportion of days covered among persistent patients", "Forecasting",
         "High", 17),
    ]),
    ("Competitive dynamics", [
        # Section 3 publishes 50-72%.
        ("Class capture", "classCapture", "% of eligible", 0.50, 0.62, 0.72, "Measured",
         CLAIMS, "Share of biologic-eligible patients on any branded therapy",
         "Forecasting", "Medium", 3),
        ("Brand attractiveness", "brandAttr", "index 0-100", 62, 78, 90, "Elicited",
         "Target product profile conjoint",
         "Rescaled 0-100. Only relative position matters; rebase every competitor "
         "score together whenever the TPP changes.", "Commercial", "Low", 8),
        ("Competitor attractiveness", "competitors.attr", "index 0-100", None, None, None,
         "Elicited", "Target product profile conjoint",
         "NOT YET COMPLETED. Must be rebased as a set alongside brandAttr, not sampled "
         "independently.", "Commercial", "Low", 10),
        ("Order-of-entry penalty", "orderPenalty", "% per prior entrant", 0.06, 0.12, 0.18,
         "Benchmarked", ANALOG,
         "Compounded once per competitor already established at launch", "Commercial",
         "Low", 12),
        ("Competitor entry -- next in class", "competitors.entry", "year", 2028, 2029, 2031,
         "Benchmarked", "Trial registry readout dates and filing guidance",
         "Anticipated US approval year", "Competitive intelligence", "Medium", None),
        ("Competitor entry -- second wave", "competitors.entry", "year", 2032, 2033, 2035,
         "Benchmarked", "Trial registry readout dates",
         "Anticipated US approval year", "Competitive intelligence", "Medium", None),
    ]),
    ("Market access", [
        ("Coverage ramp", "accessRampYears", "years", 2, 2.5, 3.5, "Benchmarked", ANALOG,
         "Years from launch to steady-state formulary position, before halo",
         "Market access", "Medium", 5),
        ("Coverage -- Commercial", "segments.access", "% covered", 0.66, 0.76, 0.84,
         "Measured", PUBLIC, "Lives with the molecule on formulary at any tier",
         "Market access", "Medium", 9),
        ("Coverage -- Medicare Part D", "segments.access", "% covered", 0.60, 0.70, 0.77,
         "Measured", PUBLIC, "Lives with the molecule on formulary at any tier",
         "Market access", "High", 9),
        ("Coverage -- Medicare Part B", "segments.access", "% covered", 0.80, 0.88, 0.94,
         "Measured", PUBLIC, "Medically-billed benefit; coverage near universal",
         "Market access", "Medium", 9),
        ("Coverage -- Medicaid", "segments.access", "% covered", 0.48, 0.60, 0.68,
         "Measured", PUBLIC, "Preferred drug list status, weighted by state enrolment",
         "Market access", "Medium", 9),
        ("Coverage -- 340B", "segments.access", "% covered", 0.72, 0.80, 0.87, "Measured",
         PUBLIC, "Covered-entity formulary position", "Market access", "Medium", 9),
        ("Coverage -- VA / DoD / FSS", "segments.access", "% covered", 0.76, 0.85, 0.92,
         "Benchmarked", PUBLIC, "Federal supply schedule listing", "Market access",
         "Medium", 9),
        ("PA / step abrasion", "segments.abrasion", "% of covered lost", 0.08, 0.12, 0.17,
         "Measured", CLAIMS, "Share of covered patients lost to prior authorisation or "
         "step edits. Written per indication, not per payer segment.", "Market access",
         "Low", 19),
        ("Payer mix -- Commercial", "segments.mix", "% of patients", 0.44, 0.52, 0.60,
         "Measured", CLAIMS, "Share of treated patients by primary payer", "Forecasting",
         "Medium", 18),
        ("Payer mix -- Medicare Part D", "segments.mix", "% of patients", 0.14, 0.18, 0.23,
         "Measured", CLAIMS, "Share of treated patients by primary payer", "Forecasting",
         "Medium", 18),
        ("Payer mix -- Medicare Part B", "segments.mix", "% of patients", 0.03, 0.05, 0.08,
         "Measured", CLAIMS, "Share of treated patients by primary payer", "Forecasting",
         "Medium", 18),
        ("Payer mix -- Medicaid", "segments.mix", "% of patients", 0.11, 0.15, 0.20,
         "Measured", CLAIMS, "Share of treated patients by primary payer", "Forecasting",
         "Medium", 18),
        ("Payer mix -- 340B", "segments.mix", "% of patients", 0.04, 0.07, 0.11, "Measured",
         CLAIMS, "Share of treated patients by primary payer", "Forecasting", "Medium", 18),
        ("Payer mix -- VA / DoD / FSS", "segments.mix", "% of patients", 0.02, 0.03, 0.05,
         "Measured", CLAIMS, "Share of treated patients by primary payer", "Forecasting",
         "High", 18),
    ]),
    ("Price & gross-to-net", [
        ("WAC per unit", "wac", "$ per unit", 5900, 6800, 8000, "Elicited",
         "Pricing committee scenario set",
         "List price at launch, per unit of the maintenance presentation", "Pricing",
         "Medium", 7),
        ("Units per patient-year", "unitsPerYear", "units", 9, 10, 11, "Measured", CLAIMS,
         "Induction plus maintenance, weighted by regimen mix", "Forecasting", "High", 15),
        ("Annual price change", "priceGrowth", "% per year", -0.01, 0.015, 0.035,
         "Benchmarked", "Class list-price history",
         "Net of the inflation-rebate penalty", "Pricing", "Low", 6),
        ("Discount -- Commercial", "segments.discount", "% of WAC", 0.26, 0.30, 0.35,
         "Benchmarked", "Contract terms and rebate accruals", "Rebates and chargebacks",
         "Pricing", "Medium", 14),
        ("Discount -- Medicare Part D", "segments.discount", "% of WAC", 0.29, 0.34, 0.41,
         "Benchmarked", "Contract terms and rebate accruals",
         "Includes the manufacturer discount in the redesigned benefit", "Pricing",
         "Medium", 14),
        ("Discount -- Medicare Part B", "segments.discount", "% of WAC", 0.06, 0.09, 0.13,
         "Measured", "ASP reporting history", "Statutory, limited contracting", "Pricing",
         "Medium", 14),
        ("Discount -- Medicaid", "segments.discount", "% of WAC", 0.53, 0.58, 0.64,
         "Measured", "Statutory rebate schedule",
         "Statutory minimum plus CPI penalty plus supplemental", "Pricing", "High", 14),
        ("Discount -- 340B", "segments.discount", "% of WAC", 0.50, 0.55, 0.62, "Measured",
         "Ceiling price calculation", "340B ceiling price", "Pricing", "Medium", 14),
        ("Discount -- VA / DoD / FSS", "segments.discount", "% of WAC", 0.38, 0.42, 0.48,
         "Measured", "Federal supply schedule pricing",
         "Big Four pricing, statutory", "Pricing", "High", 14),
        ("IRA maximum fair price", "(not yet modeled)", "first MFP year", 2033, 2034, None,
         "Elicited", "Selection-criteria analysis",
         "DOCUMENTED, NOT IMPLEMENTED. A step change in net price on a known date, not a "
         "deeper rebate rate.", "Pricing", "Low", None),
        ("Copay assistance", "copayPct", "% of gross", 0.035, 0.05, 0.068, "Measured",
         "Copay programme redemption reporting", "Programme cost over gross sales",
         "Market access", "Medium", 16),
        ("Free goods / PAP", "freeGoodsPct", "% of gross", 0.02, 0.03, 0.045,
         "Benchmarked", "Patient assistance programme reporting",
         "Free product never converts to paid in this model", "Market access", "Medium", 16),
        ("Returns & other", "returnsPct", "% of gross", 0.008, 0.012, 0.02, "Measured",
         "Returns accrual history", "Returns, spoilage and other adjustments", "Finance",
         "High", 16),
    ]),
    ("Channel & distribution", [
        ("Retail pharmacy", "chRetail", "% of units", 0.05, 0.08, 0.12, "Measured",
         "Distribution channel reporting", "Share of units by dispensing channel",
         "Trade", "Medium", None),
        ("Specialty pharmacy", "chSpecialty", "% of units", 0.62, 0.70, 0.78, "Measured",
         "Distribution channel reporting", "Share of units by dispensing channel",
         "Trade", "Medium", None),
        ("Buy-and-bill", "chBuyBill", "% of units", 0.16, 0.22, 0.29, "Measured",
         "Distribution channel reporting", "Share of units by dispensing channel",
         "Trade", "Medium", None),
        ("Wholesaler fee", "feeWholesale", "% of gross", 0.04, 0.045, 0.055, "Measured",
         "Distribution service agreements", "Fee for service, on retail and buy-and-bill",
         "Trade", "High", None),
        ("Specialty pharmacy fee", "feeSP", "% of gross", 0.02, 0.025, 0.035, "Measured",
         "Distribution service agreements", "Fee for service, on specialty units",
         "Trade", "High", None),
    ]),
    ("Finance", [
        ("Discount rate", "discountRate", "%", 0.08, 0.09, 0.11, "Measured",
         "Corporate WACC guidance", "Nominal, post-tax", "Finance", "High", None),
        ("Probability of success", "pos", "%", 0.50, 0.65, 0.75, "Benchmarked",
         "Phase-transition benchmark set",
         "Phase 3 to approval, immunology. Applied per indication.", "Portfolio",
         "Medium", None),
    ]),
]

BLUE = Font(color="1F4E79")
BOLD = Font(bold=True)
YELLOW = PatternFill("solid", fgColor="FFF2CC")
GREY = PatternFill("solid", fgColor="EDF0F3")


def build() -> None:
    wb = Workbook()

    readme = wb.active
    readme.title = "Read me"
    for i, line in enumerate([
        ("SYNTHETIC EXAMPLE REGISTER -- not a real asset", BOLD),
        ("", None),
        ("Every number in this workbook was authored for the repository as an "
         "illustrative example.", None),
        ("It is not a redacted real register and describes no real programme.", None),
        ("", None),
        ("Two deliberate properties:", BOLD),
        ("1. The six drivers tabulated in docs/MODEL_DECISIONS.md section 3 carry exactly "
         "the ranges", None),
        ("   that document publishes, so the Sobol-vs-tornado comparison stays like for "
         "like.", None),
        ("2. Molecule-level base values match params/example_ibd.yaml, so the register "
         "and the", None),
        ("   example parameters describe one asset.", None),
        ("", None),
        ("Differences from a real register worth knowing:", BOLD),
        ("- Sources describe a method rather than naming a data vendor. Vendor licences "
         "prohibit", None),
        ("  redistribution and git history is permanent.", None),
        ("- Payer segments here match the model's six. A real register may use a "
         "different cut,", None),
        ("  which is why mc.py pools segment rows into one shock per parameter.", None),
        ("- There is still no indication column, so indication-level rows carry a single", None),
        ("  representative base. mc.py applies them as relative shocks.", None),
        ("", None),
        ("Regenerate with: python tools/make_example_register.py", None),
    ], start=1):
        c = readme.cell(i, 1, line[0])
        if line[1]:
            c.font = line[1]
    readme.column_dimensions["A"].width = 100

    ws = wb.create_sheet("Register")
    ws["A1"] = "Forecast assumptions register (synthetic example)"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = "Asset A -- IBD biologic - US - base year 2026 - 12-year horizon"
    ws["A3"] = ("Synthetic illustrative values. See the Read me tab. Range width is a "
                "formula; do not overwrite it.")

    for col, head in enumerate(HEADERS, start=1):
        c = ws.cell(5, col, head)
        c.font = BOLD
        c.fill = GREY
        c.alignment = Alignment(horizontal="right" if col > 5 else "left", wrap_text=True)

    row = 6
    ident = 1
    for group, entries in ROWS:
        banner = ws.cell(row, 1, group)
        banner.font = BOLD
        banner.fill = GREY
        row += 1
        for (assumption, param, unit, lo, base, hi, src, primary, method, owner,
             conf, rank) in entries:
            ws.cell(row, 1, ident)
            ws.cell(row, 2, group)
            ws.cell(row, 3, assumption)
            ws.cell(row, 4, param)
            ws.cell(row, 5, unit)
            for offset, v in enumerate((lo, base, hi)):
                c = ws.cell(row, 6 + offset, v)
                c.font = BLUE
            if lo is not None and hi is not None and base:
                ws.cell(row, 9, (hi - lo) / base)
            ws.cell(row, 10, src)
            ws.cell(row, 11, primary)
            ws.cell(row, 12, "--")
            ws.cell(row, 13, method)
            oc = ws.cell(row, 14, owner)
            oc.fill = YELLOW
            ws.cell(row, 15, conf)
            ws.cell(row, 16, rank)
            ws.cell(row, 18, "2026-01-15").fill = YELLOW
            ws.cell(row, 19, "Annual refresh, or any change to the case definition")
            ident += 1
            row += 1

    widths = {1: 5, 2: 24, 3: 32, 4: 20, 5: 20, 6: 10, 7: 10, 8: 10, 9: 12, 10: 13,
              11: 42, 12: 12, 13: 60, 14: 24, 15: 11, 16: 10, 17: 10, 18: 14, 19: 46}
    for col, w in widths.items():
        ws.column_dimensions[ws.cell(5, col).column_letter].width = w
    ws.freeze_panes = "A6"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT} with {ident - 1} assumptions")


if __name__ == "__main__":
    build()
