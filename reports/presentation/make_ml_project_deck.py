"""Build the 10-slide ML project deck (16:9) from verified project numbers.

Deliverable: reports/presentation/ML_PROJECT_DECK.pptx
Every figure used here is traceable to evaluation/metrics/*.csv or
datasets/metadata/target_feasibility.json; see the speaker notes per slide.
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("ML_PROJECT_DECK.pptx")

NAVY = RGBColor(0x12, 0x37, 0x4F)
TEAL = RGBColor(0x2D, 0x7F, 0xA6)
LIGHT = RGBColor(0xE8, 0xEE, 0xF3)
PALE = RGBColor(0xF6, 0xF9, 0xFB)
GREY = RGBColor(0x33, 0x33, 0x33)
MID = RGBColor(0x5A, 0x6B, 0x78)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xB5, 0x7A, 0x1F)
FAIL = RGBColor(0x9E, 0x3B, 0x32)

W, H = 13.333, 7.5
FOOT = "Predicting district-level prevalence of dilapidated census houses in India  |  Census 2011"


def textbox(slide, x, y, w, h, text, size=14, bold=False, color=GREY, align=PP_ALIGN.LEFT,
            italic=False, anchor=MSO_ANCHOR.TOP, line_spacing=1.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    para = frame.paragraphs[0]
    para.alignment = align
    para.line_spacing = line_spacing
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return box


def bullets(slide, x, y, w, h, items, size=13.5, color=GREY, gap=7, marker="\u25aa  "):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.space_after = Pt(gap)
        para.line_spacing = 1.05
        bold = False
        if item.startswith("**") and "**" in item[2:]:
            head, rest = item[2:].split("**", 1)
            run = para.add_run()
            run.text = marker + head
            run.font.bold = True
            run.font.size = Pt(size)
            run.font.color.rgb = NAVY
            run.font.name = "Calibri"
            run2 = para.add_run()
            run2.text = rest
            run2.font.size = Pt(size)
            run2.font.color.rgb = color
            run2.font.name = "Calibri"
            continue
        run = para.add_run()
        run.text = marker + item
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return box


def header(slide, number, title, subtitle=None):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(W), Inches(1.02))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False
    textbox(slide, 0.45, 0.16, 1.0, 0.6, f"{number:02d}", 26, True, TEAL)
    textbox(slide, 1.28, 0.14 if subtitle else 0.24, 11.6, 0.5, title, 23, True, WHITE)
    if subtitle:
        textbox(slide, 1.28, 0.62, 11.6, 0.34, subtitle, 11.5, False, RGBColor(0xC6, 0xD6, 0xE2))


def footer(slide, number):
    line = slide.shapes.add_shape(1, Inches(0.45), Inches(7.02), Inches(W - 0.9), Inches(0.012))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xC9, 0xD5, 0xDE)
    line.line.fill.background()
    line.shadow.inherit = False
    textbox(slide, 0.45, 7.09, 10.5, 0.26, FOOT, 8.5, False, MID)
    textbox(slide, W - 1.35, 7.09, 0.9, 0.26, f"{number} / 10", 8.5, False, MID, PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, title, fill=PALE, edge=LIGHT, title_color=NAVY):
    shape = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = edge
    shape.line.width = Pt(1)
    shape.shadow.inherit = False
    textbox(slide, x + 0.22, y + 0.14, w - 0.44, 0.32, title, 12, True, title_color)
    return y + 0.52


def band(slide, x, y, w, h, text, fill, text_color=WHITE, size=12.5, bold=True):
    shape = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    frame = shape.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = Inches(0.18)
    frame.margin_right = Inches(0.18)
    para = frame.paragraphs[0]
    para.alignment = PP_ALIGN.LEFT
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = text_color
    run.font.name = "Calibri"
    return shape


def table(slide, data, x, y, w, col_widths, row_h=0.34, head_h=0.42, font=11.5,
          head_font=11.5, first_bold=False, align=None, highlight_rows=(), highlight_fill=LIGHT):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w),
                                   Inches(head_h + row_h * (rows - 1)))
    tbl = shape.table
    tbl.first_row = True
    total = sum(col_widths)
    for index, width in enumerate(col_widths):
        tbl.columns[index].width = Inches(w * width / total)
    for r in range(rows):
        tbl.rows[r].height = Inches(head_h if r == 0 else row_h)
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.text = ""
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r in highlight_rows:
                cell.fill.fore_color.rgb = highlight_fill
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else PALE
            frame = cell.text_frame
            frame.word_wrap = True
            para = frame.paragraphs[0]
            if align and c < len(align):
                para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[align[c]]
            run = para.add_run()
            run.text = str(data[r][c])
            run.font.name = "Calibri"
            run.font.size = Pt(head_font if r == 0 else font)
            run.font.bold = r == 0 or (first_bold and c == 0)
            run.font.color.rgb = WHITE if r == 0 else GREY
    return shape


def note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def new(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    # ---------------------------------------------------------------- 1
    s = new(prs)
    header(s, 1, "The baseline paper: where this project starts",
           "Predicting district-level prevalence of dilapidated census houses in India using machine learning")
    y = card(s, 0.45, 1.30, 6.05, 3.75, "BASELINE STUDY  \u2014  [13]")
    textbox(s, 0.67, y - 0.05, 5.6, 0.3,
            "Ploton et al., Nature Communications 11:4540, 2020", 11, True, TEAL)
    bullets(s, 0.67, y + 0.30, 5.6, 2.9, [
        "Large-scale ecological mapping models fitted on spatial covariates.",
        "Random cross-validation reported substantial explanatory power.",
        "Under spatial validation, predictive performance collapsed toward null.",
        "Conclusion: non-spatial validation overstates generalisation for geographically structured data.",
    ], 12.5)

    y = card(s, 6.72, 1.30, 6.16, 3.75, "PROTOCOL STUDY  \u2014  [12]")
    textbox(s, 6.94, y - 0.05, 5.7, 0.3,
            "Roberts et al., Ecography 40(8):913\u2013929, 2017", 11, True, TEAL)
    bullets(s, 6.94, y + 0.30, 5.7, 2.9, [
        "Catalogued cross-validation strategies for temporal, spatial, hierarchical and clustered data.",
        "Grouped or blocked folds are required when observations share a cluster.",
        "Supplies the design rule adopted here: never split districts of one state across train and test.",
    ], 12.5)

    band(s, 0.45, 5.28, 12.43, 0.62,
         "Adopted as the baseline protocol: fold by state; report non-grouped CV only as a reference.",
         TEAL)
    textbox(s, 0.45, 5.98, 12.43, 0.9,
            "Neither paper studies housing, and neither studies India. What this project tests is whether that "
            "protocol transfers to Indian district housing data \u2014 and what the resulting model can and cannot support.",
            12.5, False, GREY)
    footer(s, 1)
    note(s, "Baseline = the protocol we inherit, not a model we replicate. [12] Roberts et al. 2017 Ecography "
            "40(8):913-929; [13] Ploton et al. 2020 Nat. Commun. 11:4540, doi:10.1038/s41467-020-18321-y.\n"
            "Source: reports/paper/paper_content.py section II (Related Work).")

    # ---------------------------------------------------------------- 2
    s = new(prs)
    header(s, 2, "Problem, scope and objectives")
    y = card(s, 0.45, 1.30, 6.05, 4.05, "THE PROBLEM")
    bullets(s, 0.67, y, 5.6, 3.4, [
        "Housing condition is a core dimension of deprivation; Census 2011 reports Good / Livable / Dilapidated for every district.",
        "Question: can district-level prevalence of dilapidated census houses be predicted from demographic, household, infrastructure and socioeconomic features?",
        "Practical use: screen districts whose housing stock is likely worst when only covariate data are available.",
    ], 12.5)

    y = card(s, 6.72, 1.30, 6.16, 4.05, "SCOPE GUARDRAILS", fill=RGBColor(0xFB, 0xF3, 0xF2),
             edge=RGBColor(0xF0, 0xD9, 0xD6), title_color=FAIL)
    bullets(s, 6.94, y, 5.7, 3.4, [
        "District-level aggregates only \u2014 no household or individual prediction.",
        "Historical 2011 snapshot \u2014 not present-day conditions.",
        "Observational and predictive \u2014 no causal claim.",
        "Target is dilapidated census houses \u2014 this is not a homelessness model.",
        "Predictive associations are reported as such, never as effects.",
    ], 12.5)

    band(s, 0.45, 5.52, 12.43, 0.95,
         "Five objectives:  validate the target definition  \u00b7  compare configuration A vs B  \u00b7  benchmark 10 model "
         "families under nested state-grouped CV  \u00b7  analyse errors and the extreme tail  \u00b7  test tail-aware remedies "
         "against pre-registered gates", NAVY, WHITE, 12)
    footer(s, 2)
    note(s, "Scope language is enforced by the paper's QC script: 'homeless', 'causal effect', 'present-day' appear only "
            "inside explicit negations. Source: paper_content.py sections I and VII.")

    # ---------------------------------------------------------------- 3
    s = new(prs)
    header(s, 3, "Data and target construction")
    table(s, [
        ["Source", "Shape", "Geographic level"],
        ["D1  india-districts-census-2011", "640 \u00d7 118", "District"],
        ["D2  housing-hlpca-full", "1,908 \u00d7 156", "District \u00d7 rural/urban"],
        ["D5  census_2011", "1,342 \u00d7 89", "District \u00d7 rural/urban"],
    ], 0.45, 1.32, 6.2, [0.44, 0.24, 0.32], row_h=0.44, head_h=0.40, font=11.5, align="lcc")

    bullets(s, 0.45, 3.20, 6.2, 3.0, [
        "**Target T2:** Census-computed share of dilapidated census houses (D2 Total row per district).",
        "**Rejected formula:** dilapidated households \u00f7 D1 \u201cHouseholds\u201d \u2014 D1 totals 330.8 M against 246.7 M actual households, the wrong denominator universe.",
        "**Leakage control:** D2\u2019s condition-of-house block and D1 Households are excluded from all predictors.",
    ], 12)

    y = card(s, 6.85, 1.32, 6.03, 2.45, "INDEPENDENT TARGET CHECK")
    bullets(s, 7.07, y - 0.02, 5.6, 2.0, [
        "T2 vs count-based T3 (n = 628): Pearson 0.9994, Spearman 0.9992.",
        "Median absolute difference 0.053 pp.",
        "640/640 districts map from D2; 628 have a reliable D5 denominator.",
    ], 12)

    y = card(s, 6.85, 3.95, 6.03, 2.05, "TARGET DISTRIBUTION (n = 640)", fill=LIGHT, edge=LIGHT)
    textbox(s, 7.07, y, 5.6, 1.4,
            "mean 5.04 pp   \u00b7   median 4.40 pp   \u00b7   p95 11.2 pp\n"
            "max 17.7 pp   \u00b7   sd 3.12   \u00b7   skew 1.15",
            13.5, False, NAVY)
    textbox(s, 7.07, y + 0.75, 5.6, 0.6, "Right-skewed, bounded at zero, with a long upper tail.",
            11.5, False, MID, italic=True)

    band(s, 0.45, 6.14, 12.43, 0.55,
         "Composition check: Good + Livable + Dilapidated sum to 100 \u00b1 0.1 pp in all 640 districts.",
         RGBColor(0xE3, 0xEF, 0xE6), RGBColor(0x2C, 0x5F, 0x3A), 11.5)
    footer(s, 3)
    note(s, "Sources: datasets/metadata/target_feasibility.json (T2 vs T3: 0.9994 / 0.9992, median |diff| 0.053 pp, "
            "component sum max deviation 0.1000 pp); datasets/processed/district_analysis.csv (640 x 83).\n"
            "Note the discrepancy vs the original brief: additivity is +/-0.1 pp, not +/-0.5 pp.")

    # ---------------------------------------------------------------- 4
    s = new(prs)
    header(s, 4, "Methodology: features, models and validation design")
    y = card(s, 0.45, 1.28, 6.05, 3.55, "TWO FEATURE CONFIGURATIONS")
    bullets(s, 0.67, y, 5.6, 2.9, [
        "81 features built; 4 dropped at |r| > 0.98 \u2192 77 retained.",
        "**A:** 640 districts, 43 predictors, D1 socioeconomic and amenity shares, full coverage.",
        "**B:** 628 districts, 77 predictors, A plus 34 D5 household-normalised housing variables.",
        "Education block corrected to stop double-counting the Literate subtotal.",
        "A few B rates exceed 1 by at most 6.2%; documented, not clipped.",
    ], 11.5)

    y = card(s, 6.72, 1.28, 6.16, 3.55, "MODELS AND VALIDATION")
    bullets(s, 6.94, y, 5.7, 2.9, [
        "10 families: dummy, Ridge, Elastic Net, kNN, decision tree, RF, Extra Trees, gradient boosting, HistGB, SVR-RBF.",
        "Nested CV: 5 outer GroupKFold folds on state (28 train / 7 test states) \u00d7 4 inner folds.",
        "Imputation and scaling fitted inside every training partition; 76 candidate configurations.",
        "Random 5-fold nested CV run only as a secondary reference.",
    ], 11.5)

    band(s, 0.45, 5.05, 12.43, 0.72,
         "Disclosed limitation: the Phase 2 correlation screen and collinearity drop were computed once on the full "
         "sample, not inside each fold. The feature set is therefore not fully nested, and the paper does not claim "
         "that historical feature selection was fold-isolated.", AMBER, WHITE, 11.5)
    textbox(s, 0.45, 5.92, 12.43, 0.5,
            "Screen maximum absolute target correlation was 0.528 and removed no feature on target grounds, which "
            "bounds the effect without removing the limitation.", 11.5, False, MID, italic=True)
    footer(s, 4)
    note(s, "Sources: datasets/processed/feature_manifest.csv (81 records, 77 retained); "
            "evaluation/metrics/phase3_nested_cv_results.csv (76 configurations, 5x4 nested GroupKFold); "
            "requirements.txt (scikit-learn 1.9.1).")

    # ---------------------------------------------------------------- 5
    s = new(prs)
    header(s, 5, "Results: grouped nested cross-validation",
           "Primary protocol \u2014 fold means, no state appears on both sides of a split")
    table(s, [
        ["Configuration", "Districts", "Best model", "MAE", "RMSE", "R\u00b2", "Spearman"],
        ["A", "640", "SVR-RBF", "2.0293", "2.6144", "0.1699", "0.4626"],
        ["A", "628", "Extra Trees", "2.0888", "2.6540", "0.1837", "0.4966"],
        ["B", "628", "SVR-RBF", "1.7248", "2.2556", "0.3992", "0.6258"],
    ], 0.45, 1.45, 12.43, [0.17, 0.12, 0.19, 0.13, 0.13, 0.13, 0.13], row_h=0.42, head_h=0.44,
        font=13, head_font=12.5, align="lcccccc", highlight_rows=(3,))

    bullets(s, 0.45, 3.55, 6.0, 2.6, [
        "MAE and RMSE in percentage points; R\u00b2 and Spearman on out-of-fold predictions.",
        "Every family except the single decision tree beats the mean baseline (MAE 2.4495).",
        "Penalised linear and kernel models lead; a lone tree is worse than predicting the mean.",
    ], 12, marker="\u25aa  ")

    y = card(s, 6.75, 3.45, 6.13, 2.55, "SAME MODELS, RANDOM FOLDS (REFERENCE ONLY)",
             fill=RGBColor(0xFB, 0xF3, 0xF2), edge=RGBColor(0xF0, 0xD9, 0xD6), title_color=FAIL)
    textbox(s, 6.97, y, 5.7, 1.7,
            "Best random-fold MAE:  1.3257  (A, 640)\n"
            "                                       1.3390  (A, 628)\n"
            "                                       1.1401  (B, 628)",
            12, False, GREY)
    textbox(s, 6.97, y + 0.95, 5.7, 0.9,
            "Random folds understate error by 0.58\u20130.75 pp: districts of one state share context. "
            "Reported only as a reference, never as the primary estimate.", 11.5, False, MID, italic=True)

    band(s, 0.45, 6.20, 12.43, 0.52,
         "Geography, not model choice, is the dominant source of difficulty.", NAVY, WHITE, 12.5)
    footer(s, 5)
    note(s, "Source: evaluation/metrics/phase3_selected_models.csv and phase3_nested_cv_results.csv "
            "(grouped scheme). Random-fold values from the same file, scheme='random'.")

    # ---------------------------------------------------------------- 6
    s = new(prs)
    header(s, 6, "Configuration A vs B, and the formally ratified model roles")
    textbox(s, 0.45, 1.25, 6.0, 0.3, "IDENTICAL 628 DISTRICTS, IDENTICAL FOLDS", 11.5, True, NAVY)
    table(s, [
        ["Model", "A (628)", "B (628)"],
        ["SVR-RBF", "2.1714", "1.7248"],
        ["Ridge", "2.1845", "1.7490"],
        ["Elastic Net", "2.1963", "1.7476"],
    ], 0.45, 1.60, 5.55, [0.40, 0.30, 0.30], row_h=0.40, head_h=0.40, font=12.5, align="lcc")
    bullets(s, 0.45, 3.35, 5.75, 2.6, [
        "B improved 8 of 10 families; 7 improved on all five folds.",
        "Adding valid household features cuts grouped MAE by roughly 0.42\u20130.45 pp.",
        "Not causal: B also adds 34 correlated predictors and drops 12 districts.",
    ], 12)

    y = card(s, 6.55, 1.25, 6.33, 2.55, "RATIFIED ROLES (frozen after Phase 3)")
    bullets(s, 6.77, y, 5.9, 2.0, [
        "**Primary model:** Ridge, configuration B.",
        "**Secondary interpretable:** Elastic Net, configuration B.",
        "**Best benchmark:** SVR-RBF, configuration B.",
    ], 12)

    y = card(s, 6.55, 3.95, 6.33, 2.05, "WHY RIDGE AND NOT SVR?", fill=LIGHT, edge=LIGHT)
    bullets(s, 6.77, y, 5.9, 1.6, [
        "SVR\u2019s MAE advantage is 0.0242 pp \u2014 small, and not tested for statistical significance.",
        "Ridge gives an interpretable coefficient vector and stable selection (\u03b1 = 100 in all five outer folds).",
    ], 11.5)

    band(s, 0.45, 6.20, 12.43, 0.52,
         "B\u2019s gain is real but not free: 12 districts lose coverage, and none of it establishes causality.",
         AMBER, WHITE, 11.5)
    footer(s, 6)
    note(s, "Source: evaluation/metrics/phase3_ab_comparison.csv (a_mae, b_mae on shared 628 districts) and "
            "phase3_selected_models.csv. Ridge alpha=100 in all folds; coefficient stability from "
            "phase3_coefficient_stability.csv.")

    # ---------------------------------------------------------------- 7
    s = new(prs)
    header(s, 7, "Error analysis: the model is unbiased overall and blind in the tail")
    textbox(s, 0.45, 1.22, 5.9, 0.3, "LARGEST UNDER-PREDICTIONS (RIDGE B)", 11.5, True, NAVY)
    table(s, [
        ["District", "Actual", "Predicted"],
        ["Dhubri", "17.0", "7.40"],
        ["Barpeta", "17.7", "8.30"],
        ["Purba Medinipur", "17.2", "8.04"],
        ["Murshidabad", "16.0", "7.04"],
        ["Darrang", "17.2", "8.53"],
    ], 0.45, 1.58, 5.5, [0.52, 0.24, 0.24], row_h=0.37, head_h=0.38, font=12, align="lcc")
    textbox(s, 0.45, 4.05, 5.5, 0.7,
            "Concentration in Assam and West Bengal; errors reach 9.6 pp on a 17.7 pp target.",
            11.5, False, FAIL, italic=True)

    bullets(s, 6.35, 1.20, 6.55, 2.5, [
        "**Overall fit:** MAE 1.7490, R\u00b2 0.3992 \u2014 close to unbiased on average.",
        "**Tail:** 97% of p95 districts under-predicted; p95 bias +4.510 pp; p90 MAE 3.648 pp.",
        "**Distribution shift:** the worst districts look unlike the districts the model trains on.",
    ], 12)

    y = card(s, 6.35, 3.30, 6.53, 2.72, "INTERPRETATION IS POSSIBLE BUT FRAGILE")
    bullets(s, 6.57, y, 6.1, 2.2, [
        "Largest standardised coefficients: tv_share \u22120.473, hh_size_2 \u22120.449, electricity \u22120.394, radio \u22120.360.",
        "Permutation importance (SVR-B): tv_share 0.0836, mobile_only 0.0527, tapwater 0.0434.",
        "54/77 coefficients keep their sign across folds; median SD/|mean| = 0.55 \u2014 limited stability.",
    ], 11)

    band(s, 0.45, 6.22, 12.43, 0.5,
         "Correlated compositional predictors make single-feature attribution fragile \u2014 associations, not effects.",
         NAVY, WHITE, 11.5)
    footer(s, 7)
    note(s, "Sources: reports/ml_report/PHASE_3_MODELLING_REPORT.md (error cases), "
            "phase3_coefficient_stability.csv, phase3_permutation_importance.csv, phase3_ridge_permutation_importance.csv.")

    # ---------------------------------------------------------------- 8
    s = new(prs)
    header(s, 8, "Tail-aware experiments: every challenger failed the registered gates")
    table(s, [
        ["Arm", "Approach", "MAE", "R\u00b2", "p90 MAE", "p95 bias", "Sens.", "Prec."],
        ["E0", "Ridge baseline", "1.750", "0.468", "3.648", "+4.510", "0.371", "0.520"],
        ["E1", "Mild sample weighting", "1.915", "0.398", "3.032", "+3.693", "0.543", "0.388"],
        ["E2", "Stronger weighting", "1.975", "0.350", "2.831", "+3.435", "0.686", "0.387"],
        ["E3", "Top-decile weighting", "1.985", "0.347", "2.807", "+3.271", "0.600", "0.300"],
        ["E4a", "Quantile \u03c4 = 0.75", "1.914", "0.375", "2.624", "+3.109", "0.629", "0.324"],
        ["E4b", "Quantile \u03c4 = 0.90", "2.912", "\u22120.302", "2.331", "+2.994", "0.543", "0.170"],
    ], 0.45, 1.28, 12.43, [0.07, 0.27, 0.11, 0.11, 0.13, 0.12, 0.10, 0.09], row_h=0.34, head_h=0.38,
        font=11.5, head_font=11, align="llcccccc", highlight_rows=(1,))

    bullets(s, 0.45, 4.14, 6.0, 1.85, [
        "Tail bias falls in every arm (4.510 \u2192 2.994 pp), but overall MAE worsens 9.4\u201313.5% for E1\u2013E4a.",
        "E4b collapses: MAE 2.912, R\u00b2 \u22120.302 \u2014 not a usable model.",
    ], 11.5)

    bullets(s, 6.75, 4.14, 6.13, 1.85, [
        "E1 bootstrap: \u0394MAE +0.165 [+0.022, +0.327]; \u0394bias \u22120.817 [\u22121.077, \u22120.487] \u2014 the trade-off is real.",
        "Quantile coverage 0.670 at \u03c4 = 0.75 and 0.825 at \u03c4 = 0.90 \u2014 below nominal, so not calibrated.",
        "C1, C2 and C6 failed for every challenger; no model promoted.",
    ], 11.5)

    band(s, 0.45, 6.10, 12.43, 0.78,
         "No improvement was demonstrated using the tested tail-aware approaches. Limited predictor information is "
         "one possible explanation, but cannot be established from these experiments alone.", NAVY, WHITE, 11.5)
    footer(s, 8)
    note(s, "Sources: evaluation/metrics/tail_experiments.csv, tail_bootstrap.csv, tail_quantile_metrics.csv. "
            "Gates C1 (30% p90 MAE improvement + bias), C2 (<=5% MAE/RMSE loss), C6 (precision) all failed; "
            "C3 and C5 passed; C1/C2 not applicable to quantile estimands.")

    # ---------------------------------------------------------------- 9
    s = new(prs)
    header(s, 9, "Literature review: six themes and the gap they leave",
           "Numbering matches the paper\u2019s bibliography (20 sources, all cited)")
    table(s, [
        ["Theme", "Works", "What is established", "What is left open"],
        ["Measurement & data", "[1], [2], [3]", "Census housing-condition tables; global slum aggregates",
         "No district-level predictive model of dilapidated housing"],
        ["Regularised linear", "[4], [5], [6]", "Ridge, lasso and elastic-net estimation theory",
         "Rarely compared under geographic grouping on housing data"],
        ["Tree ensembles", "[7], [8], [9]", "Random forests, gradient boosting, extremely randomised trees",
         "No evidence for Indian district housing outcomes"],
        ["Kernel & instance", "[10], [11]", "SVR and k-nearest-neighbour foundations",
         "Behaviour under state-level distribution shift unknown"],
        ["Spatial validation", "[12], [13]", "Grouped/blocked CV is mandatory for clustered data",
         "Not applied to this outcome or geography"],
        ["Quantiles & intervals", "[14], [15], [16]", "Quantile regression; binomial interval estimation",
         "Tail-model calibration rarely audited empirically"],
        ["Tooling & metrics", "[17], [18], [19], [20]", "scikit-learn; forecast accuracy; bootstrap; ecological inference",
         "Methods and caveats used, not extended"],
    ], 0.45, 1.35, 12.43, [0.15, 0.09, 0.38, 0.38], row_h=0.50, head_h=0.38, font=10.5,
        head_font=11, first_bold=True, align="llll")

    band(s, 0.45, 6.08, 12.43, 0.72,
         "No prior study was located that predicts district-level dilapidated-house prevalence in India with "
         "geographically grouped validation. The paper makes no \u201cfirst of its kind\u201d claim.", TEAL, WHITE, 11.5)
    footer(s, 9)
    note(s, "Bibliography in reports/paper/references_verified.md; DOIs in REFERENCE_VERIFICATION_TABLE.md "
            "(11 resolved in one Crossref query, rest checked against publisher/official pages).")

    # ---------------------------------------------------------------- 10
    s = new(prs)
    header(s, 10, "What exists vs what this study did")
    table(s, [
        ["Prior work", "Focus", "What this study adds"],
        ["Census tabulations [1], [2]", "Published counts per district", "Turns them into a supervised target plus a validated predictor set"],
        ["Global housing reports [3]", "Cross-country aggregates", "Within-country resolution across 640 districts"],
        ["Estimator papers [4]\u2013[11]", "Algorithm properties on benchmark data", "10 families, 2 configurations, 76 candidates, identical grouped folds"],
        ["Spatial-CV papers [12], [13]", "Ecology and general statistics", "Transfers the protocol to Indian district housing; measures the 0.58\u20130.75 pp optimism"],
        ["Tail & quantile methods [14]\u2013[16]", "Quantile estimation and intervals", "Pre-registered remedies judged on cost, bias and precision \u2014 all failed"],
    ], 0.45, 1.30, 12.43, [0.26, 0.30, 0.44], row_h=0.52, head_h=0.38, font=10.5, head_font=11,
        first_bold=True, align="lll")

    y = card(s, 0.45, 4.55, 12.43, 2.25, "WHAT THE RESULTS SUPPORT", fill=LIGHT, edge=LIGHT)
    bullets(s, 0.70, y, 11.9, 1.7, [
        "Expanded household features cut grouped MAE from 2.1714 to 1.7248 on shared districts; Ridge B is the selected primary model.",
        "Grouped validation changes the answer: random folds understate error by 0.58\u20130.75 pp, so geography is the binding constraint.",
        "Tail remedies bought bias reduction with overall accuracy and failed every gate \u2014 no promotion, and no information ceiling is claimed.",
    ], 11.5)
    footer(s, 10)
    note(s, "Closing slide. Everything here is reported in reports/paper/IEEE_Paper_Dilapidated_Housing_India.pdf; "
            "numbers trace to evaluation/metrics/*.csv. No challenger was promoted; Ridge B remains primary.")

    prs.save(OUT)
    return len(prs.slides)


if __name__ == "__main__":
    count = build()
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {count} slides)")
