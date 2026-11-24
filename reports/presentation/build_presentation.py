"""Generate the final academic presentation.

Every number on every slide is copied from a file in ``evaluation/metrics/`` or
``evaluation/predictions/``; none is typed from memory. Charts are the project's
own Phase 3 figures, not fabricated images.

Run: py -3.14 reports/presentation/build_presentation.py
Output: reports/presentation/FINAL_PRESENTATION.pptx (29 slides, 16:9)
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "evaluation" / "plots"
OUT = ROOT / "reports" / "presentation" / "FINAL_PRESENTATION.pptx"

NAVY = RGBColor(0x0F, 0x27, 0x40)
TEAL = RGBColor(0x1F, 0x7A, 0x8C)
LIGHT = RGBColor(0xF4, 0xF7, 0xF9)
GREY = RGBColor(0x44, 0x4F, 0x5A)
ACCENT = RGBColor(0xC2, 0x5B, 0x3C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

TITLE_FONT = "Georgia"
BODY_FONT = "Calibri"


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def background(slide, colour: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = colour


def textbox(slide, left, top, width, height, text, size, bold=False, colour=NAVY,
            font=BODY_FONT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = colour
    run.font.name = font
    return box


def header(slide, title, number):
    background(slide, LIGHT)
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    textbox(slide, 0.45, 0.18, 11.6, 0.6, title, 26, bold=True, colour=WHITE, font=TITLE_FONT)
    textbox(slide, 12.45, 0.32, 0.7, 0.4, str(number), 12, colour=WHITE, align=PP_ALIGN.RIGHT)
    line = slide.shapes.add_shape(1, Inches(0.45), Inches(6.95), Inches(12.4), Inches(0.03))
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()
    textbox(slide, 0.45, 7.02, 12.0, 0.3,
            "District-level prevalence of dilapidated census houses, Census 2011 — not a homelessness model",
            9, colour=GREY)


def bullets(slide, items, left=0.55, top=1.25, width=12.2, height=5.4, size=17, spacing=10):
    # clamp so a box can never extend past the footer line
    height = min(height, 7.35 - top)
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    first = True
    for item in items:
        level = 0
        if item.startswith("  "):
            level, item = 1, item.strip()
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.level = level
        paragraph.space_after = Pt(spacing)
        run = paragraph.add_run()
        run.text = ("• " if level == 0 else "– ") + item
        run.font.size = Pt(size if level == 0 else size - 3)
        run.font.color.rgb = NAVY if level == 0 else GREY
        run.font.name = BODY_FONT
    return box


def table(slide, rows, left, top, width, height, font_size=12, header_row=True,
          col_widths=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    grid = shape.table
    if col_widths:
        for index, proportion in enumerate(col_widths):
            grid.columns[index].width = Inches(width * proportion)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = grid.cell(r, c)
            cell.text = str(value)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            fill = cell.fill
            fill.solid()
            fill.fore_color.rgb = NAVY if (header_row and r == 0) else (
                WHITE if r % 2 else RGBColor(0xE8, 0xEE, 0xF2))
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
                    run.font.bold = bool(header_row and r == 0)
                    run.font.color.rgb = WHITE if (header_row and r == 0) else NAVY
                    run.font.name = BODY_FONT
    return grid


def picture(slide, name, left, top, width, max_height=5.5):
    """Place a figure, shrinking it if its natural height would overflow."""
    path = PLOTS / name
    if not path.exists():
        return False
    from PIL import Image

    with Image.open(path) as image:
        aspect = image.height / image.width
    height = width * aspect
    if height > max_height:
        height, width = max_height, max_height / aspect
    if left + width > 12.9:
        width = 12.9 - left
        height = width * aspect
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width), height=Inches(height))
    return True


def content_slide(prs, title, number, items, size=17):
    slide = blank(prs)
    header(slide, title, number)
    bullets(slide, items, size=size)
    return slide


def build() -> Presentation:
    prs = new_deck()

    # 1 — title
    slide = blank(prs)
    background(slide, NAVY)
    band = slide.shapes.add_shape(1, Inches(0), Inches(4.55), Inches(13.333), Inches(0.06))
    band.fill.solid()
    band.fill.fore_color.rgb = TEAL
    band.line.fill.background()
    textbox(slide, 0.8, 1.05, 11.7, 1.4,
            "Predicting District-Level Prevalence of\nDilapidated Census Houses in India",
            40, bold=True, colour=WHITE, font=TITLE_FONT)
    textbox(slide, 0.8, 2.95, 11.7, 0.6, "Using Machine Learning on Census 2011 data",
            22, colour=RGBColor(0xBF, 0xD4, 0xDE), font=TITLE_FONT)
    textbox(slide, 0.8, 4.8, 11.7, 0.5, "Final project presentation", 16,
            colour=RGBColor(0xBF, 0xD4, 0xDE))
    textbox(slide, 0.8, 5.3, 11.7, 1.2,
            "640 districts · 35 states and UTs · 10 model families · grouped nested cross-validation",
            14, colour=RGBColor(0x8F, 0xAA, 0xB8))

    # 2 — introduction
    content_slide(prs, "Introduction", 2, [
        "Housing condition is one of the most directly observable dimensions of deprivation.",
        "Census of India records, for every district, how many census houses are Dilapidated, Livable or Good.",
        "That gives a comparable prevalence measure for all 640 districts of the 2011 geography.",
        "Question: how much of the between-district variation can be predicted from other district indicators?",
        "  And how well does such a model generalise to states it has never seen?",
        "Deliberately predictive, not explanatory: this is not a study of why housing decays.",
    ])

    # 3 — problem statement
    content_slide(prs, "Problem statement", 3, [
        "For district i with feature vector x ∈ R^77, estimate the share of census houses classified Dilapidated.",
        "Outcome y ∈ [0, 100] percentage points: mean 5.06, median 4.40, sd 3.12, skew 1.154, range 0.2–17.7.",
        "Most districts sit near 3–5%; a minority exceed 10%; the errors that matter socially are in that minority.",
        "n = 628 districts for the ratified configuration; 12 cannot be matched to a household denominator.",
        "Constraint: the unit is an aggregate. No individual or household inference is possible.",
    ])

    # 4 — objectives
    content_slide(prs, "Research objectives", 4, [
        "1. Construct a defensible district-level target; document why the original denominator was invalid.",
        "2. Build leakage-free feature configurations from district socioeconomic and housing data.",
        "3. Compare ten model families under validation that respects state-level clustering.",
        "4. Quantify the cost of honest out-of-state validation versus random cross-validation.",
        "5. Test whether tail-aware remedies reduce under-prediction of the most dilapidated districts.",
        "6. Deliver a reproducible prediction artefact and an inspectable inference path.",
    ])

    # 5 — literature / gap
    content_slide(prs, "Existing work and the research gap", 5, [
        "Housing condition as a deprivation measure: Census Houselisting and Housing tables (2011).",
        "Regularised linear models: Ridge (Hoerl & Kennard, 1970); Elastic Net (Zou & Hastie, 2005).",
        "Non-linear benchmarks: random forests (Breiman, 2001); gradient boosting (Friedman, 2001); SVR (Cortes & Vapnik, 1995).",
        "Tail modelling: quantile regression and the pinball loss (Koenker & Bassett, 1978).",
        "Validation: grouped cross-validation for clustered data (Roberts et al., 2017).",
        "Gap: a pre-registered, grouped-validation predictive benchmark for this outcome, with an explicit account of where it fails.",
    ], size=16)

    # 6 — approach
    slide = blank(prs)
    header(slide, "Proposed approach", 6)
    steps = ["Raw census\nextracts\n(D1, D2, D5)", "Geography\nreconciliation\n640/640 keys",
             "Target\nconstruction\n+ cross-check", "Features\nA: 43 / B: 77",
             "Grouped\nnested CV\n10 families", "Ridge B\nratified", "Phase 6\npredictions\n+ ranking"]
    left = 0.45
    width = 1.72
    for index, step in enumerate(steps):
        shape = slide.shapes.add_shape(1, Inches(left + index * 1.79), Inches(1.75),
                                       Inches(width), Inches(1.5))
        shape.fill.solid()
        shape.fill.fore_color.rgb = TEAL if index < 6 else ACCENT
        shape.line.fill.background()
        frame = shape.text_frame
        frame.word_wrap = True
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = step
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = BODY_FONT
    bullets(slide, [
        "Every stage is a script with a documented command, a fixed seed and a hashed output.",
        "Nothing is hand-edited: all tables and figures regenerate from the raw files.",
        "Phase 4 adds five pre-registered tail remedies between the ratified model and delivery.",
    ], top=3.6, size=15)

    # 7 — datasets
    slide = blank(prs)
    header(slide, "Dataset overview", 7)
    table(slide, [
        ["ID", "Source", "Shape", "Role"],
        ["D1", "District census 2011", "640 × 118", "Primary predictors"],
        ["D2", "Housing condition (HLPCA)", "1,908 × 156", "Target source (Rural/Urban/Total)"],
        ["D3", "HLPCA column dictionary", "156 labels", "Documentation only"],
        ["D4", "Slum population", "32 × 12", "Excluded — state level, 4 states absent"],
        ["D5", "Primary Census Abstract 2011", "1,342 × 89", "Household denominator + 34 predictors"],
    ], 0.5, 1.3, 12.3, 3.4, font_size=14, col_widths=[0.09, 0.36, 0.2, 0.35])
    bullets(slide, [
        "D1 ↔ D2 reconciles 640/640 districts once names are canonicalised as STATE||DISTRICT.",
        "Six district names occur in two states → 12 districts unscoreable in Configuration B.",
        "D4 cannot enter a district model; D6–D10 are registered but unused (wrong vintage or scope).",
    ], top=5.05, size=15)

    # 8 — target
    slide = blank(prs)
    header(slide, "Target variable construction", 8)
    bullets(slide, [
        "Target = Census-computed share of census houses classified Dilapidated (D2, Total row).",
        "Rejected denominator: D1 'Households' sums to 330,835,767 = the census-house stock, not the 246,740,228 households.",
        "  Using it would have inflated every district's rate by roughly 34%.",
        "  Caught in Phase 1, recorded as decision D-001, formula abandoned.",
        "Independent verification: reconstructed target differs by 0.00e+00.",
        "Count-based cross-check (D1 dilapidated ÷ D5 households): r = 0.999368, median difference 0.053 pp.",
        "Unit is the census house — a structural-condition measure, not homelessness.",
    ], size=16)

    # 9 — preprocessing
    slide = blank(prs)
    header(slide, "Data preprocessing", 9)
    table(slide, [
        ["Rule", "Implementation"],
        ["Missing ≠ zero", "No zero-fill; 12 unmatched districts excluded, never imputed"],
        ["No clipping / winsorising", "5 rates exceed 1.0 by ≤6.2% (universe mismatch) and are used as-is"],
        ["Target-derived columns", "Deny-list enforced in code and asserted in tests"],
        ["Preprocessing in-fold", "Imputer and scaler fitted inside every training fold"],
        ["Types", "Numeric parsing fails loudly on unparseable values"],
    ], 0.5, 1.3, 12.3, 3.2, font_size=14, col_widths=[0.28, 0.72])
    bullets(slide, [
        "Median imputation is a no-op in Configuration B (0 missing, 0 infinite) but kept for schema safety.",
        "Raw data is read-only; every derived artefact is regenerated by code.",
    ], top=5.0, size=15)

    # 10 — features
    slide = blank(prs)
    header(slide, "Feature engineering", 10)
    table(slide, [
        ["Configuration", "Districts", "Predictors", "Content"],
        ["A (base)", "640", "43", "D1 shares: assets, housing materials, water, sanitation, household size, education, work, religion, caste"],
        ["B (augmented)", "628", "77", "A's 43 + 34 D5 housing-stock and asset predictors requiring a household denominator"],
    ], 0.5, 1.3, 12.3, 1.9, font_size=13, col_widths=[0.18, 0.1, 0.1, 0.62])
    bullets(slide, [
        "Leakage control: Literate_Education subtotal excluded (levels ÷ it = 1.000000).",
        "Four collinearity drops documented with reasons; target-correlation screen flagged 0 features (max |r| = 0.528).",
        "Disclosed optimism: the screen is computed once on the full sample.",
        "Result: adding the 34 D5 predictors cuts MAE by 10–21% for every family that can use them.",
    ], top=3.6, size=15)

    # 11 — EDA
    slide = blank(prs)
    header(slide, "Exploratory data analysis", 11)
    table(slide, [
        ["Property", "Value"],
        ["Districts (A / B)", "640 / 628"],
        ["Range", "0.2 – 17.7 pp"],
        ["Mean / median", "5.035 / 4.40"],
        ["Standard deviation / skew", "3.12 / 1.154"],
        ["Districts ≥ p90 (9.20)", "64"],
        ["Districts ≥ p95 (11.20)", "35"],
    ], 0.5, 1.3, 5.6, 3.6, font_size=13, col_widths=[0.6, 0.4])
    bullets(slide, [
        "Right-skewed: most districts 3–5%, a minority above 10%.",
        "The top 35 districts are concentrated: 24 lie in Assam and West Bengal.",
        "That concentration is why random splitting leaks and state grouping is mandatory.",
        "Predictors are strongly inter-correlated — one prosperity dimension dominates.",
    ], left=6.5, top=1.35, width=6.3, size=15)

    # 12 — architecture
    slide = blank(prs)
    header(slide, "Project architecture", 12)
    table(slide, [
        ["Layer", "Contents"],
        ["src/data", "Loaders, canonical naming, audits, denominator forensics"],
        ["src/features, src/preprocessing", "Configurations A and B, in-fold preprocessing"],
        ["src/models", "Registry of ten families; five tail-aware arms"],
        ["src/training", "Nested CV, configuration builders, Phase 4 runner"],
        ["src/evaluation", "Metrics, comparison, error analysis, bootstrap, intervals"],
        ["src/prediction", "Final model fitting, persistence, inference, deliverables"],
        ["evaluation/, models/, reports/", "Metrics, predictions, plots, artefacts, documentation"],
    ], 0.5, 1.3, 12.3, 3.7, font_size=13, col_widths=[0.28, 0.72])
    bullets(slide, [
        "99 tests across six files; 0 failures, 0 skips.",
        "No network access, no machine-specific paths, no notebooks required to reproduce.",
    ], top=5.4, size=15)

    # 13 — models
    slide = blank(prs)
    header(slide, "Machine-learning models", 13)
    table(slide, [
        ["Family", "Candidates", "Family", "Candidates"],
        ["Dummy (mean)", "1", "Random forest", "8"],
        ["Ridge", "5", "Extra trees", "8"],
        ["Elastic Net", "12", "Gradient boosting", "8"],
        ["k-NN", "8", "Hist. gradient boosting", "8"],
        ["Decision tree", "12", "SVR-RBF", "6"],
    ], 0.5, 1.3, 12.3, 3.0, font_size=14, col_widths=[0.32, 0.18, 0.32, 0.18])
    bullets(slide, [
        "76 candidate configurations in total; hyperparameters chosen on inner folds only.",
        "All fits: seed 42, n_jobs = 1, identical preprocessing pipeline.",
        "Ratified roles: Ridge B primary interpretable · Elastic Net B secondary · SVR-RBF accuracy benchmark.",
    ], top=4.6, size=15)

    # 14 — validation
    slide = blank(prs)
    header(slide, "Validation methodology", 14)
    bullets(slide, [
        "5 outer folds: GroupKFold on state — 28 states train, 7 test. No state ever appears on both sides.",
        "4 inner folds on outer-training states only; the hyperparameter is chosen there, never on test data.",
        "304 inner fits per outer fold across 76 configurations → 9,120 fits over six sweeps.",
        "Outer test rows cannot influence α, thresholds, weights or coefficients — verified by corrupting them: bit-identical output.",
        "Random 5-fold CV is reported separately as a reference only.",
        "  SVR-RBF: MAE 1.1401 random vs 1.7248 grouped — a 51% overstatement if random CV were quoted.",
    ], size=16)

    # 15 — A vs B
    slide = blank(prs)
    header(slide, "Configuration A versus Configuration B", 15)
    table(slide, [
        ["Model", "A (640)", "B (628)", "Change"],
        ["Elastic Net", "2.196", "1.748", "−20.4%"],
        ["SVR-RBF", "2.171", "1.725", "−20.6%"],
        ["Ridge", "2.185", "1.749", "−19.9%"],
        ["Extra trees", "2.089", "1.810", "−13.3%"],
        ["Decision tree", "2.253", "2.506", "+11.2%"],
        ["Dummy", "2.450", "2.450", "0"],
    ], 0.5, 1.3, 6.0, 3.3, font_size=13, col_widths=[0.4, 0.2, 0.2, 0.2])
    picture(slide, "phase3_ab_comparison.png", 6.8, 1.25, 6.1)
    bullets(slide, [
        "Better in 5/5 folds for every family that benefits; B is ratified.",
        "Caveat: B covers 12 fewer districts (non-random exclusion).",
    ], top=5.0, width=6.1, size=14)

    # 16 — comparison
    slide = blank(prs)
    header(slide, "Model comparison — grouped nested CV (fold means)", 16)
    table(slide, [
        ["Model", "MAE", "RMSE", "R²", "Spearman"],
        ["SVR-RBF (benchmark)", "1.7248", "2.2556", "0.3992", "0.6258"],
        ["Ridge α=100 (primary)", "1.7490", "2.2703", "0.3992", "0.6311"],
        ["Elastic Net", "1.7476", "2.2946", "0.3923", "0.6444"],
        ["Extra trees", "1.8101", "2.3309", "0.3727", "0.6289"],
        ["Decision tree", "2.5055", "3.2127", "−0.1840", "0.3752"],
        ["Dummy (mean)", "2.4495", "3.1438", "−0.1157", "—"],
    ], 0.5, 1.3, 7.2, 3.1, font_size=13, col_widths=[0.42, 0.155, 0.155, 0.155, 0.155])
    picture(slide, "phase3_model_comparison_grouped.png", 8.0, 1.3, 4.9)
    bullets(slide, [
        "SVR leads by 0.024 pp — not distinguishable, and not worth losing interpretability.",
        "Elastic Net has the best rank agreement (ρ 0.644) at the same MAE; kept as secondary.",
        "Negative R² is retained, not clipped: those models are worse than predicting the mean.",
    ], top=4.7, width=7.2, size=14)

    # 17 — Ridge B performance
    slide = blank(prs)
    header(slide, "Ridge B — the ratified primary model", 17)
    table(slide, [
        ["Metric", "Fold mean", "Pooled OOF"],
        ["MAE (pp)", "1.7490", "1.7496"],
        ["RMSE (pp)", "2.2703", "2.2842"],
        ["R²", "0.3992", "0.4680"],
        ["Spearman ρ", "0.6311", "0.6672"],
        ["Dummy MAE", "2.4495", "—"],
    ], 0.5, 1.3, 5.4, 2.8, font_size=13, col_widths=[0.4, 0.3, 0.3])
    picture(slide, "phase3_predicted_vs_actual_b628.png", 6.2, 1.25, 6.6)
    bullets(slide, [
        "Fold MAE 1.749 ± 0.178: the result is stable across folds, not carried by one split.",
        "R² 0.40 means about 60% of between-district variance is unexplained — the model ranks, it does not explain.",
    ], top=4.5, width=5.4, size=13)

    # 18 — importance
    slide = blank(prs)
    header(slide, "Feature importance and coefficient stability", 18)
    table(slide, [
        ["Ridge coefficient (B)", "Mean", "Folds"],
        ["tv_share", "−0.473", "0/5 positive"],
        ["hh_size_2_share", "−0.449", "0/5"],
        ["electricity_share", "−0.394", "0/5"],
        ["hh_size_3_share", "+0.313", "5/5"],
        ["other_cooking_fuel_share", "+0.301", "5/5"],
    ], 0.5, 1.3, 6.4, 2.9, font_size=13, col_widths=[0.5, 0.25, 0.25])
    picture(slide, "phase3_importance_b628.png", 7.2, 1.25, 5.7)
    bullets(slide, [
        "54 of 77 coefficients keep their sign in all five folds; 23 flip and are not interpreted.",
        "Two attribution methods agree: a single prosperity dimension dominates (top 10 features = 57% of importance).",
        "Association only. Correlated shares measured on the same base are not independent mechanisms.",
    ], top=4.6, size=13)

    # 19 — extreme districts
    slide = blank(prs)
    header(slide, "The extreme-district problem", 19)
    table(slide, [
        ["Group", "n", "MAE", "Mean actual", "Mean predicted", "Bias"],
        ["All districts", "628", "1.750", "5.058", "4.966", "+0.092"],
        ["≥ p90 (9.20)", "64", "3.648", "11.769", "8.175", "+3.593"],
        ["≥ p95 (11.20)", "35", "4.527", "13.231", "8.722", "+4.510"],
    ], 0.5, 1.3, 12.3, 2.2, font_size=14, col_widths=[0.26, 0.1, 0.15, 0.17, 0.17, 0.15])
    bullets(slide, [
        "The model is almost unbiased overall (+0.09 pp) while under-predicting the worst-off by 4.5 pp.",
        "It compresses the distribution toward the mean — the failure is invisible in the headline MAE.",
        "For identifying the most dilapidated districts, this is the central defect. It motivates Phase 4.",
    ], top=4.0, size=16)

    # 20 — Phase 4 arms
    slide = blank(prs)
    header(slide, "Phase 4 — five pre-registered tail remedies", 20)
    table(slide, [
        ["Arm", "Definition", "Grid"],
        ["E0", "Ratified Phase 3 Ridge OOF, reused unchanged", "—"],
        ["E1", "Weighted Ridge, threshold q75(train), λ = 2", "α ∈ {30, 100, 300}"],
        ["E2", "Weighted Ridge, threshold q75, λ = 4", "α ∈ {30, 100, 300}"],
        ["E3", "Weighted Ridge, threshold q90, λ = 4", "α ∈ {30, 100, 300}"],
        ["E4a", "Linear quantile regression, τ = 0.75", "L1 α ∈ {0.001, 0.01, 0.1}"],
        ["E4b", "Linear quantile regression, τ = 0.90", "L1 α ∈ {0.001, 0.01, 0.1}"],
    ], 0.5, 1.3, 12.3, 2.8, font_size=14, col_widths=[0.1, 0.62, 0.28])
    bullets(slide, [
        "325 fits; weights and thresholds recomputed inside every training split.",
        "Six gates fixed before execution: tail reduction, overall cost, fold consistency, state concentration, over-correction, precision.",
    ], top=4.5, size=15)

    # 21 — Phase 4 results
    slide = blank(prs)
    header(slide, "Phase 4 findings — no challenger promoted", 21)
    table(slide, [
        ["Arm", "MAE", "p95 bias", "Reduction", "C1", "C2 (cost)", "Verdict"],
        ["E0", "1.750", "+4.510", "—", "—", "—", "reference"],
        ["E1", "1.915", "+3.693", "−18.1%", "FAIL", "FAIL (+9.5%)", "not accepted"],
        ["E2", "1.975", "+3.435", "−23.8%", "FAIL", "FAIL (+12.9%)", "not accepted"],
        ["E3", "1.985", "+3.271", "−27.5%", "FAIL", "FAIL (+13.5%)", "not accepted"],
        ["E4a", "1.914", "+3.109", "−31.1%", "N/A", "FAIL (+9.4%)", "not accepted"],
        ["E4b", "2.912", "+2.994", "−33.6%", "N/A", "FAIL (+66.4%)", "not accepted"],
    ], 0.5, 1.25, 12.3, 2.9, font_size=13, col_widths=[0.08, 0.12, 0.13, 0.14, 0.08, 0.2, 0.25])
    bullets(slide, [
        "The tail moved consistently — but the C1 bar was a 30% reduction and the cost allowance was 5%.",
        "E4b over-corrected: West Bengal got worse; 63% of its gain sat in Assam (C4 FAIL).",
        "Quantile arms under-cover: E4b delivers 82.5% at nominal 90% (Wilson 79.3–85.3).",
        "Trade quantified: 0.12–0.21 pp of overall MAE per 1 pp of tail bias recovered.",
    ], top=4.35, size=14)

    # 22 — final system
    slide = blank(prs)
    header(slide, "Final prediction system (Phase 6)", 22)
    table(slide, [
        ["Artefact", "Content"],
        ["models/final/ridge_b_pipeline.joblib", "Fitted pipeline: imputer → scaler → Ridge(α = 100)"],
        ["ridge_b_metadata.json", "Versions, data hashes, target definition, reload warning"],
        ["feature_manifest.csv", "The 77 predictors in frozen order"],
        ["final_predictions_ridge_b_628.csv", "628 districts: key, district, state, predicted %, range flag"],
        ["district_ranking_ridge_b_628.csv", "Rank 1–628 + relative quintile bands"],
        ["inference.py", "Schema-validated scoring; 19 tests"],
    ], 0.5, 1.25, 12.3, 2.9, font_size=13, col_widths=[0.38, 0.62])
    bullets(slide, [
        "Fitted on all 628 districts → these are in-sample values, labelled refit_in_sample, NOT a performance estimate.",
        "Predictions span −0.636 to 13.535 pp. Two districts are slightly negative; flagged, not clipped.",
        "Bands are relative quintiles (126/125/126/125/126), not validated risk classes.",
    ], top=4.4, size=14)

    # 23 — ranking examples
    slide = blank(prs)
    header(slide, "District ranking — top of the list", 23)
    table(slide, [
        ["Rank", "District", "State", "Predicted %", "Band"],
        ["1", "Nadia", "West Bengal", "13.535", "B5"],
        ["2", "Koch Bihar", "West Bengal", "13.423", "B5"],
        ["3", "Murshidabad", "West Bengal", "13.109", "B5"],
        ["4", "Maldah", "West Bengal", "13.027", "B5"],
        ["5", "Barpeta", "Assam", "12.499", "B5"],
        ["6", "Dhubri", "Assam", "11.989", "B5"],
    ], 0.5, 1.3, 8.4, 2.8, font_size=14, col_widths=[0.12, 0.28, 0.26, 0.2, 0.14])
    bullets(slide, [
        "Consistent with the evaluation: Assam and West Bengal dominate the extreme group.",
        "Rank is a position among 628 eligible districts, not a national percentile.",
        "The top of the ranking is directionally useful but not calibrated — these are not magnitude forecasts.",
        "12 of 640 districts are absent by design; 2 fitted values are negative and flagged.",
    ], left=9.2, top=1.35, width=3.7, size=13)

    # 24 — limitations
    content_slide(prs, "Limitations", 24, [
        "Historical: 2011 only — no temporal validation, no claim about the present.",
        "Ecological: district averages say nothing about any household.",
        "Modest fit: R² ≈ 0.40 leaves ~60% of between-district variance unexplained.",
        "Out-of-state generalisation costs 0.58 pp MAE versus random CV — the higher number is the honest one.",
        "12 districts unscoreable (duplicated names); the tail is 35 districts, 24 in two states.",
        "Tail under-prediction (~4.5 pp) remains; five remedies failed their gates.",
        "23 of 77 coefficients flip sign; associations are not causal; no per-district uncertainty.",
    ], size=15)

    # 25 — contribution
    slide = blank(prs)
    header(slide, "Existing work versus this contribution", 25)
    table(slide, [
        ["Aspect", "Typical district study", "This project"],
        ["Validation", "Random k-fold", "State-grouped nested CV (5×4)"],
        ["Reported metric", "Single R² / accuracy", "MAE, RMSE, R², Spearman; fold-mean and pooled, labelled"],
        ["Tail behaviour", "Not reported", "Quantified: +4.51 pp under-prediction at p95"],
        ["Remedies", "None / ad hoc", "Five pre-registered arms, six gates, no promotion"],
        ["Deliverable", "Discussion", "Persisted model, 628 predictions, ranking, inference API"],
        ["Uncertainty", "Usually absent", "Paired bootstrap (both cluster units) + binomial intervals"],
        ["Audit", "None", "Independent audit, 156 checks, defect register D-01…D-21"],
    ], 0.5, 1.3, 12.3, 3.8, font_size=12, col_widths=[0.2, 0.28, 0.52])
    bullets(slide, [
        "The contribution is rigour and disclosure, not a new algorithm.",
    ], top=5.4, size=14)

    # 26 — conclusion
    content_slide(prs, "Conclusion", 26, [
        "A transparent linear model predicts district-level dilapidated-housing prevalence at MAE 1.75 pp, R² 0.40.",
        "That is 0.70 pp better than a mean baseline — and 0.58 pp worse than random CV would have claimed.",
        "Ten families compared; the accuracy leader's 0.024 pp edge did not justify losing interpretability.",
        "The real finding is the tail: the worst-off districts are under-predicted by 4.5 pp.",
        "Five registered remedies reduced that by 18–34% but all failed their acceptance gates.",
        "Every number traces to a file; every artefact regenerates from documented commands.",
    ], size=16)

    # 27 — future work
    content_slide(prs, "Future work", 27, [
        "New predictors rather than new models: housing age, construction material, climate exposure, migration.",
        "Longitudinal design: link 2001 and 2011 district data to model change.",
        "Hierarchical modelling: state random effects instead of treating clustering as a nuisance.",
        "Per-district prediction intervals (conformal or quantile-based) for cost-asymmetric decisions.",
        "External validation on an independent district-year with the same target definition.",
    ], size=16)

    # 28 — references
    slide = blank(prs)
    header(slide, "References", 28)
    bullets(slide, [
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
        "Census of India (2011). Houselisting and Housing Census; Primary Census Abstract.",
        "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
        "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
        "Friedman, J. H. (2001). Greedy function approximation. Annals of Statistics, 29(5), 1189–1232.",
        "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression. Technometrics, 12(1), 55–67.",
        "Koenker, R., & Bassett, G. (1978). Regression quantiles. Econometrica, 46(1), 33–50.",
        "Pedregosa, F., et al. (2011). Scikit-learn. JMLR, 12, 2825–2830.",
        "Roberts, D. R., et al. (2017). Cross-validation strategies. Ecography, 40(8), 913–929.",
        "Wilson, E. B. (1927). Probable inference. JASA, 22(158), 209–212.",
        "Zou, H., & Hastie, T. (2005). Elastic net. JRSS-B, 67(2), 301–320.",
    ], size=13, spacing=4)

    # 29 — thank you
    slide = blank(prs)
    background(slide, NAVY)
    textbox(slide, 0.9, 2.6, 11.5, 1.0, "Thank you", 48, bold=True, colour=WHITE, font=TITLE_FONT)
    textbox(slide, 0.9, 3.75, 11.5, 0.6, "Questions and discussion", 22,
            colour=RGBColor(0xBF, 0xD4, 0xDE), font=TITLE_FONT)
    textbox(slide, 0.9, 5.3, 11.5, 1.0,
            "All results reproducible with: py -3.14  ·  seed 42  ·  99 tests passing",
            14, colour=RGBColor(0x8F, 0xAA, 0xB8))

    return prs


def main() -> int:
    prs = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)

    slides = len(prs.slides)
    shapes = sum(len(slide.shapes) for slide in prs.slides)
    pictures = sum(1 for slide in prs.slides for shape in slide.shapes if shape.shape_type == 13)
    print(f"saved {OUT.relative_to(ROOT)}")
    print(f"  slides   {slides}")
    print(f"  shapes   {shapes}")
    print(f"  pictures {pictures} (real Phase 3 figures)")
    print(f"  size     {OUT.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
