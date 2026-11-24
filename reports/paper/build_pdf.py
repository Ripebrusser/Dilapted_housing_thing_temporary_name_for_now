"""Render paper_content.py to a two-column IEEE-style PDF (reportlab).

No LaTeX distribution is available on this machine, so the PDF is typeset
directly. Layout mirrors the IEEE conference template: Letter, 0.7 in side
margins, full-width title block on page 1, two 3.4 in columns thereafter,
Times 9 pt body, 7 pt tables.
"""

from pathlib import Path
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).parent))
import paper_content as C  # noqa: E402

OUT = Path(__file__).with_name("IEEE_Paper_Dilapidated_Housing_India.pdf")
FIG = Path(__file__).with_name("fig1_pipeline.png")

PW, PH = letter
LM = RM = 0.7 * inch
TM = BM = 0.75 * inch
TOP_H = 2.45 * inch
GAP = 0.28 * inch
COL_W = (PW - LM - RM - GAP) / 2
BODY_H = PH - TM - TOP_H - 0.18 * inch - BM

NAVY = colors.HexColor("#12374f")
RULE = colors.HexColor("#9fb3c0")
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

def roman_numeral(n):
    return ROMAN[n - 1]



def esc(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


S = {
    "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=16.5, leading=19.5,
                            alignment=TA_CENTER, spaceAfter=8),
    "author": ParagraphStyle("author", fontName="Times-Roman", fontSize=10, leading=12,
                             alignment=TA_CENTER),
    "affil": ParagraphStyle("affil", fontName="Times-Italic", fontSize=8.5, leading=10,
                            alignment=TA_CENTER, spaceAfter=6),
    "abstract": ParagraphStyle("abstract", fontName="Times-Roman", fontSize=8.4, leading=10.1,
                               alignment=TA_JUSTIFY, spaceAfter=4),
    "terms": ParagraphStyle("terms", fontName="Times-Italic", fontSize=8.4, leading=10.1,
                            alignment=TA_JUSTIFY, spaceAfter=2),
    "h": ParagraphStyle("h", fontName="Times-Bold", fontSize=9.2, leading=11,
                        alignment=TA_CENTER, spaceBefore=7, spaceAfter=3, textColor=NAVY),
    "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=8.7, leading=10.2,
                           alignment=TA_JUSTIFY, firstLineIndent=8, spaceAfter=2.6),
    "caption": ParagraphStyle("caption", fontName="Times-Roman", fontSize=7.2, leading=8.6,
                              alignment=TA_LEFT, spaceBefore=4, spaceAfter=2),
    "tablecap": ParagraphStyle("tablecap", fontName="Times-Roman", fontSize=7.2, leading=8.6,
                               alignment=TA_LEFT, spaceBefore=5, spaceAfter=2.5),
    "ref": ParagraphStyle("ref", fontName="Times-Roman", fontSize=7.1, leading=8.4,
                          alignment=TA_JUSTIFY, leftIndent=10, firstLineIndent=-10, spaceAfter=1.6),
}


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(LM, BM - 12, PW - RM, BM - 12)
    canvas.setFont("Times-Roman", 7.5)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawCentredString(PW / 2, BM - 22, str(canvas.getPageNumber()))
    canvas.restoreState()


def make_table(spec):
    rows = [[Paragraph(f"<b>{esc(str(c))}</b>", ParagraphStyle("th", fontName="Times-Bold",
                                                               fontSize=6.7, leading=7.8))
             for c in spec["header"]]]
    for row in spec["rows"]:
        rows.append([Paragraph(esc(str(c)), ParagraphStyle("td", fontName="Times-Roman",
                                                           fontSize=6.7, leading=7.8)) for c in row])
    widths = [COL_W * w for w in spec["widths"]]
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    return table


def build_story():
    story = [NextPageTemplate("later")]
    story.append(Paragraph(esc(C.TITLE), S["title"]))
    story.append(Paragraph(esc(C.AUTHOR), S["author"]))
    for line in C.AFFILIATION:
        story.append(Paragraph(esc(line), S["affil"]))
    story.append(Paragraph("<b>Abstract</b>&mdash;" + esc(C.ABSTRACT), S["abstract"]))
    story.append(Paragraph("<b><i>Index Terms</i></b>&mdash;" + esc(C.INDEX_TERMS), S["terms"]))
    story.append(FrameBreak())

    numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    figure_added = False
    table_no = 1
    for index, sec in enumerate(C.SECTIONS):
        story.append(Paragraph(f"{numerals[index]}.&nbsp;&nbsp;{esc(sec['heading'].upper())}", S["h"]))
        for kind, payload in sec["blocks"]:
            if kind == "p":
                story.append(Paragraph(esc(payload), S["body"]))
            elif kind == "table":
                story.append(Paragraph(f"TABLE {roman_numeral(table_no)}. {esc(payload['caption'])}",
                                       S["tablecap"]))
                table_no += 1
                story.append(make_table(payload))
                story.append(Spacer(1, 3))
            elif kind == "fig":
                pass
        if sec["heading"].startswith("Feature Engineering") and not figure_added and FIG.exists():
            img = Image(str(FIG))
            ratio = img.imageHeight / img.imageWidth
            img.drawWidth = COL_W
            img.drawHeight = COL_W * ratio
            story.append(Spacer(1, 4))
            story.append(img)
            story.append(Paragraph(
                "Fig. 1.&nbsp;&nbsp;End-to-end workflow. Each stage is a script with a fixed seed and a "
                "hashed output, so every reported number regenerates from the raw census files.",
                S["caption"]))
            figure_added = True

    story.append(Paragraph("REFERENCES", S["h"]))
    for index, ref in enumerate(C.REFERENCES, start=1):
        story.append(Paragraph(f"[{index}]&nbsp;{esc(ref)}", S["ref"]))
    return story


def main() -> int:
    doc = BaseDocTemplate(str(OUT), pagesize=letter,
                          leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
                          title=C.TITLE, author=C.AUTHOR,
                          subject="IEEE conference paper", creator="reportlab")

    top = Frame(LM, PH - TM - TOP_H, PW - LM - RM, TOP_H, id="top",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    left = Frame(LM, BM, COL_W, BODY_H, id="left",
                 leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    right = Frame(LM + COL_W + GAP, BM, COL_W, BODY_H, id="right",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    full_h = PH - TM - BM
    later_left = Frame(LM, BM, COL_W, full_h, id="l2",
                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    later_right = Frame(LM + COL_W + GAP, BM, COL_W, full_h, id="r2",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    doc.addPageTemplates([
        PageTemplate(id="first", frames=[top, left, right], onPage=decorate),
        PageTemplate(id="later", frames=[later_left, later_right], onPage=decorate),
    ])

    doc.build(build_story())
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"  PAGES: {doc.page}  (limit 6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
