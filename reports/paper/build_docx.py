"""Render paper_content.py to an IEEE-styled editable Word document.

Layout: US Letter, 0.75 in margins, single-column title/author/abstract block, then
a continuous section break into two columns of 3.4 in, Times New Roman 10 pt body.
"""

from pathlib import Path
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).parent))
import paper_content as C  # noqa: E402

OUT = Path(__file__).with_name("IEEE_Paper_Dilapidated_Housing_India.docx")
FONT = "Times New Roman"


def set_columns(section, number: int, space_twips: int = 340) -> None:
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(number))
    cols.set(qn("w:space"), str(space_twips))


def style_doc(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.0
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), FONT)


def para(doc, text, size=10, bold=False, italic=False, align="justify", space_before=0,
         space_after=0, indent=0.0, caps=False, color=None):
    p = doc.add_paragraph()
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.first_line_indent = Inches(indent)
    run = p.add_run(text)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if caps:
        run.font.small_caps = True
    if color:
        run.font.color.rgb = color
    return p


def rich_para(doc, text, size=10, indent=0.2):
    """Paragraph where **text** becomes bold (used for lead-ins)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(indent)
    p.paragraph_format.space_after = Pt(0)
    for index, chunk in enumerate(text.split("**")):
        if not chunk:
            continue
        run = p.add_run(chunk)
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = index % 2 == 1
    return p


def shade(cell, colour: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), colour)
    tc_pr.append(shd)


def add_table(doc, spec, width_in=3.4, number="I"):
    para(doc, f"TABLE {number}. {spec['caption']}", size=8, align="left",
         space_before=6, space_after=2)
    rows = [spec["header"]] + spec["rows"]
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.width = Inches(width_in * spec["widths"][c])
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            run.font.name = FONT
            run.font.size = Pt(7.5)
            run.font.bold = r == 0
            if r == 0:
                shade(cell, "E8EEF3")
    para(doc, "", size=4, space_after=0)


def main() -> int:
    doc = Document()
    style_doc(doc)
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    for attr in ("top_margin", "bottom_margin"):
        setattr(section, attr, Inches(0.75))
    section.left_margin = section.right_margin = Inches(0.7)
    set_columns(section, 1)

    para(doc, C.TITLE, size=20, bold=True, align="center", space_before=0, space_after=10)
    para(doc, C.AUTHOR, size=11, align="center", space_after=1)
    for line in C.AFFILIATION:
        para(doc, line, size=9, italic=True, align="center", space_after=1)
    para(doc, "", size=6, space_after=4)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    lead = p.add_run("Abstract\u2014")
    lead.font.bold = True
    lead.font.size = Pt(9)
    lead.font.name = FONT
    body = p.add_run(C.ABSTRACT)
    body.font.size = Pt(9)
    body.font.name = FONT

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    lead = p.add_run("Index Terms\u2014")
    lead.font.bold = True
    lead.font.italic = True
    lead.font.size = Pt(9)
    lead.font.name = FONT
    terms = p.add_run(C.INDEX_TERMS)
    terms.font.size = Pt(9)
    terms.font.italic = True
    terms.font.name = FONT

    # continuous section break -> two columns
    body_section = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(body_section, 2)
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(body_section, attr, getattr(section, attr))

    numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    figure_added = False
    table_no = 1
    for index, sec in enumerate(C.SECTIONS):
        para(doc, f"{numerals[index]}.  {sec['heading'].upper()}", size=10, bold=False,
             caps=True, align="center", space_before=10, space_after=4)
        for kind, payload in sec["blocks"]:
            if kind == "p":
                rich_para(doc, payload, size=10)
            elif kind == "table":
                add_table(doc, payload, number=["I", "II", "III", "IV", "V"][table_no - 1])
                table_no += 1
            elif kind == "fig":
                doc.add_picture(str(Path(__file__).with_name(payload["file"])), width=Inches(3.3))
                para(doc, payload["caption"], size=8, align="left", space_before=2, space_after=6)
        if sec["heading"].startswith("Feature Engineering") and not figure_added:
            doc.add_picture(str(Path(__file__).with_name("fig1_pipeline.png")), width=Inches(3.3))
            para(doc, "Fig. 1.  End-to-end workflow. Each stage is a script with a fixed seed and a "
                      "hashed output, so every reported number regenerates from the raw census files.",
                 size=8, align="left", space_before=2, space_after=6)
            figure_added = True

    para(doc, "REFERENCES", size=10, caps=True, align="center", space_before=10, space_after=4)
    for index, ref in enumerate(C.REFERENCES, start=1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.first_line_indent = Inches(-0.22)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(f"[{index}]\t{ref}")
        run.font.name = FONT
        run.font.size = Pt(7.5)

    doc.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"  sections: {len(doc.sections)} (title block single column, body two columns)")
    print(f"  paragraphs: {len(doc.paragraphs)} | tables: {len(doc.tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
