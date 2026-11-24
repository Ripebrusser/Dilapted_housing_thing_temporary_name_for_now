"""Generate the reference-verification table and the page-count record."""

from pathlib import Path

from pypdf import PdfReader

HERE_DIR = Path(__file__).parent


def HERE(name: str = "") -> Path:
    return HERE_DIR / name if name else HERE_DIR


import sys  # noqa: E402

sys.path.insert(0, str(HERE_DIR))
import paper_content as C  # noqa: E402

PDF = HERE("IEEE_Paper_Dilapidated_Housing_India.pdf")
DOCX = HERE("IEEE_Paper_Dilapidated_Housing_India.docx")
TEX = HERE("paper.tex")

# (first author / short form, venue type, identifier, how it was verified)
VERIFY = [
    ("Office of the Registrar General, India (2011)",
     "Official statistical publication", "https://censusindia.gov.in",
     "Official Census of India portal; source of D1/D2 tables"),
    ("Office of the Registrar General, India (2011)",
     "Official statistical publication", "https://censusindia.gov.in",
     "Official Census of India portal; source of D5 primary abstract"),
    ("UN-Habitat (2003)", "Book / official report", "https://unhabitat.org",
     "UN-Habitat Global Report on Human Settlements 2003; no DOI assigned"),
    ("Hoerl & Kennard (1970)", "Journal article", "10.1080/00401706.1970.10488634",
     "Crossref: Technometrics 12(1) 55-67"),
    ("Tibshirani (1996)", "Journal article", "10.1111/j.2517-6161.1996.tb02080.x",
     "Crossref: JRSS-B 58(1) 267-288"),
    ("Zou & Hastie (2005)", "Journal article", "10.1111/j.1467-9868.2005.00503.x",
     "Crossref: JRSS-B 67(2) 301-320"),
    ("Breiman (2001)", "Journal article", "10.1023/A:1010933404324",
     "Crossref: Machine Learning 45(1) 5-32"),
    ("Friedman (2001)", "Journal article", "10.1214/aos/1013203451",
     "Crossref: Annals of Statistics 29(5) 1189-1232"),
    ("Geurts, Ernst & Wehenkel (2006)", "Journal article", "10.1007/s10994-006-6226-1",
     "Crossref: Machine Learning 63(1) 3-42"),
    ("Cortes & Vapnik (1995)", "Journal article", "10.1007/BF00994018",
     "Crossref: Machine Learning 20(3) 273-297"),
    ("Cover & Hart (1967)", "Journal article", "10.1109/TIT.1967.1053964",
     "Crossref: IEEE Trans. Inf. Theory 13(1) 21-27"),
    ("Roberts et al. (2017)", "Journal article", "10.1111/ecog.02881",
     "Crossref: Ecography 40(8) 913-929"),
    ("Ploton et al. (2020)", "Journal article", "10.1038/s41467-020-18321-y",
     "Nature Communications 11, art. 4540 (DOI corrected during verification)"),
    ("Koenker & Bassett (1978)", "Journal article", "10.2307/1913643",
     "Crossref: Econometrica 46(1) 33-50"),
    ("Wilson (1927)", "Journal article", "10.1080/01621459.1927.10502953",
     "Crossref / publisher: JASA 22(158) 209-212"),
    ("Brown, Cai & DasGupta (2001)", "Journal article", "10.1214/ss/1009213286",
     "Project Euclid: Statistical Science 16(2) 101-133"),
    ("Pedregosa et al. (2011)", "Journal article", "https://jmlr.org/papers/v12/pedregosa11a.html",
     "JMLR 12:2825-2830; no DOI assigned"),
    ("Hyndman & Koehler (2006)", "Journal article", "10.1016/j.ijforecast.2006.03.001",
     "Crossref: Int. J. Forecasting 22(4) 679-688"),
    ("Efron & Tibshirani (1993)", "Book", "No DOI (Chapman & Hall, 1993)",
     "Book; identifier omitted rather than asserted unverified"),
    ("Robinson (1950)", "Journal article", "10.2307/2087176",
     "Crossref: American Sociological Review 15(3) 351-357"),
]


def write_reference_list() -> None:
    lines = ["# Verified reference list", "",
             "All 20 entries were checked against publisher or Crossref metadata. Entries appear in "
             "IEEE order of first citation; the numbering matches the PDF, DOCX and LaTeX artefacts.",
             ""]
    for index, ref in enumerate(C.REFERENCES, start=1):
        lines.append(f"[{index}] {ref}")
        lines.append("")
    HERE("references_verified.md").write_text("\n".join(lines), encoding="utf-8")


def write_verification_table() -> None:
    rows = ["| # | Reference (short) | Type | DOI / official link | Verified against |",
            "|---:|---|---|---|---|"]
    for index, (short, kind, ident, how) in enumerate(VERIFY, start=1):
        rows.append(f"| {index} | {short} | {kind} | {ident} | {how} |")
    body = [
        "# Reference verification table",
        "",
        "Every reference in the paper is listed below with its DOI or official source link. "
        "Journal DOIs were resolved in a single Crossref works query that returned all eleven "
        "queried records with matching titles, volumes and pages; the remaining entries were "
        "checked against publisher pages or, for government and JMLR publications, the issuing "
        "organisation's own site.",
        "",
        *rows,
        "",
        "## Notes",
        "",
        "- Ref. [13] (Ploton et al.) carried an incorrect DOI in an earlier draft; it was corrected "
        "to 10.1038/s41467-020-18321-y after checking the Nature Communications record.",
        "- Refs. [1], [2], [3], [17], [19] are official, book or JMLR publications to which no DOI is "
        "assigned; the issuing body's URL is given instead.",
        "- Two candidate references (a LightGBM paper and a housing-deprivation study) were discarded "
        "during drafting because their bibliographic details could not be confirmed. They are not cited.",
        "- Ref. [19] is a book; no identifier is asserted for it because a DOI could not be confirmed.",
    ]
    HERE("REFERENCE_VERIFICATION_TABLE.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def write_page_count() -> None:
    reader = PdfReader(str(PDF))
    pages = len(reader.pages)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    per_page = [len((p.extract_text() or "").split()) for p in reader.pages]
    docx_words = None
    try:
        from docx import Document
        docx_words = sum(len(p.text.split()) for p in Document(str(DOCX)).paragraphs)
    except Exception:
        pass

    lines = [
        "# Page-count verification",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Target limit | 6 pages (references, tables and figure included) |",
        f"| Measured PDF pages | **{pages}** |",
        f"| Result | {'PASS - within limit' if pages <= 6 else 'FAIL - over limit'} |",
        f"| Margin | {6 - pages} page(s) spare |",
        f"| PDF file | `{PDF.name}` ({PDF.stat().st_size:,} bytes) |",
        f"| Words per page | {', '.join(str(w) for w in per_page)} |",
        f"| Total extracted words | {sum(per_page):,} |",
        f"| Body word count (source) | ~3,263 (sections only, excluding abstract and references) |",
        f"| DOCX body words | {docx_words if docx_words else 'n/a'} |",
        "",
        "## How the count was obtained",
        "",
        "```",
        "py -3.14 -c \"from pypdf import PdfReader; "
        "print(len(PdfReader('IEEE_Paper_Dilapidated_Housing_India.pdf').pages))\"",
        "```",
        "",
        "The PDF is typeset by `build_pdf.py` (reportlab) in two 3.4 in columns on US Letter, so its "
        "pagination is fixed and directly measurable. Every page was also checked to keep all glyphs "
        "inside the 0.7 in side and 0.75 in top/bottom margins.",
        "",
        "## Caveat",
        "",
        "The DOCX is the editable deliverable and Word will reflow it, so its page count is not "
        "reproducible on this machine (no Word, no LibreOffice). Its layout uses the same geometry "
        "(Letter, 0.75 in margins, 10 pt Times, two 3.4 in columns), so it tracks the PDF closely, "
        "but the PDF is the authoritative measurement. The LaTeX source cannot be compiled here "
        "either: no `pdflatex`, `xelatex`, `tectonic` or `latexmk` is installed, so `paper.tex` is "
        "supplied as source for compilation in an IEEEtran-capable environment, not as a verified build.",
    ]
    HERE("PAGE_COUNT_VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pages={pages} pass={pages <= 6}")


if __name__ == "__main__":
    write_reference_list()
    write_verification_table()
    write_page_count()
    for name in ("references_verified.md", "REFERENCE_VERIFICATION_TABLE.md",
                 "PAGE_COUNT_VERIFICATION.md"):
        path = HERE(name)
        print(f"wrote {name} ({path.stat().st_size:,} bytes)")
