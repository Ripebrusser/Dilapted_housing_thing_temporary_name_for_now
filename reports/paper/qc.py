"""Quality control for the generated paper artefacts."""

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
import paper_content as C  # noqa: E402
from pypdf import PdfReader  # noqa: E402

HERE = Path(__file__).parent
PDF = HERE / "IEEE_Paper_Dilapidated_Housing_India.pdf"
DOCX = HERE / "IEEE_Paper_Dilapidated_Housing_India.docx"
TEX = HERE / "paper.tex"

FORBIDDEN = [
    "homeless",
    "causally",
    "causal effect",
    "first of its kind",
    "proves an information ceiling",
    "proved an information ceiling",
    "statistically significant",
    "present-day",
    "current-day",
]
# phrases allowed only inside an explicit negation
NEGATION = ("not ", "no ", "cannot", "never", "neither", "without", "does not", "is not")


def main() -> int:
    report, problems = [], []

    def check(ok, label, detail=""):
        report.append(f"{'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            problems.append(label)

    # ---- 1. artefacts exist
    for path in (PDF, DOCX, TEX, HERE / "fig1_pipeline.png"):
        check(path.exists(), f"artefact exists: {path.name}",
              f"{path.stat().st_size:,} bytes" if path.exists() else "missing")

    # ---- 2. page count
    reader = PdfReader(str(PDF))
    pages = len(reader.pages)
    check(pages <= 6, "PDF page count <= 6", f"{pages} pages")

    text = "\n".join(p.extract_text() or "" for p in reader.pages)

    # ---- 3. encoding / placeholders
    check("\u2014" in text or "Abstract" in text, "PDF text extracts cleanly")
    for bad in ("â€", "Â", "Ã", "nan", "None", "TODO", "XXX", "Lorem"):
        # word boundaries so "Provenance"/"dominant" do not trip the 'nan' check
        check(re.search(r"\b" + re.escape(bad) + r"\b", text) is None,
              f"no artefact of '{bad}'")

    # ---- 4. citations: every section citation resolves, order is monotonic
    body = " ".join(p for s in C.SECTIONS for k, p in s["blocks"] if k == "p")
    cites = [int(n) for n in re.findall(r"\[(\d+)\]", body)]
    check(len(cites) > 0, "body contains numeric citations", f"{len(cites)} citations")
    check(all(1 <= c <= len(C.REFERENCES) for c in cites),
          "every cited number has a bibliography entry",
          f"max cited = {max(cites)}, refs = {len(C.REFERENCES)}")
    first_seen, order_ok = [], True
    for c in cites:
        if c not in first_seen:
            if first_seen and c != max(first_seen) + 1:
                order_ok = False
            first_seen.append(c)
    check(order_ok, "citations first appear in order 1..N", f"order = {first_seen}")
    check(sorted(set(cites)) == list(range(1, len(C.REFERENCES) + 1)),
          "every reference is cited at least once")
    check(len(C.REFERENCES) >= 15, "reference count >= 15", f"{len(C.REFERENCES)} references")

    # ---- 5. integrity phrasing
    lowered = body.lower()
    for phrase in FORBIDDEN:
        if phrase not in lowered:
            check(True, f"phrase absent: '{phrase}'")
            continue
        bad = [m.start() for m in re.finditer(re.escape(phrase), lowered)
               if not any(neg in lowered[max(0, m.start() - 120):m.start()] for neg in NEGATION)]
        check(not bad, f"phrase only used in negated form: '{phrase}'",
              f"{len(bad)} unnegated use(s)")

    # ---- 6. required disclosures and numbers
    required = {
        "full-sample screening limitation": "screen",
        "grouped CV is primary": "grouped",
        "0.0242 difference mentioned": "0.0242",
        "no promotion stated": "no model was promoted",
        "0.9994 target validation": "0.9994",
    }
    for label, needle in required.items():
        check(needle in lowered, f"present: {label}")

    # ---- 7. table sanity
    tables = [p for s in C.SECTIONS for k, p in s["blocks"] if k == "table"]
    check(len(tables) >= 3, "tables present", f"{len(tables)} tables")
    for t in tables:
        widths = t["widths"]
        ragged = [r for r in t["rows"] if len(r) != len(t["header"])]
        check(abs(sum(widths) - 1.0) < 1e-6, f"table widths sum to 1: {t['caption'][:38]}",
              f"{sum(widths):.3f}")
        check(not ragged, f"table rows rectangular: {t['caption'][:38]}")

    # ---- 8. docx structure
    from docx import Document
    from docx.oxml.ns import qn
    doc = Document(str(DOCX))
    cols = [s._sectPr.find(qn("w:cols")).get(qn("w:num")) for s in doc.sections]
    check(cols[-1] == "2", "DOCX body section uses two columns", f"section cols = {cols}")
    check(len(doc.tables) == len(tables), "DOCX table count matches content",
          f"{len(doc.tables)} vs {len(tables)}")
    docx_words = sum(len(p.text.split()) for p in doc.paragraphs)
    check(docx_words > 2500, "DOCX body word count plausible", f"{docx_words} words")

    # ---- 9. latex structure
    tex_src = TEX.read_text(encoding="utf-8")
    for needle in ("\\documentclass[conference]{IEEEtran}", "\\begin{abstract}",
                   "\\begin{IEEEkeywords}", "\\begin{thebibliography}", "\\end{document}"):
        check(needle in tex_src, f"LaTeX contains {needle}")
    check(tex_src.count("\\bibitem") == len(C.REFERENCES), "LaTeX bibitem count matches",
          f"{tex_src.count('\\bibitem')}")
    check("\\begin{figure}" in tex_src, "LaTeX includes the workflow figure")

    out = ["# Paper QC report", "", f"Artefacts: {PDF.name} ({pages} pages), {DOCX.name}, "
           f"{TEX.name}", ""] + report
    out.append("")
    out.append(f"RESULT: {'ALL CHECKS PASSED' if not problems else 'FAILURES: ' + '; '.join(problems)}")
    (HERE / "QC_REPORT.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    print("\n".join(report))
    print()
    print(f"PAGES: {pages}")
    print("RESULT:", "ALL CHECKS PASSED" if not problems else f"FAILURES: {problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
