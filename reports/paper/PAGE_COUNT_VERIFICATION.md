# Page-count verification

| Item | Value |
|---|---|
| Target limit | 6 pages (references, tables and figure included) |
| Measured PDF pages | **5** |
| Result | PASS - within limit |
| Margin | 1 page(s) spare |
| PDF file | `IEEE_Paper_Dilapidated_Housing_India.pdf` (155,450 bytes) |
| Words per page | 675, 1131, 1021, 1218, 287 |
| Total extracted words | 4,332 |
| Body word count (source) | ~3,263 (sections only, excluding abstract and references) |
| DOCX body words | 4141 |

## How the count was obtained

```
py -3.14 -c "from pypdf import PdfReader; print(len(PdfReader('IEEE_Paper_Dilapidated_Housing_India.pdf').pages))"
```

The PDF is typeset by `build_pdf.py` (reportlab) in two 3.4 in columns on US Letter, so its pagination is fixed and directly measurable. Every page was also checked to keep all glyphs inside the 0.7 in side and 0.75 in top/bottom margins.

## Caveat

The DOCX is the editable deliverable and Word will reflow it, so its page count is not reproducible on this machine (no Word, no LibreOffice). Its layout uses the same geometry (Letter, 0.75 in margins, 10 pt Times, two 3.4 in columns), so it tracks the PDF closely, but the PDF is the authoritative measurement. The LaTeX source cannot be compiled here either: no `pdflatex`, `xelatex`, `tectonic` or `latexmk` is installed, so `paper.tex` is supplied as source for compilation in an IEEEtran-capable environment, not as a verified build.
