"""Render paper_content.py to IEEEtran LaTeX."""

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
import paper_content as C  # noqa: E402

OUT = Path(__file__).with_name("paper.tex")


def tex(text: str) -> str:
    """Escape LaTeX specials; keep unicode quotes/dashes (utf8 inputenc)."""
    text = text.replace("\\", r"\textbackslash{}")
    for char, repl in (("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_"),
                       ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(char, repl)
    text = text.replace("||", r"\textbar\textbar{}")
    text = text.replace("alpha =", r"$\alpha$ =").replace("tau =", r"$\tau$ =")
    text = text.replace("R-squared", r"$R^2$")
    text = text.replace(" plus or minus ", r" $\pm$ ")
    text = text.replace(r"$\alpha$ = 100$", r"$\alpha$ = 100")
    return text


def table_block(spec: dict) -> list[str]:
    n = len(spec["header"])
    align = "l" + "c" * (n - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + tex(spec["caption"]) + "}",
             r"\label{tab:" + str(abs(hash(spec["caption"])) % 10000) + "}",
             r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{" + align + "}", r"\toprule"]
    lines.append(" & ".join(r"\textbf{" + tex(h) + "}" for h in spec["header"]) + r" \\")
    lines.append(r"\midrule")
    for row in spec["rows"]:
        lines.append(" & ".join(tex(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def main() -> int:
    out: list[str] = []
    out += [
        r"% IEEE conference paper — compile with pdflatex (IEEEtran required).",
        r"% Source of truth: reports/paper/paper_content.py",
        r"\documentclass[conference]{IEEEtran}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage{array}",
        r"\usepackage{url}",
        r"\hyphenpenalty=1000",
        r"\begin{document}",
        r"\title{" + tex(C.TITLE) + "}",
        r"\author{\IEEEauthorblockN{" + tex(C.AUTHOR) + "}",
        r"\IEEEauthorblockA{" + r" \\ ".join(tex(a) for a in C.AFFILIATION) + "}}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        tex(C.ABSTRACT),
        r"\end{abstract}",
        "",
        r"\begin{IEEEkeywords}",
        tex(C.INDEX_TERMS),
        r"\end{IEEEkeywords}",
        "",
    ]

    for index, section in enumerate(C.SECTIONS, start=1):
        out.append(r"\section{" + tex(section["heading"]) + "}")
        out.append("")
        for kind, payload in section["blocks"]:
            if kind == "p":
                # bold lead-ins written as **...** in the content module
                text = tex(payload)
                text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
                out += [text, ""]
            elif kind == "table":
                out += table_block(payload)
            elif kind == "fig":
                out += [r"\begin{figure}[t]", r"\centering",
                        r"\includegraphics[width=\columnwidth]{" + payload["file"] + "}",
                        r"\caption{" + tex(payload["caption"]) + "}", r"\end{figure}", ""]
        # insert the workflow figure after the methodology section
        if section["heading"].startswith("Feature Engineering"):
            out += [r"\begin{figure}[t]", r"\centering",
                    r"\includegraphics[width=\columnwidth]{fig1_pipeline.png}",
                    r"\caption{End-to-end workflow. Each stage is a script with a fixed seed and a "
                    r"hashed output, so the reported numbers regenerate from the raw census files.}",
                    r"\label{fig:pipeline}", r"\end{figure}", ""]

    out += [r"\section*{References}", r"\small", r"\begin{thebibliography}{99}"]
    for ref in C.REFERENCES:
        out.append(r"\bibitem{ref} " + tex(ref))
    out += [r"\end{thebibliography}", "", r"\end{document}"]

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    words = sum(len(str(p).split()) for s in C.SECTIONS for _, p in s["blocks"] if isinstance(p, str))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(C.SECTIONS)} sections, "
          f"{len(C.REFERENCES)} references, ~{words} words of body text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
