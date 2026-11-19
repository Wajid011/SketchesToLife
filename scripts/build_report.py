#!/usr/bin/env python3
"""Build report/REPORT.pdf from report/REPORT.md.

The Markdown is rendered to HTML with Python-Markdown and printed to an A4 PDF by a
headless Chrome / Chromium / Edge, so no LaTeX or pandoc installation is needed::

    pip install markdown
    python scripts/build_report.py            # writes report/REPORT.pdf

Set ``CHROME`` to the browser executable if it is not found automatically.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

CSS = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Cambria", "Georgia", "Times New Roman", serif; font-size: 9.6pt;
       line-height: 1.32; color: #111; margin: 0; text-align: justify; hyphens: auto; }
.title-block { text-align: center; margin-bottom: 6pt; }
.title-block h1 { font-size: 15pt; line-height: 1.2; margin: 0 0 5pt 0; }
.title-block p { margin: 0; font-size: 9.4pt; text-align: center; }
h2 { font-size: 11pt; margin: 8pt 0 3pt 0; border-bottom: 0.6pt solid #999;
     padding-bottom: 1pt; break-after: avoid; }
p { margin: 0 0 4.5pt 0; }
a { color: #0b4f9c; text-decoration: none; }
code { font-family: "Consolas", monospace; font-size: 8.6pt; }
table { border-collapse: collapse; margin: 2pt auto 6pt auto; font-size: 8.6pt;
        break-inside: avoid; }
th, td { border-top: 0.5pt solid #bbb; border-bottom: 0.5pt solid #bbb;
         padding: 1.6pt 6pt; text-align: center; }
th { background: #eef2f7; }
td:first-child, th:first-child { text-align: left; }
figure { margin: 4pt 0 6pt 0; text-align: center; break-inside: avoid; }
figure img { width: 100%; display: block; margin: 0 auto 2pt auto; }
figcaption, .caption { font-size: 8.4pt; color: #333; text-align: center; margin-top: 1pt; }
.caption { margin-bottom: 1pt; break-after: avoid; }
.row { display: flex; gap: 10pt; align-items: center; break-inside: avoid; }
.col { flex: 1; }
.row .col:first-child { flex: 0.8; }
figure .row { align-items: flex-start; gap: 6pt; }
figure img.ui { width: 44%; }
.stack { width: 56%; }
.refs { font-size: 8.4pt; line-height: 1.3; text-align: left; }
"""


def find_browser() -> str:
    """Return the path of a Chromium-based browser able to print PDFs."""
    candidates = [
        os.environ.get("CHROME", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    names = ["google-chrome", "chromium", "chromium-browser", "chrome", "msedge"]
    candidates += [shutil.which(n) or "" for n in names]
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit("No Chrome/Chromium/Edge found; set the CHROME environment variable.")


def build(md_path: Path) -> Path:
    """Render ``md_path`` to HTML and PDF next to it; return the PDF path."""
    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "md_in_html", "sane_lists"],
    )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{md_path.stem}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    html_path = md_path.with_suffix(".html")
    pdf_path = md_path.with_suffix(".pdf")
    html_path.write_text(html, encoding="utf-8")
    subprocess.run(
        [
            find_browser(),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    html_path.unlink()
    return pdf_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, default=ROOT / "report" / "REPORT.md")
    args = ap.parse_args(argv)
    print(f"Wrote {build(args.input)}")


if __name__ == "__main__":
    main()
