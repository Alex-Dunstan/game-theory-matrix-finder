# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pypdf>=5.0",
#   "reportlab>=4.0",
# ]
# ///
"""Apply the project author, ORCID, and citation block to the tracked PDF.

Run from the repository root with:
    uv run scripts/add_pdf_identity.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "output/pdf/game-theory-matrix-finder-writeup.pdf"
AUTHOR = "Alex Lewis Dunstan"
ORCID = "0009-0007-7869-809X"
ORCID_URL = f"https://orcid.org/{ORCID}"
REPOSITORY_URL = "https://github.com/Alex-Dunstan/game-theory-matrix-finder"
PAGE_WIDTH, PAGE_HEIGHT = 612, 792


def wrapped_lines(text: str, font: str, font_size: float, max_width: float) -> list[str]:
    """Wrap text using the actual PDF font metrics."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and stringWidth(candidate, font, font_size) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def page_overlay(page_number: int, page_count: int) -> BytesIO:
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    # Replace prior overlay content so the script can run repeatedly without stacking.
    canvas.setFillColor(white)
    canvas.rect(0, 20, PAGE_WIDTH, 36, fill=1, stroke=0)
    if page_number == 1:
        canvas.rect(105, 614, 140, 30, fill=1, stroke=0)
    if page_number == page_count:
        canvas.rect(68, 78, PAGE_WIDTH - 136, 86, fill=1, stroke=0)

    canvas.setStrokeColor(HexColor("#a8a8a8"))
    canvas.setLineWidth(0.35)
    canvas.line(50, 50, PAGE_WIDTH - 50, 50)
    canvas.setFont("Times-Roman", 7.2)
    canvas.setFillColor(HexColor("#555555"))
    footer = f"{AUTHOR}  |  ORCID: {ORCID}  |  github.com/Alex-Dunstan"
    footer_width = stringWidth(footer, "Times-Roman", 7.2)
    footer_x = (PAGE_WIDTH - footer_width) / 2
    canvas.drawString(footer_x, 35, footer)
    orcid_x = footer_x + stringWidth(f"{AUTHOR}  |  ORCID: ", "Times-Roman", 7.2)
    canvas.linkURL(ORCID_URL, (orcid_x, 32, orcid_x + stringWidth(ORCID, "Times-Roman", 7.2), 43), relative=0)
    repo_x = footer_x + stringWidth(f"{AUTHOR}  |  ORCID: {ORCID}  |  ", "Times-Roman", 7.2)
    canvas.linkURL(REPOSITORY_URL, (repo_x, 32, footer_x + footer_width, 43), relative=0)

    if page_number == 1:
        canvas.setFillColor(HexColor("#3d3d3d"))
        canvas.setFont("Times-Roman", 10)
        byline_width = stringWidth(AUTHOR, "Times-Roman", 10)
        byline_x = 175 - (byline_width / 2)
        canvas.drawString(byline_x, 631, AUTHOR)
        canvas.setFont("Times-Roman", 8)
        orcid_line = f"ORCID: {ORCID}"
        orcid_width = stringWidth(orcid_line, "Times-Roman", 8)
        orcid_line_x = 175 - (orcid_width / 2)
        canvas.drawString(orcid_line_x, 616, orcid_line)
        canvas.linkURL(ORCID_URL, (orcid_line_x, 613, orcid_line_x + orcid_width, 624), relative=0)

    if page_number == page_count:
        citation = (
            "Dunstan, A. L. (2026). Every 2x2 Game, Counted: An exhaustive computational "
            "tour of pure-strategy Nash equilibria. Game Theory Matrix Finder. "
            f"{REPOSITORY_URL}"
        )
        canvas.setFillColor(HexColor("#3d3d3d"))
        canvas.setFont("Times-Bold", 8)
        canvas.drawString(76, 146, "Citation")
        canvas.setFont("Times-Roman", 7.4)
        y = 133
        for line in wrapped_lines(citation, "Times-Roman", 7.4, PAGE_WIDTH - 152):
            canvas.drawString(76, y, line)
            y -= 10
        orcid_line = f"Author ORCID: {ORCID_URL}"
        canvas.drawString(76, y - 2, orcid_line)
        canvas.linkURL(REPOSITORY_URL, (72, 104, PAGE_WIDTH - 72, 149), relative=0)
        canvas.linkURL(ORCID_URL, (72, y - 5, 300, y + 7), relative=0)

    canvas.save()
    stream.seek(0)
    return stream


def main() -> None:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    tracked_pdf = PDF_PATH.relative_to(ROOT).as_posix()
    baseline = subprocess.run(
        ["git", "show", f"HEAD:{tracked_pdf}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    reader = PdfReader(BytesIO(baseline.stdout))
    writer = PdfWriter()
    page_count = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        page.merge_page(PdfReader(page_overlay(index, page_count)).pages[0])
        writer.add_page(page)

    writer.add_metadata(
        {
            "/Title": "Every 2x2 Game, Counted",
            "/Author": AUTHOR,
            "/Subject": "An exhaustive computational tour of pure-strategy Nash equilibria",
            "/Keywords": f"game theory, Nash equilibrium, payoff matrices, computational economics, ORCID {ORCID}",
        }
    )
    temporary_path = PDF_PATH.with_suffix(".tmp.pdf")
    with temporary_path.open("wb") as output:
        writer.write(output)
    temporary_path.replace(PDF_PATH)


if __name__ == "__main__":
    main()
