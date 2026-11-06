import json
from pathlib import Path

import pymupdf

from ingest.parse import parse_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_html_sections_and_heading_paths():
    result = parse_file(FIXTURES / "sample.html")
    assert result["source"] == "sample.html"
    assert result["doc_type"] == "html"

    sections = [(s["heading_path"], s["text"]) for s in result["sections"]]
    assert sections == [
        ([], "Intro text before any heading."),
        (["Benefits"], "Overview of benefits."),
        (["Benefits", "Parental Leave"],
         "Employees get 16 weeks of paid leave.\nApplies to all parents.\nMust give notice."),
        (["Benefits", "Pensions"], "Employer matches 5%."),
        (["Conduct"], "Be kind."),
    ]
    assert all(s["page"] is None for s in result["sections"])


def test_html_drops_boilerplate():
    all_text = " ".join(s["text"] for s in parse_file(FIXTURES / "sample.html")["sections"])
    for junk in ("Home | About", "console.log", "Copyright", "comment", "color: red"):
        assert junk not in all_text


def make_pdf(path: Path) -> None:
    """Build a 3-page PDF with bookmarks: Intro (p1), Benefits (p2) > Leave (p3)."""
    doc = pymupdf.open()
    for text in ("Welcome to the handbook.", "Benefits overview.", "Leave is 16 weeks."):
        doc.new_page().insert_text((72, 72), text)
    doc.set_toc([[1, "Intro", 1], [1, "Benefits", 2], [2, "Leave", 3]])
    doc.save(path)


def test_pdf_pages_and_heading_paths(tmp_path):
    pdf = tmp_path / "handbook.pdf"
    make_pdf(pdf)
    result = parse_file(pdf)
    assert result["doc_type"] == "pdf"
    assert [(s["page"], s["heading_path"], s["text"]) for s in result["sections"]] == [
        (1, ["Intro"], "Welcome to the handbook."),
        (2, ["Benefits"], "Benefits overview."),
        (3, ["Benefits", "Leave"], "Leave is 16 weeks."),
    ]


def test_cli_writes_json(tmp_path, monkeypatch):
    import sys
    from ingest import parse

    out = tmp_path / "parsed"
    monkeypatch.setattr(sys, "argv", ["parse", str(FIXTURES / "sample.html"), "--out", str(out)])
    assert parse.main() == 0
    saved = json.loads((out / "sample.html.json").read_text())
    assert saved["source"] == "sample.html"
    assert len(saved["sections"]) == 5
