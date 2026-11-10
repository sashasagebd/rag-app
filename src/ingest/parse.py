"""Parse PDF/HTML files into clean text sections with metadata, saved as JSON.

Usage (from the repo root):
    python -m ingest.parse                      # every file in raw_dir
    python -m ingest.parse data/raw/foo.pdf     # specific files or directories

Output: one JSON file per input, at <parsed_dir>/<input filename>.json:
    {
      "source": "foo.pdf",
      "doc_type": "pdf" | "html",
      "parsed_at": "2026-11-05T12:00:00+00:00",
      "sections": [
        {"page": 3, "heading_path": ["Benefits", "Leave"], "text": "..."}
      ]
    }

How sections are made:
  - PDF:  one section per page. `heading_path` comes from the PDF's bookmarks
          (table of contents): the path in effect once all bookmarks up to and
          including this page are applied. No bookmarks -> heading_path = [].
  - HTML: a new section starts at every <h1>-<h6>. `page` is null.
          Each table row becomes one line, cells joined with " | ".

This step does NOT chunk. Chunking reads these JSON files later.
"""

import argparse
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup, NavigableString, Tag

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Elements whose text is page furniture rather than document content.
DROP_TAGS = ["script", "style", "noscript", "template", "nav", "header", "footer", "aside", "form"]
# Elements that start a new line of text.
BLOCK_TAGS = {
    "p", "div", "li", "pre", "blockquote", "td", "th", "tr", "dt", "dd",
    "section", "article", "main", "figcaption", "caption", "table", "ul", "ol",
} | HEADING_TAGS

SUFFIX_TO_TYPE = {".pdf": "pdf", ".html": "html", ".htm": "html"}


def clean_text(text: str) -> str:
    """Normalise whitespace: trim lines, collapse runs of spaces, at most one blank line."""
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def push_heading(stack: list[tuple[int, str]], level: int, title: str) -> None:
    """Update a heading stack in place: drop headings at the same or deeper level, then add this one."""
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


# ---------- PDF ----------

def parse_pdf(path: Path) -> list[dict]:
    sections = []
    with pymupdf.open(path) as doc:
        toc = doc.get_toc(simple=True)  # [[level, title, page_number], ...], pages are 1-based
        stack: list[tuple[int, str]] = []
        toc_i = 0
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            while toc_i < len(toc) and toc[toc_i][2] <= page_no:
                level, title, _ = toc[toc_i]
                push_heading(stack, level, title.strip())
                toc_i += 1
            text = clean_text(page.get_text("text"))
            if text:
                sections.append({
                    "page": page_no,
                    "heading_path": [title for _, title in stack],
                    "text": text,
                })
    return sections


# ---------- HTML ----------

def parse_html(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    root = soup.body or soup

    sections = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []  # text of the current section, one entry per block

    def flush() -> None:
        text = clean_text("\n".join(lines))
        if text:
            sections.append({"page": None, "heading_path": [t for _, t in stack], "text": text})
        lines.clear()

    # Where the previous piece of text came from, to decide how to join the next one:
    # same block -> " ", same table row but a new cell -> " | ", otherwise a new line.
    last_block = last_row = last_cell = None
    for node in root.descendants:
        if isinstance(node, Tag) and node.name in HEADING_TAGS:
            flush()
            push_heading(stack, int(node.name[1]), clean_text(node.get_text(" ")))
            last_block = last_row = last_cell = None
        elif type(node) is NavigableString:  # excludes comments, doctype, CDATA
            text = node.strip()
            if not text or any(p.name in HEADING_TAGS for p in node.parents):
                continue
            block = next((p for p in node.parents if p.name in BLOCK_TAGS), None)
            cell = node.find_parent(["td", "th"])
            row = cell.find_parent("tr") if cell else None

            if lines and row is not None and row is last_row:
                lines[-1] += (" " if cell is last_cell else " | ") + text
            elif lines and block is last_block:
                lines[-1] += " " + text
            else:
                lines.append(text)
            last_block, last_row, last_cell = block, row, cell
    flush()
    return sections


# ---------- driver ----------

def parse_file(path: Path) -> dict:
    doc_type = SUFFIX_TO_TYPE.get(path.suffix.lower())
    if doc_type is None:
        raise ValueError(f"unsupported file type: {path.name}")
    sections = parse_pdf(path) if doc_type == "pdf" else parse_html(path)
    return {
        "source": path.name,
        "doc_type": doc_type,
        "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sections": sections,
    }


def collect_inputs(args: list[str]) -> list[Path]:
    paths = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            paths += sorted(f for f in p.iterdir() if f.suffix.lower() in SUFFIX_TO_TYPE)
        else:
            paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="files or directories (default: raw_dir from config)")
    parser.add_argument("--out", help="output directory (default: parsed_dir from config)")
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()

    config = tomllib.loads(Path(args.config).read_text())
    inputs = collect_inputs(args.inputs or [config["paths"]["raw_dir"]])
    out_dir = Path(args.out or config["paths"]["parsed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inputs:
        print("no PDF/HTML files found", file=sys.stderr)
        return 1

    failures = 0
    for path in inputs:
        try:
            result = parse_file(path)
        except Exception as e:  # keep going so one bad file doesn't stop the batch
            print(f"FAIL {path}: {e}", file=sys.stderr)
            failures += 1
            continue
        out_path = write_parsed(result, out_dir)
        print(f"ok   {path} -> {out_path} ({len(result['sections'])} sections)")
    return 1 if failures else 0


def write_parsed(result: dict, out_dir: Path) -> Path:
    """Save one parse result as <out_dir>/<source filename>.json and return the path."""
    out_path = out_dir / f"{result['source']}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    sys.exit(main())
