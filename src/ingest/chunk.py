"""Turn parsed documents (data/parsed/*.json) into chunks for embedding.

Usage (from the repo root):
    python -m ingest.chunk data/parsed/dl-s01.html.json ... --collection dl-slides

Output: <chunks_dir>/<collection>.jsonl, one chunk per line:
    {"chunk_key": "dl-s01.html#3", "doc": "dl-s01.html",
     "sections": ["Course Introduction > Grading", ...], "pages": [],
     "text": "...", "n_tokens": 412, "chunker": "pack_sections:500"}

Chunkers are plain functions listed in CHUNKERS; config picks one by name.
Every chunker has the same signature:

    chunker(doc: dict, count_tokens: Callable[[str], int], max_tokens: int) -> list[dict]

and returns [{"sections": [...], "text": "..."}] for ONE document, so chunks
never cross documents. To add a chunker, write the function and add it to CHUNKERS.

Each section's text is prefixed with its heading path ("## A > B"), so the
embedding sees which slide or section the text came from.

With [chunking] clean_latex = true, inline math in the chunk text is rewritten
as plain symbols ("\\(1\\times1\\)" -> "1×1") so it embeds like ordinary words.
`sections` keeps the original headings, since citations must match source_refs.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

from common.config import load_config

TokenCounter = Callable[[str], int]


def section_name(section: dict) -> str:
    """The citation form of a section: its heading path joined with " > "."""
    return " > ".join(section["heading_path"])


def section_block(section: dict) -> str:
    """A section's text with its heading path as the first line."""
    name = section_name(section)
    return f"## {name}\n{section['text']}" if name else section["text"]


def split_to_fit(text: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """Split text into pieces of at most max_tokens: by lines first, by words for an over-long line."""
    pieces: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            pieces.append("\n".join(current))
            current.clear()

    for line in text.split("\n"):
        if count_tokens(line) > max_tokens:  # a single huge line: fall back to words
            flush()
            words: list[str] = []
            for word in line.split():
                if words and count_tokens(" ".join(words + [word])) > max_tokens:
                    pieces.append(" ".join(words))
                    words = []
                words.append(word)
            if words:
                pieces.append(" ".join(words))
        elif current and count_tokens("\n".join(current + [line])) > max_tokens:
            flush()
            current.append(line)
        else:
            current.append(line)
    flush()
    return pieces


# ---------- chunkers ----------

def pack_sections(doc: dict, count_tokens: TokenCounter, max_tokens: int) -> list[dict]:
    """Pack consecutive sections of one document into chunks of up to max_tokens.

    A section is never split unless it alone exceeds max_tokens; then it is split
    into pieces, each repeating the heading line. Each chunk lists the distinct
    section names and the PDF pages (empty for HTML) it covers.
    """
    chunks: list[dict] = []
    sections: list[str] = []
    pages: list[int] = []
    blocks: list[str] = []
    used = 0

    def flush():
        nonlocal used
        if blocks:
            chunks.append({"sections": list(dict.fromkeys(sections)), "pages": sorted(set(pages)),
                           "text": "\n\n".join(blocks)})
        sections.clear()
        pages.clear()
        blocks.clear()
        used = 0

    for section in doc["sections"]:
        block = section_block(section)
        size = count_tokens(block)
        page = [section["page"]] if section.get("page") else []
        if size > max_tokens:  # oversized section: give it chunks of its own
            flush()
            heading = block.split("\n", 1)[0] + "\n" if section_name(section) else ""
            for piece in split_to_fit(section["text"], max_tokens - count_tokens(heading), count_tokens):
                chunks.append({"sections": [section_name(section)], "pages": page, "text": heading + piece})
            continue
        if used + size > max_tokens:
            flush()
        sections.append(section_name(section))
        pages.extend(page)
        blocks.append(block)
        used += size
    flush()
    return chunks


CHUNKERS: dict[str, Callable[..., list[dict]]] = {
    "pack_sections": pack_sections,
}


# ---------- LaTeX cleaning ----------

LATEX_SYMBOLS = {
    "times": "×", "cdot": "·", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "approx": "≈", "neq": "≠",
    "in": "∈", "to": "→", "rightarrow": "→", "longrightarrow": "→", "infty": "∞", "partial": "∂",
    "nabla": "∇", "sum": "Σ", "prod": "Π", "odot": "⊙", "oslash": "⊘", "top": "ᵀ", "lVert": "‖",
    "rVert": "‖", "Vert": "‖", "ldots": "...", "cdots": "...", "pm": "±", "sim": "~", "mid": "|",
    "propto": "∝", "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "eta": "η", "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π", "sigma": "σ",
    "tau": "τ", "phi": "φ", "kappa": "κ", "Sigma": "Σ", "Delta": "Δ",
}
_UNWRAP = r"\\(?:mathrm|mathbf|mathbb|mathcal|boldsymbol|text|textbf|operatorname|mathit|hat|widehat|bar|tilde)\{([^{}]*)\}"
_MATH_SPAN = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)


def _clean_math(m: str) -> str:
    for _ in range(3):  # a few passes handle simple nesting like \frac{\sqrt{x}}{2}
        m = re.sub(r"\\textcolor\{[^{}]*\}\{([^{}]*)\}", r"\1", m)
        m = re.sub(r"\\[td]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", m)
        m = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", m)
        m = re.sub(_UNWRAP, r"\1", m)
    m = re.sub(r"\\(?:left|right|big|Big)\b", "", m)
    m = re.sub(r"\\[,;:! ]|\\q?quad\b", " ", m)
    m = re.sub(r"\\([A-Za-z]+)", lambda x: LATEX_SYMBOLS.get(x.group(1), x.group(1)), m)
    m = m.replace("^ᵀ", "ᵀ")  # K^\top -> Kᵀ
    m = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", m)
    m = m.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", m).strip()


def clean_latex(text: str) -> str:
    """Rewrite \\( ... \\) and \\[ ... \\] spans as plain text. Text outside math (e.g. code) is untouched."""
    return _MATH_SPAN.sub(lambda m: _clean_math(m.group(1) or m.group(2)), text)


# ---------- driver ----------

def make_token_counter(model_name: str) -> TokenCounter:
    """Count tokens the way the embedding model will see them (without its special tokens)."""
    from transformers import AutoTokenizer  # installed with sentence-transformers

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return lambda text: len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


def chunk_docs(docs: list[dict], method: str, max_tokens: int, count_tokens: TokenCounter,
               clean_latex_text: bool = False) -> list[dict]:
    """Run the chosen chunker over every document and add ids and bookkeeping fields."""
    chunker = CHUNKERS[method]
    label = f"{method}:{max_tokens}" + ("+clean_latex" if clean_latex_text else "")
    out = []
    for doc in docs:
        for i, c in enumerate(chunker(doc, count_tokens, max_tokens)):
            text = clean_latex(c["text"]) if clean_latex_text else c["text"]
            out.append({
                "chunk_key": f"{doc['source']}#{i}",
                "doc": doc["source"],
                "sections": c["sections"],
                "pages": c.get("pages", []),
                "text": text,
                "n_tokens": count_tokens(text),
                "chunker": label,
            })
    return out


def write_chunks(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def collect_parsed(args: list[str]) -> list[Path]:
    paths = []
    for arg in args:
        p = Path(arg)
        paths += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="parsed JSON files or directories")
    parser.add_argument("--collection", required=True, help="output name, e.g. dl-slides")
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in collect_parsed(args.inputs)]
    if not docs:
        print("no parsed JSON files found", file=sys.stderr)
        return 1

    count_tokens = make_token_counter(config["embedding"]["model"])
    ch = config["chunking"]
    chunks = chunk_docs(docs, ch["method"], ch["max_tokens"], count_tokens, ch.get("clean_latex", False))
    out_path = Path(config["paths"]["chunks_dir"]) / f"{args.collection}.jsonl"
    write_chunks(chunks, out_path)
    print(f"{len(docs)} docs -> {len(chunks)} chunks -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
