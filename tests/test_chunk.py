"""Chunker tests use a word counter instead of the real tokenizer, so they need no model download."""

from ingest.chunk import chunk_docs, pack_sections, split_to_fit


def count_words(text: str) -> int:
    return len(text.split())


def doc(*sections, source="d.html"):
    return {"source": source, "sections": [{"page": None, "heading_path": hp, "text": t} for hp, t in sections]}


def test_packs_consecutive_sections_up_to_the_limit():
    d = doc((["A"], "one two three"), (["A", "B"], "four five"), (["C"], "six seven eight nine"))
    # blocks: "## A\none two three" = 5 words, "## A > B\nfour five" = 6, "## C\n..." = 6
    chunks = pack_sections(d, count_words, max_tokens=11)
    assert [c["sections"] for c in chunks] == [["A", "A > B"], ["C"]]
    assert chunks[0]["text"] == "## A\none two three\n\n## A > B\nfour five"


def test_section_without_heading_has_no_heading_line():
    chunks = pack_sections(doc(([], "Session S01")), count_words, max_tokens=50)
    assert chunks == [{"sections": [""], "text": "Session S01"}]


def test_oversized_section_is_split_and_each_piece_keeps_the_heading():
    text = "\n".join(["w w w"] * 4)  # 12 words over 4 lines
    chunks = pack_sections(doc((["Big"], text)), count_words, max_tokens=8)
    # heading "## Big" = 2 words, leaving 6 per piece = 2 lines
    assert [c["text"] for c in chunks] == ["## Big\nw w w\nw w w"] * 2
    assert all(c["sections"] == ["Big"] for c in chunks)


def test_split_to_fit_breaks_an_overlong_line_by_words():
    assert split_to_fit("a b c d e", 2, count_words) == ["a b", "c d", "e"]


def test_chunk_docs_never_crosses_documents_and_adds_keys():
    docs = [doc((["A"], "x"), source="one.html"), doc((["B"], "y"), source="two.html")]
    chunks = chunk_docs(docs, "pack_sections", 100, count_words)
    assert [(c["chunk_key"], c["doc"], c["sections"]) for c in chunks] == [
        ("one.html#0", "one.html", ["A"]),
        ("two.html#0", "two.html", ["B"]),
    ]
    assert chunks[0]["chunker"] == "pack_sections:100"
    assert chunks[0]["n_tokens"] == 3  # "## A\nx"


def test_clean_latex_rewrites_math_spans_only():
    from ingest.chunk import clean_latex

    assert clean_latex(r"A \(1\times1\) convolution") == "A 1×1 convolution"
    assert clean_latex(r"divide by \(\sqrt{d_k}\)") == "divide by √(d_k)"
    assert clean_latex(r"\(\tfrac{\alpha}{r}BA\)") == "(α)/(r)BA"
    assert clean_latex(r"\(\operatorname{softmax}(\textcolor{#8250df}{Q}K^\top)\)") == "softmax(QKᵀ)"
    code = "d = {'a': 1}  # not math"
    assert clean_latex(code) == code


def test_clean_latex_keeps_original_section_names():
    from ingest.chunk import chunk_docs

    d = doc(([r"A \(1\times1\) conv"], r"mixes \(C\) channels"))
    [c] = chunk_docs([d], "pack_sections", 100, count_words, clean_latex_text=True)
    assert c["text"] == "## A 1×1 conv\nmixes C channels"
    assert c["sections"] == [r"A \(1\times1\) conv"]
    assert c["chunker"] == "pack_sections:100+clean_latex"
