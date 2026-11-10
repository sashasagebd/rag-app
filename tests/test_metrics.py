from eval.metrics import aggregate, chunk_covers_ref, score_question

ALT = {"review.html": ["a.html"]}


def chunk(key, doc, *sections):
    return {"chunk_key": key, "doc": doc, "sections": list(sections)}


def q(category, refs):
    return {"category": category, "source_refs": refs}


def test_section_must_match_on_same_doc():
    c = chunk("a#0", "a.html", "X > Y", "X > Z")
    assert chunk_covers_ref(c, {"doc": "a.html", "section": "X > Z"}, ALT)
    assert not chunk_covers_ref(c, {"doc": "a.html", "section": "X"}, ALT)
    assert not chunk_covers_ref(c, {"doc": "b.html", "section": "X > Z"}, ALT)


def test_alternate_source_covers_any_section_of_listed_docs():
    c = chunk("r#0", "review.html", "Recap")
    assert chunk_covers_ref(c, {"doc": "a.html", "section": "Anything"}, ALT)
    assert not chunk_covers_ref(c, {"doc": "b.html", "section": "Anything"}, ALT)


def test_multi_hop_recall_and_reciprocal_rank():
    refs = [{"doc": "a.html", "section": "S1"}, {"doc": "b.html", "section": "S2"}]
    chunks = [chunk("x#0", "x.html", "N"), chunk("a#0", "a.html", "S1"), chunk("y#0", "y.html", "N")]
    m = score_question(q("multi_hop", refs), chunks, {"found": True, "answer": "...", "citations": ["a#0"]},
                       "partial", ALT)
    assert m["hit"] and m["recall"] == 0.5 and m["rr"] == 0.5
    assert m["verdict"] == "partial"
    assert m["citations_valid"] and m["citation_supports"]


def test_abstaining_on_answerable_counts_as_incorrect():
    refs = [{"doc": "a.html", "section": "S1"}]
    m = score_question(q("single_fact", refs), [chunk("a#0", "a.html", "S1")],
                       {"found": False, "answer": "Not found", "citations": []}, None, ALT)
    assert m["abstained"] and m["verdict"] == "incorrect"
    assert "citations_valid" not in m


def test_citation_to_unretrieved_chunk_is_invalid():
    refs = [{"doc": "a.html", "section": "S1"}]
    m = score_question(q("exact", refs), [chunk("a#0", "a.html", "S1")],
                       {"found": True, "answer": "...", "citations": ["made-up#9"]}, "correct", ALT)
    assert not m["citations_valid"] and not m["citation_supports"]


def test_aggregate_separates_answerable_and_unanswerable():
    records = [
        {"category": "exact", "metrics": {"abstained": False, "hit": True, "recall": 1.0, "rr": 1.0,
                                          "verdict": "correct", "citations_valid": True, "citation_supports": True}},
        {"category": "exact", "metrics": {"abstained": True, "hit": False, "recall": 0.0, "rr": 0.0,
                                          "verdict": "incorrect"}},
        {"category": "unanswerable", "metrics": {"abstained": True}},
        {"category": "unanswerable", "metrics": {"abstained": False, "citations_valid": True}},
    ]
    agg = aggregate(records)
    assert (agg["n_answerable"], agg["n_unanswerable"]) == (2, 2)
    assert agg["hit_at_k"] == 0.5 and agg["correct"] == 0.5 and agg["false_abstention"] == 0.5
    assert agg["correct_abstention"] == 0.5
    assert agg["citation_supports"] == 1.0  # only over answered answerable questions


def test_alternatives_count_as_one_needed_source():
    alt = [{"doc": "a.html", "section": "S1"}, {"doc": "b.html", "section": "S9"}]
    second = {"doc": "c.html", "section": "S2"}
    chunks = [chunk("b#0", "b.html", "S9")]
    m = score_question(q("multi_hop", [alt, second]), chunks,
                       {"found": True, "answer": "...", "citations": ["b#0"]}, "partial", ALT)
    assert m["recall"] == 0.5  # the alternative group is covered, the second source is not
    assert m["citation_supports"]
    single = score_question(q("exact", [alt]), chunks, {"found": True, "answer": "...", "citations": []}, "correct", ALT)
    assert single["recall"] == 1.0  # one alternative found = fully covered
