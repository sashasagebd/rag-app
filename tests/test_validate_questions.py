import json
from pathlib import Path

from eval.validate_questions import validate_files, validate_question

REPO = Path(__file__).parent.parent


def good_question(**overrides):
    q = {
        "id": "q1",
        "corpus": "example",
        "split": "dev",
        "question": "How long is leave?",
        "expected_answer": "16 weeks.",
        "source_refs": [{"doc": "handbook.pdf", "page": 3}],
        "category": "single_fact",
    }
    q.update(overrides)
    return q


def test_example_file_is_valid():
    assert validate_files([REPO / "data/questions/examples.jsonl"]) == []


def test_good_question_passes():
    assert validate_question(good_question()) == []


def test_missing_and_unknown_keys():
    q = good_question(extra=1)
    del q["split"]
    errors = validate_question(q)
    assert any("missing keys" in e for e in errors)
    assert any("unknown keys" in e for e in errors)


def test_bad_enums():
    errors = validate_question(good_question(split="train", category="trivia"))
    assert len(errors) == 2


def test_ref_needs_page_or_section():
    errors = validate_question(good_question(source_refs=[{"doc": "handbook.pdf"}]))
    assert errors == ["source_refs[0]: needs 'page' and/or 'section'"]


def test_unanswerable_rules():
    ok = good_question(category="unanswerable", expected_answer=None, source_refs=[])
    assert validate_question(ok) == []
    bad = good_question(category="unanswerable")  # still has an answer and refs
    assert len(validate_question(bad)) == 2


def test_answerable_needs_refs():
    assert validate_question(good_question(source_refs=[])) == [
        "answerable questions need at least one source_ref"
    ]


def test_duplicate_ids_and_bad_json(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(json.dumps(good_question()) + "\n" + json.dumps(good_question()) + "\n{not json\n")
    errors = validate_files([f])
    assert len(errors) == 2
    assert "duplicate id 'q1'" in errors[0] and ":2:" in errors[0]
    assert "invalid JSON" in errors[1] and ":3:" in errors[1]
