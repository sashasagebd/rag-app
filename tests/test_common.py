import os

from answer.answer import build_user_message
from common.cache import Cache
from common.config import deep_merge, load_dotenv


def test_cache_roundtrip_ignores_key_order_and_separates_namespaces(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    cache.put("llm", {"a": 1, "b": [1, 2]}, {"answer": "x"})
    assert cache.get("llm", {"b": [1, 2], "a": 1}) == {"answer": "x"}
    assert cache.get("embedding", {"a": 1, "b": [1, 2]}) is None
    assert cache.get("llm", {"a": 2, "b": [1, 2]}) is None


def test_cache_persists_across_instances(tmp_path):
    Cache(tmp_path / "c.sqlite").put("llm", "req", [1.5, 2.5])
    assert Cache(tmp_path / "c.sqlite").get("llm", "req") == [1.5, 2.5]


def test_deep_merge_keeps_untouched_keys():
    base = {"retrieval": {"k": 5, "x": 1}, "answer": {"model": "a"}}
    merged = deep_merge(base, {"retrieval": {"k": 3}})
    assert merged == {"retrieval": {"k": 3, "x": 1}, "answer": {"model": "a"}}
    assert base["retrieval"]["k"] == 5  # input not mutated


def test_dotenv_reads_values_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nRAG_TEST_A=one\nRAG_TEST_B="two"\n\nRAG_TEST_C=from-file\n')
    monkeypatch.delenv("RAG_TEST_A", raising=False)
    monkeypatch.delenv("RAG_TEST_B", raising=False)
    monkeypatch.setenv("RAG_TEST_C", "from-shell")
    load_dotenv(env)
    assert (os.environ["RAG_TEST_A"], os.environ["RAG_TEST_B"], os.environ["RAG_TEST_C"]) == (
        "one", "two", "from-shell")
    monkeypatch.delenv("RAG_TEST_A")
    monkeypatch.delenv("RAG_TEST_B")


def test_answer_prompt_wraps_each_chunk_with_its_key():
    msg = build_user_message("Why?", [{"chunk_key": "d.html#0", "text": "Because."},
                                      {"chunk_key": "d.html#1", "text": "Also."}])
    assert '<document chunk_key="d.html#0">\nBecause.\n</document>' in msg
    assert msg.endswith("Question: Why?")
    assert msg.count("<document ") == 2
