"""Validate eval question files (JSONL, one question object per line).

Usage (from the repo root):
    python -m eval.validate_questions                 # all *.jsonl in questions_dir
    python -m eval.validate_questions path/a.jsonl dir/

Exits 0 if everything is valid, 1 otherwise. Prints every error with file:line.

Each entry of source_refs is one source the answer needs: either a ref
{"doc", "page"?, "section"?} or a list of 2+ refs that are alternatives
(the same fact stated in several places; any one of them is enough).
A multi_hop question has several entries; a single-fact question has one.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

REQUIRED_KEYS = {"id", "corpus", "split", "question", "expected_answer", "source_refs", "category"}
SPLITS = {"dev", "test"}
CATEGORIES = {"single_fact", "exact", "multi_hop", "unanswerable", "adversarial"}
REF_KEYS = {"doc", "page", "section"}


def validate_question(q: object) -> list[str]:
    """Return a list of problems with one question object (empty list = valid)."""
    if not isinstance(q, dict):
        return ["line is not a JSON object"]

    errors = []
    missing = REQUIRED_KEYS - q.keys()
    unknown = q.keys() - REQUIRED_KEYS
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if unknown:
        errors.append(f"unknown keys: {sorted(unknown)}")
    if missing:
        return errors  # the checks below assume every key is present

    for key in ("id", "corpus", "question"):
        if not isinstance(q[key], str) or not q[key].strip():
            errors.append(f"'{key}' must be a non-empty string")
    if q["split"] not in SPLITS:
        errors.append(f"'split' must be one of {sorted(SPLITS)}, got {q['split']!r}")
    if q["category"] not in CATEGORIES:
        errors.append(f"'category' must be one of {sorted(CATEGORIES)}, got {q['category']!r}")

    refs = q["source_refs"]
    if not isinstance(refs, list):
        errors.append("'source_refs' must be a list")
        refs = []
    for i, entry in enumerate(refs):
        if isinstance(entry, list):  # alternatives: any one of these sources is enough
            if len(entry) < 2:
                errors.append(f"source_refs[{i}]: a list of alternatives needs at least 2 refs")
            for j, ref in enumerate(entry):
                errors += [f"source_refs[{i}][{j}]: {e}" for e in validate_ref(ref)]
        else:
            errors += [f"source_refs[{i}]: {e}" for e in validate_ref(entry)]

    if q["category"] == "unanswerable":
        if q["expected_answer"] is not None:
            errors.append("unanswerable questions must have expected_answer = null")
        if refs:
            errors.append("unanswerable questions must have source_refs = []")
    else:
        if not isinstance(q["expected_answer"], str) or not q["expected_answer"].strip():
            errors.append("'expected_answer' must be a non-empty string")
        if not refs:
            errors.append("answerable questions need at least one source_ref")

    return errors


def validate_ref(ref: object) -> list[str]:
    if not isinstance(ref, dict):
        return ["must be an object"]
    errors = []
    if ref.keys() - REF_KEYS:
        errors.append(f"unknown keys: {sorted(ref.keys() - REF_KEYS)}")
    if not isinstance(ref.get("doc"), str) or not ref["doc"].strip():
        errors.append("'doc' must be a non-empty string")
    if "page" not in ref and "section" not in ref:
        errors.append("needs 'page' and/or 'section'")
    if "page" in ref and (not isinstance(ref["page"], int) or isinstance(ref["page"], bool) or ref["page"] < 1):
        errors.append("'page' must be an integer >= 1")
    if "section" in ref and (not isinstance(ref["section"], str) or not ref["section"].strip()):
        errors.append("'section' must be a non-empty string")
    return errors


def validate_files(paths: list[Path]) -> list[str]:
    """Validate every line of every file. Also checks ids are unique across all files."""
    errors = []
    seen_ids: dict[str, str] = {}  # id -> "file:line" where first seen
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{path}:{lineno}"
                try:
                    q = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"{where}: invalid JSON ({e.msg})")
                    continue
                errors += [f"{where}: {e}" for e in validate_question(q)]
                qid = q.get("id") if isinstance(q, dict) else None
                if isinstance(qid, str):
                    if qid in seen_ids:
                        errors.append(f"{where}: duplicate id {qid!r} (first seen at {seen_ids[qid]})")
                    else:
                        seen_ids[qid] = where
    return errors


def collect_paths(args: list[str]) -> list[Path]:
    paths = []
    for arg in args:
        p = Path(arg)
        paths += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="JSONL files or directories (default: questions_dir from config)")
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()

    if not args.paths:
        config = tomllib.loads(Path(args.config).read_text())
        args.paths = [config["paths"]["questions_dir"]]
    paths = collect_paths(args.paths)
    if not paths:
        print("no .jsonl files found", file=sys.stderr)
        return 1

    errors = validate_files(paths)
    for e in errors:
        print(e)
    print(f"\nchecked {len(paths)} file(s): {'OK' if not errors else f'{len(errors)} error(s)'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
