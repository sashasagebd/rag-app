"""LLM judge: grade a model answer against the expected answer.

    result = judge(question, expected, answer, category, model=..., effort=..., max_tokens=..., cache=...)
    result["data"] == {"reason": "...", "verdict": "correct" | "partial" | "incorrect"}

Only answerable questions are judged. Unanswerable ones are scored from the
answer's "found" flag, with no LLM call. "reason" comes before "verdict" in the
schema so the judge explains first and decides second.
"""

from common.llm import complete_json

SYSTEM = """You grade answers to questions about course slides. You are given the question, a reference answer written by the course's student, and a model's answer.

Verdicts:
- "correct": the model answer contains the key facts of the reference answer and contradicts none of them. Extra correct detail, different wording, or a shorter phrasing of the same facts is fine.
- "partial": some key facts are present, but at least one is missing or wrong.
- "incorrect": the key facts are missing or wrong, or the model said it could not find the answer.

For category "adversarial", the question contains a false premise. The answer is "correct" only if it rejects or corrects the premise as the reference answer does; going along with the premise is "incorrect".

Judge only against the reference answer. Do not use your own knowledge of the topic. Give a one-sentence reason, then the verdict."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}


def build_user_message(question: str, expected: str, answer: str, category: str) -> str:
    return (
        f"Category: {category}\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<reference_answer>\n{expected}\n</reference_answer>\n\n"
        f"<model_answer>\n{answer}\n</model_answer>"
    )


def judge(question: str, expected: str, answer: str, category: str, *, model: str, effort: str,
          max_tokens: int, cache=None) -> dict:
    return complete_json(
        model=model, system=SYSTEM, user=build_user_message(question, expected, answer, category),
        schema=SCHEMA, effort=effort, max_tokens=max_tokens, cache=cache,
    )
