"""One function for every LLM call: send a prompt, get JSON back that matches a schema.

    result = complete_json(model=..., system=..., user=..., schema=..., effort="low",
                           max_tokens=4000, cache=cache)
    result["data"]   # the parsed JSON object
    result["usage"]  # {"input_tokens": ..., "output_tokens": ...}
    result["cached"] # True if served from the local cache (cost 0)

Uses the Anthropic API with structured outputs (output_config.format), so the
reply is guaranteed to be valid JSON for the schema. Sampling settings such as
temperature are not sent: current models reject them. Repeatability comes from
the cache instead: the same request always returns the same stored answer.

Check your API key (one tiny Haiku call):
    python -m common.llm
"""

import json
import sys

import anthropic

from common.config import load_dotenv

_client = None


class LLMError(RuntimeError):
    pass


def client(api_key: str | None = None) -> anthropic.Anthropic:
    """The server's client (ANTHROPIC_API_KEY from .env), or a client for a user's own key."""
    global _client
    if api_key:
        return anthropic.Anthropic(api_key=api_key, max_retries=1)
    if _client is None:
        load_dotenv()
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def complete_json(*, model: str, system: str, user: str, schema: dict, effort: str,
                  max_tokens: int, cache=None, api_key: str | None = None) -> dict:
    """api_key: bill this call to that key instead of the server's. It is never part of the
    cache key, so a cached answer is shared only between identical requests (same documents)."""
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    }
    if cache:
        hit = cache.get("llm", request)
        if hit is not None:
            return {**hit, "cached": True}

    response = client(api_key).messages.create(**request)
    if response.stop_reason == "refusal":
        raise LLMError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise LLMError(f"hit max_tokens={max_tokens}; raise it in the config")
    text = next(b.text for b in response.content if b.type == "text")

    result = {
        "data": json.loads(text),
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }
    if cache:
        cache.put("llm", request, result)
    return {**result, "cached": False}


def cost_usd(usage: dict, model: str, pricing: dict) -> float:
    """Dollar cost of one call, from per-million-token prices in the config's [pricing] table."""
    p = pricing[model]
    return (usage["input_tokens"] * p["input"] + usage["output_tokens"] * p["output"]) / 1_000_000


def main() -> int:
    try:
        result = complete_json(
            model="claude-haiku-5-5",
            system="You are a connectivity check.",
            user='Reply with {"ok": true}.',
            schema={"type": "object", "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"], "additionalProperties": False},
            effort="low", max_tokens=1000,
        )
    except anthropic.AuthenticationError:
        print("API key rejected: check ANTHROPIC_API_KEY in .env", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("could not reach the API: check your network", file=sys.stderr)
        return 1
    print(f"API key works. Reply: {result['data']}, tokens: {result['usage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
