"""Local embeddings with sentence-transformers, cached per text.

    vectors = embed(["some text", ...], "BAAI/bge-small-en-v1.5", cache=cache)

Returns an (n, dim) float32 numpy array, L2-normalized (so cosine = dot product).
The model is loaded once per process. With a cache, only texts never seen
before are embedded; re-ingesting unchanged chunks is nearly instant.

bge models want a prefix on QUERIES (not on passages); see QUERY_PREFIX.

MODEL_LOCK serializes every model load (embedder, tokenizer, reranker) across
threads: the transformers library imports lazily and is not thread-safe while
importing, so two web requests loading models at once could crash one of them.
"""

import threading

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_models: dict = {}
MODEL_LOCK = threading.RLock()


def get_model(name: str):
    with MODEL_LOCK:
        if name not in _models:
            from sentence_transformers import SentenceTransformer  # slow import; only when needed

            _models[name] = SentenceTransformer(name)
        return _models[name]


def embed(texts: list[str], model_name: str, batch_size: int = 32, cache=None, progress: bool = False) -> np.ndarray:
    vectors: list = [None] * len(texts)
    missing = []
    for i, text in enumerate(texts):
        hit = cache.get("embedding", [model_name, text]) if cache else None
        if hit is None:
            missing.append(i)
        else:
            vectors[i] = hit

    if missing:
        computed = get_model(model_name).encode(
            [texts[i] for i in missing], batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=progress,
        )
        for i, vec in zip(missing, computed):
            vectors[i] = vec.tolist()
            if cache:
                cache.put("embedding", [model_name, texts[i]], vectors[i])

    return np.array(vectors, dtype=np.float32)
