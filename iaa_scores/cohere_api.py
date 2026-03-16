"""Strict Cohere Embed API client for semantic pair matching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"
DEFAULT_MODEL = "embed-v4.0"
DEFAULT_INPUT_TYPE = "classification"
DEFAULT_OUTPUT_DIMENSION = 512
MAX_TEXTS_PER_CALL = 96


def load_repo_cohere_api_key(repo_root: Path) -> str:
    """Read COHERE_API_KEY from repo_root/.env only."""

    try:
        from dotenv import dotenv_values
    except ImportError as exc:  # pragma: no cover - dependency failure by design
        raise ImportError(
            "python-dotenv is required to load COHERE_API_KEY from repo_root/.env."
        ) from exc

    env_path = repo_root / ".env"
    if not env_path.exists() or not env_path.is_file():
        raise FileNotFoundError(f"Missing required .env file at {env_path}")

    values = dotenv_values(env_path)
    api_key = values.get("COHERE_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(
            f"COHERE_API_KEY is missing or empty in {env_path}. "
            "System environment variables are intentionally not used."
        )
    return api_key.strip()


def embed_texts(
    texts: Sequence[str],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    input_type: str = DEFAULT_INPUT_TYPE,
    output_dimension: int = DEFAULT_OUTPUT_DIMENSION,
    batch_size: int = MAX_TEXTS_PER_CALL,
    timeout_seconds: int = 120,
) -> List[List[float]]:
    """Embed texts with Cohere v2 /embed and fail loudly on API inconsistencies."""

    _validate_embed_config(
        texts=texts,
        api_key=api_key,
        model=model,
        input_type=input_type,
        output_dimension=output_dimension,
        batch_size=batch_size,
    )

    embeddings: List[List[float]] = []
    for batch_start in range(0, len(texts), batch_size):
        batch = list(texts[batch_start : batch_start + batch_size])
        embeddings.extend(
            _embed_batch(
                batch,
                api_key=api_key,
                model=model,
                input_type=input_type,
                output_dimension=output_dimension,
                timeout_seconds=timeout_seconds,
            )
        )
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Cohere embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
        )
    return embeddings


def _embed_batch(
    texts: Sequence[str],
    *,
    api_key: str,
    model: str,
    input_type: str,
    output_dimension: int,
    timeout_seconds: int,
) -> List[List[float]]:
    payload = {
        "model": model,
        "texts": list(texts),
        "input_type": input_type,
        "output_dimension": output_dimension,
        "embedding_types": ["float"],
        "truncate": "NONE",
    }
    request = Request(
        COHERE_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Cohere embed request failed with HTTP {exc.code}: {message}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Cohere embed request failed: {exc}") from exc

    parsed = json.loads(body)
    embeddings = (((parsed.get("embeddings") or {}).get("float")) or [])
    if not isinstance(embeddings, list):
        raise RuntimeError("Cohere response did not contain embeddings.float as a list.")
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Cohere returned {len(embeddings)} embeddings for {len(texts)} texts."
        )

    normalized: List[List[float]] = []
    for index, vector in enumerate(embeddings):
        if not isinstance(vector, list):
            raise RuntimeError(f"Cohere embedding at index {index} is not a list.")
        if len(vector) != output_dimension:
            raise RuntimeError(
                f"Cohere embedding at index {index} has dimension {len(vector)}; expected {output_dimension}."
            )
        normalized.append([float(value) for value in vector])
    return normalized


def _validate_embed_config(
    *,
    texts: Sequence[str],
    api_key: str,
    model: str,
    input_type: str,
    output_dimension: int,
    batch_size: int,
) -> None:
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string.")
    if not texts:
        raise ValueError("texts must be a non-empty sequence.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("All texts must be non-empty strings.")
    if model != "embed-v4.0":
        raise ValueError(f"Unsupported Cohere embedding model: {model}")
    if input_type not in {"search_document", "search_query", "classification", "clustering"}:
        raise ValueError(f"Unsupported Cohere input_type: {input_type}")
    if output_dimension not in {256, 512, 1024, 1536}:
        raise ValueError(
            f"Unsupported Cohere output_dimension {output_dimension}; expected one of 256, 512, 1024, 1536."
        )
    if batch_size <= 0 or batch_size > MAX_TEXTS_PER_CALL:
        raise ValueError(
            f"batch_size must be between 1 and {MAX_TEXTS_PER_CALL}, got {batch_size}."
        )
