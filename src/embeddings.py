"""Embedding service using an OpenAI-compatible embedding API.

Supports any OpenAI-compatible endpoint: OpenAI, Azure OpenAI, GitHub Models,
Ollama, LiteLLM, etc.  Only used when CREATE_EMBEDDINGS=true.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)


class EmbeddingTooLargeError(Exception):
    """Raised when the input exceeds the model's context length."""

# Defaults
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_CHARS = 8000
DEFAULT_TIMEOUT = 60  # seconds per API call


class EmbeddingService:
    """Calls an OpenAI-compatible ``/embeddings`` endpoint.

    All configuration is read from environment variables (or constructor args):

    * ``EMBEDDING_API_ENDPOINT`` - base URL, e.g. ``https://models.inference.ai.azure.com``
    * ``EMBEDDING_API_KEY``      - bearer token / API key
    * ``EMBEDDING_MODEL``        - model name sent in the request body
    * ``EMBEDDING_DIMENSIONS``   - expected vector dimensionality (required)
    * ``EMBEDDING_BATCH_SIZE``   - texts per API call (default 100)
    * ``EMBEDDING_MAX_CHARS``    - truncate input texts to this length (default 8000)
    """

    def __init__(
        self,
        *,
        api_endpoint: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        max_chars: int | None = None,
    ):
        self.api_endpoint = (api_endpoint or os.getenv("EMBEDDING_API_ENDPOINT", "")).rstrip("/")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "")

        dim_str = os.getenv("EMBEDDING_DIMENSIONS", "")
        self._dimension = dimensions or (int(dim_str) if dim_str.strip() else 0)

        bs_str = os.getenv("EMBEDDING_BATCH_SIZE", "")
        self.batch_size = batch_size or (int(bs_str) if bs_str.strip() else DEFAULT_BATCH_SIZE)

        mc_str = os.getenv("EMBEDDING_MAX_CHARS", "")
        self.max_chars = max_chars or (int(mc_str) if mc_str.strip() else DEFAULT_MAX_CHARS)

        # Validate required fields
        if not self.api_endpoint:
            raise ValueError("EMBEDDING_API_ENDPOINT is required when embeddings are enabled")
        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY is required when embeddings are enabled")
        if not self.model_name:
            raise ValueError("EMBEDDING_MODEL is required when embeddings are enabled")
        if self._dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be a positive integer")

        # Construct the embeddings URL
        # If the endpoint already ends with /embeddings, use as-is
        if self.api_endpoint.endswith("/embeddings"):
            self._url = self.api_endpoint
        else:
            # Strip trailing /v1 if present, then add /v1/embeddings
            base = self.api_endpoint.rstrip("/")
            if base.endswith("/v1"):
                self._url = f"{base}/embeddings"
            else:
                self._url = f"{base}/v1/embeddings"

        self._client = httpx.Client(
            timeout=DEFAULT_TIMEOUT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "EmbeddingService configured: endpoint=%s model=%s dim=%d batch=%d",
            self.api_endpoint,
            self.model_name,
            self._dimension,
            self.batch_size,
        )

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string via the API.

        Returns:
            List of floats (embedding vector).
        """
        if not text.strip():
            return [0.0] * self._dimension

        truncated = text[: self.max_chars] if len(text) > self.max_chars else text
        result = self._call_api([truncated])
        return result[0]

    def embed_batch(self, texts: list[str], batch_size: int | None = None, *, show_progress: bool = True) -> list[list[float]]:
        """Embed a batch of texts via chunked API calls.

        Args:
            texts: List of texts to embed.
            batch_size: Override instance batch_size for this call.
            show_progress: Log per-batch progress (disable when caller tracks progress).

        Returns:
            List of embedding vectors in the same order as input.
        """
        if not texts:
            return []

        bs = batch_size or self.batch_size
        total = len(texts)
        all_embeddings: list[list[float]] = []

        # Truncate
        truncated = [t[: self.max_chars] if len(t) > self.max_chars else t for t in texts]

        if show_progress:
            logger.info("Embedding %d texts (batch_size=%d, model=%s)...", total, bs, self.model_name)
        start = time.time()

        for offset in range(0, total, bs):
            chunk = truncated[offset : offset + bs]
            # Replace empty strings with a single space (API rejects empty input)
            chunk = [t if t.strip() else " " for t in chunk]

            try:
                chunk_vectors = self._call_api(chunk)
            except EmbeddingTooLargeError:
                # At least one text in this batch exceeds the model context.
                # Fall back to embedding each text individually.
                logger.warning(
                    "Batch at offset %d failed (input too large), falling back to one-by-one",
                    offset,
                )
                chunk_vectors = self._embed_chunk_individually(chunk)
            all_embeddings.extend(chunk_vectors)

            if show_progress:
                done = min(offset + bs, total)
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                if done < total:
                    eta = (total - done) / rate if rate > 0 else 0
                    logger.info(
                        "  Progress: %d/%d (%d%%, %.0f texts/s, ETA %.0fs)",
                        done, total, done * 100 // total, rate, eta,
                    )
                else:
                    logger.info("  Done: %d/%d (%.0f texts/s)", done, total, rate)

        elapsed = time.time() - start
        if show_progress:
            logger.info("Embedded %d texts in %.1fs", total, elapsed)
        return all_embeddings

    def _embed_chunk_individually(self, chunk: list[str]) -> list[list[float]]:
        """Embed texts one-by-one, returning zero vectors for any that fail."""
        vectors: list[list[float]] = []
        zero = [0.0] * self._dimension
        failed = 0
        for text in chunk:
            try:
                vecs = self._call_api([text])
                vectors.append(vecs[0])
            except (EmbeddingTooLargeError, httpx.HTTPStatusError):
                vectors.append(zero)
                failed += 1
        if failed:
            logger.warning(
                "  %d/%d texts in chunk exceeded model context -- used zero vectors",
                failed, len(chunk),
            )
        return vectors

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Make a single API call with retry on transient errors.

        Retries up to 3 times on 429 (rate limit) and 5xx (server error)
        with exponential backoff.  Raises ``EmbeddingTooLargeError`` on 400
        when the error message indicates the input exceeds the model context.
        """
        payload: dict = {
            "input": texts,
            "model": self.model_name,
        }
        # Include dimensions if the API supports it (OpenAI, GitHub Models)
        if self._dimension:
            payload["dimensions"] = self._dimension

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self._client.post(self._url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    # Sort by index to guarantee order
                    embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                    return [item["embedding"] for item in embeddings_data]

                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Embedding API returned %d (attempt %d/%d), retrying in %ds...",
                        response.status_code,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                # Non-retryable error — log the response body for diagnosis
                try:
                    body = response.text[:500]
                except Exception:
                    body = "(could not read response body)"

                # Check if this is a context-length error (Ollama / vLLM pattern)
                if response.status_code == 400 and "context length" in body.lower():
                    logger.warning(
                        "Embedding API: input too large for %d texts (%d chars max): %s",
                        len(texts),
                        max(len(t) for t in texts) if texts else 0,
                        body,
                    )
                    raise EmbeddingTooLargeError(body)

                logger.error(
                    "Embedding API error %d for %d texts: %s",
                    response.status_code, len(texts), body,
                )
                response.raise_for_status()

            except httpx.TimeoutException:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Embedding API timeout (attempt %d/%d), retrying in %ds...",
                        attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                raise

        # Should not reach here, but just in case
        raise RuntimeError(f"Embedding API failed after {max_retries} retries")

    def close(self):
        """Close the HTTP client."""
        self._client.close()
