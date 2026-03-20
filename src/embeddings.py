"""Embedding service for semantic search using sentence-transformers."""

import logging
import os
import time

logger = logging.getLogger(__name__)

# Default model: fast, small (22MB), 384-dim, good quality for English technical docs
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Manages sentence-transformer model for generating text embeddings."""

    def __init__(self, model_name: str | None = None):
        """Initialize embedding service and load model.

        Args:
            model_name: HuggingFace model name. Defaults to AS_HELP_EMBEDDING_MODEL
                        env var or 'all-MiniLM-L6-v2'.
        """
        self.model_name = model_name or os.getenv("AS_HELP_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
        self._model = None
        self._dimension: int | None = None

    def _load_model(self):
        """Lazy-load the sentence-transformer model on first use."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model '{self.model_name}' (first use may download ~22MB)...")
        start = time.time()
        self._model = SentenceTransformer(self.model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        elapsed = time.time() - start
        logger.info(f"Embedding model loaded in {elapsed:.1f}s (dimension={self._dimension})")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the loaded model."""
        self._load_model()
        return self._dimension  # type: ignore[return-value]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed. Empty string produces a zero-ish vector.

        Returns:
            List of floats (embedding vector).
        """
        self._load_model()
        # Truncate to model's max sequence length (handled internally by sentence-transformers,
        # but we trim obviously huge texts to save tokenizer overhead)
        max_chars = 2048
        if len(text) > max_chars:
            text = text[:max_chars]
        embedding = self._model.encode(text, show_progress_bar=False)  # type: ignore[union-attr]
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a batch of texts efficiently.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding (adjust based on GPU/CPU memory).
                        Default 64 is tuned for CPU; increase for GPU.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        self._load_model()

        # Truncate long texts
        max_chars = 2048
        truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]

        total = len(truncated)
        logger.info(f"Embedding {total} texts (batch_size={batch_size})...")
        start = time.time()

        # show_progress_bar=True outputs tqdm to stderr, which is safe
        # since MCP stdio transport uses stdout for JSON-RPC.
        embeddings = self._model.encode(  # type: ignore[union-attr]
            truncated, batch_size=batch_size, show_progress_bar=True
        )

        elapsed = time.time() - start
        rate = total / elapsed if elapsed > 0 else 0
        logger.info(f"Embedded {total} texts in {elapsed:.1f}s ({rate:.0f} texts/s)")

        return [e.tolist() for e in embeddings]
