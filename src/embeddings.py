"""Embedding service for semantic search using sentence-transformers."""

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# Default model: fast, small (22MB), 384-dim, good quality for English technical docs
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _flush_logs():
    """Flush stderr so log output is visible immediately through MCP stdio pipe."""
    sys.stderr.flush()


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
        self._device: str | None = None

    def _load_model(self):
        """Lazy-load the sentence-transformer model on first use."""
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        # Auto-detect best available device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Loading embedding model '{self.model_name}' on {device} (first use may download ~22MB)...")
        _flush_logs()
        start = time.time()
        self._model = SentenceTransformer(self.model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        self._device = device
        elapsed = time.time() - start
        logger.info(f"Embedding model loaded in {elapsed:.1f}s (device={device}, dimension={self._dimension})")
        _flush_logs()

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

    def embed_batch(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed a batch of texts efficiently.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding. Auto-selected based on device
                        if not specified (256 for GPU, 64 for CPU).

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        self._load_model()

        # Auto-select batch size based on device if not specified
        if batch_size is None:
            batch_size = 256 if self._device in ("cuda", "mps") else 64

        # Truncate long texts
        max_chars = 2048
        truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]

        total = len(truncated)
        # Process in chunks to provide visible progress.
        # tqdm progress bars use \r carriage returns that don't render through
        # MCP stdio stderr pipe, so we log periodic updates with stderr flush.
        chunk_size = batch_size * 20
        all_embeddings: list[list[float]] = []

        logger.info(f"Embedding {total} texts (batch_size={batch_size}, device={self._device})...")
        _flush_logs()
        start = time.time()

        for offset in range(0, total, chunk_size):
            chunk = truncated[offset : offset + chunk_size]
            chunk_embeddings = self._model.encode(  # type: ignore[union-attr]
                chunk, batch_size=batch_size, show_progress_bar=False
            )
            all_embeddings.extend(e.tolist() for e in chunk_embeddings)

            done = min(offset + chunk_size, total)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            pct = done * 100 // total
            if done < total:
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(f"  Progress: {done}/{total} ({pct}%, {rate:.0f} texts/s, ETA {eta:.0f}s)")
            else:
                logger.info(f"  Done: {done}/{total} ({rate:.0f} texts/s)")
            _flush_logs()

        elapsed = time.time() - start
        rate = total / elapsed if elapsed > 0 else 0
        logger.info(f"Embedded {total} texts in {elapsed:.1f}s ({rate:.0f} texts/s)")
        _flush_logs()

        return all_embeddings
