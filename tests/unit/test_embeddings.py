"""Unit tests for embeddings.py - API-based EmbeddingService."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.embeddings import EmbeddingService, EmbeddingTooLargeError


def _make_embedding_response(vectors: list[list[float]]) -> dict:
    """Build a mock OpenAI embedding API response."""
    return {
        "data": [{"index": i, "embedding": vec} for i, vec in enumerate(vectors)],
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


class TestEmbeddingServiceInit:
    """Test EmbeddingService initialization and validation."""

    def test_constructor_with_all_args(self):
        """Verify constructor accepts all explicit arguments."""
        service = EmbeddingService(
            api_endpoint="https://api.example.com",
            api_key="sk-test",
            model_name="text-embedding-3-small",
            dimensions=1536,
            batch_size=50,
            max_chars=4000,
        )
        assert service.api_endpoint == "https://api.example.com"
        assert service.model_name == "text-embedding-3-small"
        assert service.dimension == 1536
        assert service.batch_size == 50
        assert service.max_chars == 4000
        service.close()

    def test_constructor_from_env_vars(self, monkeypatch):
        """Verify constructor reads from environment variables."""
        monkeypatch.setenv("EMBEDDING_API_ENDPOINT", "https://env.example.com")
        monkeypatch.setenv("EMBEDDING_API_KEY", "env-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "200")
        monkeypatch.setenv("EMBEDDING_MAX_CHARS", "5000")

        service = EmbeddingService()
        assert service.api_endpoint == "https://env.example.com"
        assert service.api_key == "env-key"
        assert service.model_name == "env-model"
        assert service.dimension == 768
        assert service.batch_size == 200
        assert service.max_chars == 5000
        service.close()

    def test_explicit_args_override_env_vars(self, monkeypatch):
        """Verify explicit args take precedence over env vars."""
        monkeypatch.setenv("EMBEDDING_API_ENDPOINT", "https://env.example.com")
        monkeypatch.setenv("EMBEDDING_API_KEY", "env-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")

        service = EmbeddingService(
            api_endpoint="https://explicit.example.com",
            api_key="explicit-key",
            model_name="explicit-model",
            dimensions=1536,
        )
        assert service.api_endpoint == "https://explicit.example.com"
        assert service.api_key == "explicit-key"
        assert service.model_name == "explicit-model"
        assert service.dimension == 1536
        service.close()

    def test_missing_endpoint_raises(self, monkeypatch):
        """Verify missing endpoint raises ValueError."""
        monkeypatch.delenv("EMBEDDING_API_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="EMBEDDING_API_ENDPOINT"):
            EmbeddingService(api_key="key", model_name="model", dimensions=384)

    def test_missing_api_key_raises(self, monkeypatch):
        """Verify missing API key raises ValueError."""
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
            EmbeddingService(api_endpoint="https://api.test", model_name="model", dimensions=384)

    def test_missing_model_raises(self, monkeypatch):
        """Verify missing model name raises ValueError."""
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
            EmbeddingService(api_endpoint="https://api.test", api_key="key", dimensions=384)

    def test_missing_dimensions_raises(self, monkeypatch):
        """Verify missing dimensions raises ValueError."""
        monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
        with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
            EmbeddingService(api_endpoint="https://api.test", api_key="key", model_name="model")

    def test_zero_dimensions_raises(self):
        """Verify zero dimensions raises ValueError."""
        with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
            EmbeddingService(api_endpoint="https://api.test", api_key="key", model_name="model", dimensions=0)


class TestURLConstruction:
    """Test embedding API URL construction."""

    def _make(self, endpoint: str) -> EmbeddingService:
        return EmbeddingService(api_endpoint=endpoint, api_key="key", model_name="m", dimensions=8)

    def test_plain_base_url(self):
        """Verify /v1/embeddings is appended to plain base URL."""
        s = self._make("https://api.openai.com")
        assert s._url == "https://api.openai.com/v1/embeddings"
        s.close()

    def test_base_url_with_v1(self):
        """Verify /embeddings appended when /v1 already present."""
        s = self._make("https://api.openai.com/v1")
        assert s._url == "https://api.openai.com/v1/embeddings"
        s.close()

    def test_base_url_with_full_path(self):
        """Verify URL used as-is when it ends with /embeddings."""
        s = self._make("https://api.openai.com/v1/embeddings")
        assert s._url == "https://api.openai.com/v1/embeddings"
        s.close()

    def test_trailing_slash_stripped(self):
        """Verify trailing slash is handled."""
        s = self._make("https://api.openai.com/")
        assert s._url == "https://api.openai.com/v1/embeddings"
        s.close()


class TestEmbedText:
    """Test single text embedding via API."""

    @pytest.fixture
    def service(self):
        s = EmbeddingService(api_endpoint="https://api.test", api_key="key", model_name="model", dimensions=4)
        yield s
        s.close()

    def test_embed_text_returns_vector(self, service):
        """Verify embed_text returns a list of floats with correct dimension."""
        mock_response = httpx.Response(
            200,
            json=_make_embedding_response([[0.1, 0.2, 0.3, 0.4]]),
        )
        service._client = MagicMock()
        service._client.post.return_value = mock_response

        result = service.embed_text("hello")
        assert result == [0.1, 0.2, 0.3, 0.4]

    def test_embed_empty_text_returns_zeros(self, service):
        """Verify empty text returns zero vector without API call."""
        service._client = MagicMock()
        result = service.embed_text("")
        assert result == [0.0, 0.0, 0.0, 0.0]
        service._client.post.assert_not_called()

    def test_embed_text_truncates_long_input(self, service):
        """Verify long text is truncated to max_chars."""
        service.max_chars = 100
        mock_response = httpx.Response(
            200,
            json=_make_embedding_response([[1.0, 2.0, 3.0, 4.0]]),
        )
        service._client = MagicMock()
        service._client.post.return_value = mock_response

        long_text = "x" * 500
        service.embed_text(long_text)

        call_payload = service._client.post.call_args[1]["json"]
        assert len(call_payload["input"][0]) == 100


class TestEmbedBatch:
    """Test batch embedding via API."""

    @pytest.fixture
    def service(self):
        s = EmbeddingService(
            api_endpoint="https://api.test",
            api_key="key",
            model_name="model",
            dimensions=4,
            batch_size=2,
        )
        yield s
        s.close()

    def test_embed_batch_empty(self, service):
        """Verify empty list returns empty list."""
        assert service.embed_batch([]) == []

    def test_embed_batch_single_chunk(self, service):
        """Verify batch within batch_size makes one API call."""
        mock_response = httpx.Response(
            200,
            json=_make_embedding_response([[1, 2, 3, 4], [5, 6, 7, 8]]),
        )
        service._client = MagicMock()
        service._client.post.return_value = mock_response

        result = service.embed_batch(["a", "b"])
        assert len(result) == 2
        assert service._client.post.call_count == 1

    def test_embed_batch_multiple_chunks(self, service):
        """Verify batch exceeding batch_size makes multiple API calls."""
        mock_response = httpx.Response(
            200,
            json=_make_embedding_response([[1, 2, 3, 4], [5, 6, 7, 8]]),
        )
        service._client = MagicMock()
        service._client.post.return_value = mock_response

        result = service.embed_batch(["a", "b", "c", "d"])
        assert len(result) == 4
        assert service._client.post.call_count == 2

    def test_embed_batch_replaces_empty_strings(self, service):
        """Verify empty strings in batch are replaced with space."""
        mock_response = httpx.Response(
            200,
            json=_make_embedding_response([[1, 2, 3, 4], [5, 6, 7, 8]]),
        )
        service._client = MagicMock()
        service._client.post.return_value = mock_response

        service.embed_batch(["hello", ""])

        call_payload = service._client.post.call_args[1]["json"]
        assert call_payload["input"] == ["hello", " "]


class TestAPIRetry:
    """Test retry logic for transient API errors."""

    @pytest.fixture
    def service(self):
        s = EmbeddingService(api_endpoint="https://api.test", api_key="key", model_name="model", dimensions=4)
        yield s
        s.close()

    def test_retry_on_429(self, service):
        """Verify retry on rate limit (429)."""
        rate_limit_response = httpx.Response(429, json={"error": "rate limited"})
        success_response = httpx.Response(200, json=_make_embedding_response([[1, 2, 3, 4]]))

        service._client = MagicMock()
        service._client.post.side_effect = [rate_limit_response, success_response]

        with patch("src.embeddings.time.sleep"):
            result = service._call_api(["test"])

        assert result == [[1, 2, 3, 4]]
        assert service._client.post.call_count == 2

    def test_retry_on_500(self, service):
        """Verify retry on server error (500)."""
        error_response = httpx.Response(500, json={"error": "internal"})
        success_response = httpx.Response(200, json=_make_embedding_response([[1, 2, 3, 4]]))

        service._client = MagicMock()
        service._client.post.side_effect = [error_response, success_response]

        with patch("src.embeddings.time.sleep"):
            result = service._call_api(["test"])

        assert result == [[1, 2, 3, 4]]

    def test_retry_on_timeout(self, service):
        """Verify retry on timeout."""
        success_response = httpx.Response(200, json=_make_embedding_response([[1, 2, 3, 4]]))

        service._client = MagicMock()
        service._client.post.side_effect = [
            httpx.TimeoutException("timeout"),
            success_response,
        ]

        with patch("src.embeddings.time.sleep"):
            result = service._call_api(["test"])

        assert result == [[1, 2, 3, 4]]

    def test_400_context_length_raises_too_large(self, service):
        """Verify 400 with 'context length' raises EmbeddingTooLargeError."""
        error_body = '{"error":{"message":"the input length exceeds the context length"}}'
        error_response = httpx.Response(400, text=error_body)
        error_response.request = httpx.Request("POST", "https://api.test/v1/embeddings")

        service._client = MagicMock()
        service._client.post.return_value = error_response

        with pytest.raises(EmbeddingTooLargeError):
            service._call_api(["test"])

        assert service._client.post.call_count == 1

    def test_400_generic_raises_http_error(self, service):
        """Verify 400 without context-length message raises HTTPStatusError."""
        error_response = httpx.Response(400, json={"error": "bad request"})
        error_response.request = httpx.Request("POST", "https://api.test/v1/embeddings")

        service._client = MagicMock()
        service._client.post.return_value = error_response

        with pytest.raises(httpx.HTTPStatusError):
            service._call_api(["test"])

        assert service._client.post.call_count == 1

    def test_exhausted_retries_raises(self, service):
        """Verify error raised after all retries exhausted."""
        error_response = httpx.Response(500, json={"error": "internal"})
        error_response.request = httpx.Request("POST", "https://api.test/v1/embeddings")

        service._client = MagicMock()
        service._client.post.return_value = error_response

        with patch("src.embeddings.time.sleep"):
            with pytest.raises(httpx.HTTPStatusError):
                service._call_api(["test"])

        assert service._client.post.call_count == 4  # initial + 3 retries


class TestBatchFallback:
    """Test fallback to one-by-one embedding when a batch exceeds context length."""

    @pytest.fixture
    def service(self):
        s = EmbeddingService(
            api_endpoint="https://api.test",
            api_key="key",
            model_name="model",
            dimensions=4,
            batch_size=3,
        )
        yield s
        s.close()

    def test_batch_fallback_on_context_length_error(self, service):
        """Verify batch falls back to binary-split when API returns context-length 400."""
        context_error = httpx.Response(
            400,
            text='{"error":{"message":"the input length exceeds the context length"}}',
        )
        context_error.request = httpx.Request("POST", "https://api.test/v1/embeddings")

        # Batch of 3 fails -> split [a] + [b,c]
        # [a] succeeds, [b,c] succeeds
        ok_a = httpx.Response(200, json=_make_embedding_response([[1, 2, 3, 4]]))
        ok_bc = httpx.Response(200, json=_make_embedding_response([[5, 6, 7, 8], [9, 10, 11, 12]]))

        service._client = MagicMock()
        service._client.post.side_effect = [context_error, ok_a, ok_bc]

        result = service.embed_batch(["a", "b", "c"])
        assert result == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        assert service._client.post.call_count == 3

    def test_fallback_uses_zero_vector_for_individual_failure(self, service):
        """Verify individual texts that still fail get zero vectors via binary split."""
        context_error = httpx.Response(
            400,
            text='{"error":{"message":"the input length exceeds the context length"}}',
        )
        context_error.request = httpx.Request("POST", "https://api.test/v1/embeddings")

        ok_a = httpx.Response(200, json=_make_embedding_response([[1, 2, 3, 4]]))
        ok_c = httpx.Response(200, json=_make_embedding_response([[9, 10, 11, 12]]))

        service._client = MagicMock()
        # Batch [a, long, c] fails -> split [a] + [long, c]
        # [a] ok, [long, c] fails -> split [long] + [c]
        # [long] fails (zero vec), [c] ok
        service._client.post.side_effect = [context_error, ok_a, context_error, context_error, ok_c]

        result = service.embed_batch(["a", "long_text", "c"])
        assert result[0] == [1, 2, 3, 4]
        assert result[1] == [0.0, 0.0, 0.0, 0.0]  # zero vector
        assert result[2] == [9, 10, 11, 12]

    def test_fallback_all_fail_returns_all_zeros(self, service):
        """Verify all texts failing individually returns all zero vectors."""
        context_error = httpx.Response(
            400,
            text='{"error":{"message":"the input length exceeds the context length"}}',
        )
        context_error.request = httpx.Request("POST", "https://api.test/v1/embeddings")

        service._client = MagicMock()
        service._client.post.return_value = context_error

        result = service.embed_batch(["a", "b"])
        assert result == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


class TestResponseOrdering:
    """Test that embeddings are returned in correct order."""

    @pytest.fixture
    def service(self):
        s = EmbeddingService(api_endpoint="https://api.test", api_key="key", model_name="model", dimensions=4)
        yield s
        s.close()

    def test_out_of_order_response_is_sorted(self, service):
        """Verify embeddings are sorted by index even if API returns them out of order."""
        out_of_order_response = {
            "data": [
                {"index": 1, "embedding": [5, 6, 7, 8]},
                {"index": 0, "embedding": [1, 2, 3, 4]},
            ],
            "model": "test",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_response = httpx.Response(200, json=out_of_order_response)
        service._client = MagicMock()
        service._client.post.return_value = mock_response

        result = service._call_api(["first", "second"])
        assert result == [[1, 2, 3, 4], [5, 6, 7, 8]]
