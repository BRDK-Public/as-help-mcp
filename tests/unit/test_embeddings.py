"""Unit tests for embeddings.py - EmbeddingService."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings import DEFAULT_MODEL_NAME, EmbeddingService


class TestEmbeddingServiceInit:
    """Test EmbeddingService initialization."""

    def test_default_model_name(self):
        """Verify default model name is set correctly."""
        service = EmbeddingService()
        assert service.model_name == DEFAULT_MODEL_NAME

    def test_custom_model_name(self):
        """Verify custom model name is accepted."""
        service = EmbeddingService(model_name="custom-model")
        assert service.model_name == "custom-model"

    def test_env_var_model_name(self, monkeypatch):
        """Verify model name from environment variable."""
        monkeypatch.setenv("AS_HELP_EMBEDDING_MODEL", "env-model")
        service = EmbeddingService()
        assert service.model_name == "env-model"

    def test_explicit_overrides_env_var(self, monkeypatch):
        """Verify explicit model name overrides env var."""
        monkeypatch.setenv("AS_HELP_EMBEDDING_MODEL", "env-model")
        service = EmbeddingService(model_name="explicit-model")
        assert service.model_name == "explicit-model"

    def test_model_not_loaded_on_init(self):
        """Verify model is lazily loaded, not on __init__."""
        service = EmbeddingService()
        assert service._model is None


class TestEmbedText:
    """Test single text embedding."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked SentenceTransformer."""
        service = EmbeddingService()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.zeros(384)
        service._model = mock_model
        service._dimension = 384
        return service

    def test_embed_text_returns_list(self, mock_service):
        """Verify embed_text returns a list of floats."""
        result = mock_service.embed_text("hello world")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_text_calls_encode(self, mock_service):
        """Verify encode is called with correct args."""
        mock_service.embed_text("test input")
        mock_service._model.encode.assert_called_once_with("test input", show_progress_bar=False)

    def test_embed_text_truncates_long_input(self, mock_service):
        """Verify very long text is truncated."""
        long_text = "x" * 5000
        mock_service.embed_text(long_text)
        call_args = mock_service._model.encode.call_args[0][0]
        assert len(call_args) == 2048

    def test_embed_text_short_input_not_truncated(self, mock_service):
        """Verify short text is passed as-is."""
        short_text = "hello"
        mock_service.embed_text(short_text)
        call_args = mock_service._model.encode.call_args[0][0]
        assert call_args == "hello"


class TestEmbedBatch:
    """Test batch text embedding."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked SentenceTransformer."""
        service = EmbeddingService()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.zeros((3, 384))
        service._model = mock_model
        service._dimension = 384
        service._device = "cpu"
        return service

    def test_embed_batch_empty_list(self, mock_service):
        """Verify empty list returns empty list."""
        result = mock_service.embed_batch([])
        assert result == []

    def test_embed_batch_returns_list_of_lists(self, mock_service):
        """Verify batch returns correct structure."""
        result = mock_service.embed_batch(["a", "b", "c"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, list) for v in result)

    def test_embed_batch_truncates_long_texts(self, mock_service):
        """Verify long texts are truncated in batch."""
        texts = ["short", "x" * 5000, "also short"]
        mock_service.embed_batch(texts)
        call_args = mock_service._model.encode.call_args[0][0]
        assert len(call_args[1]) == 2048
        assert call_args[0] == "short"
        assert call_args[2] == "also short"

    def test_embed_batch_custom_batch_size(self, mock_service):
        """Verify custom batch_size is passed to encode."""
        mock_service.embed_batch(["a", "b", "c"], batch_size=32)
        mock_service._model.encode.assert_called_once()
        _, kwargs = mock_service._model.encode.call_args
        assert kwargs["batch_size"] == 32
        assert kwargs["show_progress_bar"] is False

    def test_embed_batch_uses_env_batch_size(self, mock_service, monkeypatch):
        """Verify AS_HELP_EMBED_BATCH_SIZE overrides default auto batch size."""
        monkeypatch.setenv("AS_HELP_EMBED_BATCH_SIZE", "128")

        mock_service.embed_batch(["a", "b", "c"])

        _, kwargs = mock_service._model.encode.call_args
        assert kwargs["batch_size"] == 128

    def test_embed_batch_invalid_env_batch_size_falls_back_default(self, mock_service, monkeypatch):
        """Verify invalid AS_HELP_EMBED_BATCH_SIZE falls back to device default."""
        monkeypatch.setenv("AS_HELP_EMBED_BATCH_SIZE", "not-a-number")

        mock_service.embed_batch(["a", "b", "c"])

        _, kwargs = mock_service._model.encode.call_args
        assert kwargs["batch_size"] == 64


class TestLazyLoading:
    """Test lazy model loading behavior."""

    @patch("torch.cuda.is_available", return_value=False)
    @patch("sentence_transformers.SentenceTransformer")
    def test_dimension_triggers_load(self, mock_st_class, _mock_cuda):
        """Verify accessing dimension triggers model load."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        service = EmbeddingService()
        assert service._model is None

        dim = service.dimension
        assert dim == 384
        mock_st_class.assert_called_once_with(DEFAULT_MODEL_NAME, device="cpu")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loaded_once(self, mock_st_class, _mock_cuda):
        """Verify model is only loaded once across multiple calls."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.zeros(384)
        mock_st_class.return_value = mock_model

        service = EmbeddingService()
        service.embed_text("first")
        service.embed_text("second")

        mock_st_class.assert_called_once()
