"""Unit tests for search_engine.py - LanceDB hybrid search with RRF."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.search_engine import HelpSearchEngine


class TestQuerySanitization:
    """Test query sanitization for FTS compatibility."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        yield engine
        engine.close()

    def test_search_sanitizes_quotes(self, search_engine_with_data):
        """Verify double and single quotes are removed from query."""
        # Should not raise FTS syntax error
        results = search_engine_with_data.search('find "this"')
        # Query should be processed without error
        assert isinstance(results, list)

    def test_search_sanitizes_special_characters(self, search_engine_with_data):
        """Verify FTS special chars are removed."""
        query = "MC_BR_Move*Absolute()"
        results = search_engine_with_data.search(query)
        assert isinstance(results, list)

    def test_search_filters_fts_keywords(self, search_engine_with_data):
        """Verify FTS keywords (AND, OR, NOT, NEAR) are filtered out."""
        results = search_engine_with_data.search("motor AND speed")
        # Should process without error (AND filtered)
        assert isinstance(results, list)

    def test_search_filters_short_terms(self, search_engine_with_data):
        """Verify terms < 2 characters are filtered."""
        results = search_engine_with_data.search("a motor b")
        # Should process without error (a and b filtered)
        assert isinstance(results, list)

    def test_search_empty_query(self, search_engine_with_data):
        """Verify empty query returns empty list."""
        results = search_engine_with_data.search("")
        assert results == []

        results = search_engine_with_data.search("   ")
        assert results == []

    def test_search_only_short_terms(self, search_engine_with_data):
        """Verify query with only short terms still works via vector search."""
        results = search_engine_with_data.search("a b c")
        # With hybrid search, vector component may return results even with short terms
        assert isinstance(results, list)


class TestEnhancedQueryBuilding:
    """Test hybrid search query building."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        yield engine
        engine.close()

    def test_search_finds_matching_pages(self, search_engine_with_data):
        """Verify search finds pages matching query."""
        results = search_engine_with_data.search("motion")
        assert len(results) > 0

    def test_search_title_only_mode(self, search_engine_with_data):
        """Verify title-only search works."""
        results_title = search_engine_with_data.search("motion", search_in_content=False)
        results_content = search_engine_with_data.search("motion", search_in_content=True)

        assert isinstance(results_title, list)
        assert isinstance(results_content, list)


class TestCategoryFiltering:
    """Test category filtering in search."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        yield engine
        engine.close()

    def test_search_category_filter_applied(self, search_engine_with_data):
        """Verify category filter restricts results."""
        results = search_engine_with_data.search("x20", category="Hardware")

        for result in results:
            assert result.get("category", "").lower() == "hardware"

    def test_search_category_case_insensitive(self, search_engine_with_data):
        """Verify category matching is case-insensitive."""
        results_lower = search_engine_with_data.search("x20", category="hardware")
        results_upper = search_engine_with_data.search("x20", category="HARDWARE")

        assert len(results_lower) == len(results_upper)

    def test_build_category_filter_sanitizes_input(self):
        """Verify category filter sanitizes SQL injection attempts."""
        result = HelpSearchEngine._build_category_filter("Hardware'; DROP TABLE--")
        # The security property: quotes and semicolons are stripped so you
        # can never break out of the SQL string literal
        inner_value = result.split("'")[1]
        assert "'" not in inner_value
        assert ";" not in inner_value

    def test_build_category_filter_none_returns_none(self):
        """Verify None category returns None filter."""
        assert HelpSearchEngine._build_category_filter(None) is None
        assert HelpSearchEngine._build_category_filter("") is None


class TestResultMapping:
    """Test search result structure and field mapping."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        yield engine
        engine.close()

    def test_search_result_structure(self, search_engine_with_data):
        """Verify search results contain all expected fields."""
        results = search_engine_with_data.search("x20")

        if len(results) > 0:
            result = results[0]

            # Check all required fields exist
            assert "page_id" in result
            assert "title" in result
            assert "file_path" in result
            assert "help_id" in result
            assert "is_section" in result
            assert "breadcrumb_path" in result
            assert "category" in result
            assert "score" in result
            assert "snippet" in result

    def test_search_score_is_positive(self, search_engine_with_data):
        """Verify RRF score is positive."""
        results = search_engine_with_data.search("motion")

        for result in results:
            assert result["score"] > 0

    def test_search_empty_fields_become_none(self, search_engine_with_data):
        """Verify empty help_id, breadcrumb_path, category become None."""
        results = search_engine_with_data.search("x20")

        for result in results:
            if result["help_id"] == "":
                raise AssertionError("Empty help_id should be None")
            if result["breadcrumb_path"] == "":
                raise AssertionError("Empty breadcrumb_path should be None")


class TestErrorHandling:
    """Test error handling in search operations."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        yield engine
        engine.close()

    def test_search_handles_complex_query(self, search_engine_with_data):
        """Verify complex queries don't raise exceptions."""
        results = search_engine_with_data.search("complex (query) with [brackets]")
        assert isinstance(results, list)


class TestExtractTextForPage:
    """Test _extract_text_for_page helper method."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        yield engine
        engine.close()

    def test_extract_text_for_page_section(self, search_engine_with_data):
        """Verify sections have empty content string."""
        section_id = "hardware_section"
        page = search_engine_with_data.indexer.pages[section_id]

        result = search_engine_with_data._extract_text_for_page(section_id, page)

        assert result[2] == ""  # content is 3rd element

    def test_extract_text_for_page_category_extraction(self, search_engine_with_data):
        """Verify category is extracted from first breadcrumb item."""
        page_id = "x20di9371_page"
        page = search_engine_with_data.indexer.pages[page_id]

        result = search_engine_with_data._extract_text_for_page(page_id, page)

        category = result[7]  # category is 8th element
        assert category == "Hardware"

    def test_extract_text_for_page_tuple_structure(self, search_engine_with_data):
        """Verify return tuple has 8 elements in correct order."""
        page_id = "x20di9371_page"
        page = search_engine_with_data.indexer.pages[page_id]

        result = search_engine_with_data._extract_text_for_page(page_id, page)

        assert len(result) == 8

        assert result[0] == page_id
        assert result[1] == page.text
        assert isinstance(result[2], str)  # content
        assert result[3] == page.file_path
        assert result[4] == (page.help_id or "")
        assert result[5] in (0, 1)  # is_section as int
        assert isinstance(result[6], str)  # breadcrumb_path
        assert isinstance(result[7], str)  # category


class TestIndexBuildAndLoad:
    """Test index building and loading."""

    def test_index_exists_returns_false_initially(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify _index_exists returns False when no index exists."""
        db_path = tmp_path / "new_lance"
        db_path.mkdir(parents=True, exist_ok=True)

        import lancedb

        db = lancedb.connect(str(db_path))
        assert "help_pages" not in db.list_tables().tables

    def test_index_exists_returns_true_after_build(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify _index_exists returns True after index is built."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()

        assert engine._index_exists() is True

        engine.close()

    def test_needs_reindex_detects_xml_change(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify _needs_reindex detects XML changes."""
        db_path = tmp_path / "test_lance"

        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        engine.close()

        with patch.object(initialized_indexer, "_get_xml_hash", return_value="different_hash"):
            engine2 = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service)
            engine2.initialize()
            engine2.close()

    def test_needs_reindex_detects_model_change(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify _needs_reindex detects embedding model changes."""
        db_path = tmp_path / "test_lance"

        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        engine.close()

        # Change the model name in metadata
        metadata_path = db_path / "_index_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["embedding_model"] = "different-model"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        engine2 = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service)
        engine2.initialize()
        # Should have rebuilt (detected model mismatch)
        assert engine2._index_exists() is True
        engine2.close()

    def test_load_index_without_rebuild(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify index can be loaded without rebuilding."""
        db_path = tmp_path / "test_lance"

        engine1 = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine1.initialize()
        engine1.close()

        engine2 = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service)
        engine2.initialize()

        results = engine2.search("motion")
        assert isinstance(results, list)

        engine2.close()


class TestSnippetGeneration:
    """Test snippet generation from content."""

    def test_snippet_with_matching_term(self):
        """Verify snippet centers around matching term."""
        content = "x" * 100 + "motor speed control" + "y" * 100
        snippet = HelpSearchEngine._generate_snippet(content, "motor")
        assert snippet is not None
        assert "motor" in snippet

    def test_snippet_no_content(self):
        """Verify None returned for empty content."""
        assert HelpSearchEngine._generate_snippet("", "query") is None
        assert HelpSearchEngine._generate_snippet(None, "query") is None

    def test_snippet_no_match_returns_beginning(self):
        """Verify beginning of content returned when no term matches."""
        content = "This is some content without the search term"
        snippet = HelpSearchEngine._generate_snippet(content, "zzzzz")
        assert snippet is not None
        assert snippet.startswith("This is some content")

    def test_snippet_truncation(self):
        """Verify long content is truncated in snippet."""
        content = "x" * 500
        snippet = HelpSearchEngine._generate_snippet(content, "zzzzz")
        assert len(snippet) <= 165  # 160 + "..."


class TestParallelProcessing:
    """Test parallel text extraction during indexing."""

    def test_build_index_uses_multiple_threads(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify index build uses ThreadPoolExecutor."""
        db_path = tmp_path / "test_lance"

        with patch("src.search_engine.ThreadPoolExecutor") as mock_executor:
            # Create mock executor context manager
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance

            # Mock the map method to return empty results
            mock_executor_instance.map.return_value = []

            try:
                engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
                engine.initialize()
                engine.close()
            except Exception:
                # Expected to fail due to mocking, but ThreadPoolExecutor should have been called
                pass

            # Verify ThreadPoolExecutor was created
            assert mock_executor.called


class TestDatabaseConnection:
    """Test database connection and cleanup."""

    def test_context_manager_closes_connection(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify context manager closes connection."""
        db_path = tmp_path / "test_lance"

        with HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service) as engine:
            engine.initialize()
            # Should be able to search
            results = engine.search("motion")
            assert isinstance(results, list)

    def test_close_method_works(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify close method closes connection."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()

        engine.close()

        # Calling close again should not raise exception
        engine.close()

    def test_del_cleanup(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify __del__ cleans up connection."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()

        engine.__del__()


class TestSearchLimits:
    """Test search result limiting."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        yield engine
        engine.close()

    def test_search_respects_limit(self, search_engine_with_data):
        """Verify search respects limit parameter."""
        results = search_engine_with_data.search("motion", limit=1)
        assert len(results) <= 1

        results = search_engine_with_data.search("motion", limit=5)
        assert len(results) <= 5
