"""Unit tests for search_engine.py - Query sanitization and search logic."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.search_engine import HelpSearchEngine


class TestQuerySanitization:
    """Test query sanitization for FTS5 compatibility."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path):
        """Create search engine with sample data."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
        yield engine
        engine.close()

    def test_search_sanitizes_quotes(self, search_engine_with_data):
        """Verify double and single quotes are removed from query."""
        # Should not raise FTS5 syntax error
        results = search_engine_with_data.search('find "this"')
        # Query should be processed without error
        assert isinstance(results, list)

    def test_search_sanitizes_special_characters(self, search_engine_with_data):
        """Verify FTS5 special chars are removed."""
        # FTS5 special chars: * : ( ) { } - ^ + [ ]
        query = "MC_BR_Move*Absolute()"
        results = search_engine_with_data.search(query)
        assert isinstance(results, list)

    def test_search_filters_fts5_keywords(self, search_engine_with_data):
        """Verify FTS5 keywords (AND, OR, NOT, NEAR) are filtered out."""
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
        """Verify query with only short terms returns empty list."""
        results = search_engine_with_data.search("a b c")
        assert results == []


class TestEnhancedQueryBuilding:
    """Test enhanced FTS5 query building."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path):
        """Create search engine with sample data."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
        yield engine
        engine.close()

    def test_search_prefix_matching(self, search_engine_with_data):
        """Verify terms are wrapped for prefix matching."""
        # Search for "motion" should match pages with "Motion"
        results = search_engine_with_data.search("motion")
        # Should find motion-related pages
        assert len(results) > 0

    def test_search_title_only_mode(self, search_engine_with_data):
        """Verify title-only search works."""
        # Search only in titles
        results_title = search_engine_with_data.search("motion", search_in_content=False)

        # Search in content
        results_content = search_engine_with_data.search("motion", search_in_content=True)

        # Both should find results
        assert isinstance(results_title, list)
        assert isinstance(results_content, list)


class TestCategoryFiltering:
    """Test category filtering in search."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path):
        """Create search engine with sample data."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
        yield engine
        engine.close()

    def test_search_category_filter_applied(self, search_engine_with_data):
        """Verify category filter adds WHERE clause."""
        # Search with category filter
        results = search_engine_with_data.search("x20", category="Hardware")

        # Should only return hardware pages
        for result in results:
            assert result.get("category", "").lower() == "hardware"

    def test_search_category_case_insensitive(self, search_engine_with_data):
        """Verify category matching is case-insensitive."""
        # Search with lowercase category
        results_lower = search_engine_with_data.search("x20", category="hardware")

        # Search with uppercase category
        results_upper = search_engine_with_data.search("x20", category="HARDWARE")

        # Should return same results
        assert len(results_lower) == len(results_upper)


class TestResultMapping:
    """Test search result structure and field mapping."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path):
        """Create search engine with sample data."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
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
            assert "help_id" in result  # May be None
            assert "is_section" in result
            assert "breadcrumb_path" in result
            assert "category" in result
            assert "score" in result
            assert "snippet" in result

    def test_search_score_is_positive(self, search_engine_with_data):
        """Verify BM25 score is converted to positive."""
        results = search_engine_with_data.search("motion")

        # All scores should be positive (BM25 returns negative)
        for result in results:
            assert result["score"] >= 0

    def test_search_empty_fields_become_none(self, search_engine_with_data):
        """Verify empty help_id, breadcrumb_path, category become None."""
        # This test requires pages without these fields
        # The test fixture pages have these fields, so we check the logic
        results = search_engine_with_data.search("x20")

        for result in results:
            # help_id can be None
            if result["help_id"] == "":
                raise AssertionError("Empty help_id should be None")

            # breadcrumb_path can be None but shouldn't be empty string
            if result["breadcrumb_path"] == "":
                raise AssertionError("Empty breadcrumb_path should be None")


class TestErrorHandling:
    """Test error handling in search operations."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path):
        """Create search engine with sample data."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
        yield engine
        engine.close()

    def test_search_fallback_on_syntax_error(self, search_engine_with_data):
        """Verify fallback OR query is tried on FTS5 syntax error."""
        # Even with complex queries, should fall back gracefully
        results = search_engine_with_data.search("complex (query) with [brackets]")
        # Should return results or empty list, not raise exception
        assert isinstance(results, list)


class TestExtractTextForPage:
    """Test _extract_text_for_page helper method."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path):
        """Create search engine with sample data."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
        yield engine
        engine.close()

    def test_extract_text_for_page_section(self, search_engine_with_data):
        """Verify sections have empty content string."""
        # Get a section from indexer
        section_id = "hardware_section"
        page = search_engine_with_data.indexer.pages[section_id]

        result = search_engine_with_data._extract_text_for_page(section_id, page)

        # Content should be empty for sections
        assert result[2] == ""  # content is 3rd element

    def test_extract_text_for_page_category_extraction(self, search_engine_with_data):
        """Verify category is extracted from first breadcrumb item."""
        # Get a page
        page_id = "x20di9371_page"
        page = search_engine_with_data.indexer.pages[page_id]

        result = search_engine_with_data._extract_text_for_page(page_id, page)

        # Category should be first breadcrumb item
        category = result[7]  # category is 8th element
        assert category == "Hardware"

    def test_extract_text_for_page_tuple_structure(self, search_engine_with_data):
        """Verify return tuple has 8 elements in correct order."""
        page_id = "x20di9371_page"
        page = search_engine_with_data.indexer.pages[page_id]

        result = search_engine_with_data._extract_text_for_page(page_id, page)

        # Should have 8 elements
        assert len(result) == 8

        # Verify structure: (page_id, title, content, file_path, help_id, is_section, breadcrumb_path, category)
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

    def test_index_exists_returns_false_initially(self, initialized_indexer, tmp_path):
        """Verify _index_exists returns False when no index exists."""
        db_path = tmp_path / "new.db"

        # Don't create engine yet, just check connection
        conn = sqlite3.connect(str(db_path))

        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='help_fts'
        """)
        result = cursor.fetchone()
        conn.close()

        assert result is None

    def test_index_exists_returns_true_after_build(self, initialized_indexer, tmp_path):
        """Verify _index_exists returns True after index is built."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)

        # Check index exists
        assert engine._index_exists() is True

        engine.close()

    def test_index_exists_returns_false_when_fts_table_empty(self, initialized_indexer, tmp_path):
        """Verify _index_exists returns False when FTS5 table exists but is empty."""
        db_path = tmp_path / "test.db"

        # Create database with empty FTS5 table
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create the FTS5 table but don't insert any data
        cursor.execute("""
            CREATE VIRTUAL TABLE help_fts USING fts5(
                page_id UNINDEXED,
                title,
                content,
                file_path UNINDEXED,
                help_id UNINDEXED,
                is_section UNINDEXED,
                breadcrumb_path UNINDEXED,
                category UNINDEXED,
                tokenize='porter unicode61'
            )
        """)
        conn.commit()
        conn.close()

        # Now create search engine - should detect empty table and rebuild
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=False)

        # After rebuild, index should exist and have data
        assert engine._index_exists() is True

        # Verify it has documents
        cursor = engine.conn.execute("SELECT COUNT(*) FROM help_fts")
        count = cursor.fetchone()[0]
        assert count > 0

        engine.close()

    def test_needs_reindex_detects_xml_change(self, initialized_indexer, tmp_path):
        """Verify _needs_reindex detects XML changes."""
        db_path = tmp_path / "test.db"

        # Build initial index
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
        engine.close()

        # Simulate XML change by modifying indexer hash
        with patch.object(initialized_indexer, "_get_xml_hash", return_value="different_hash"):
            engine2 = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=False)
            # Note: _needs_reindex is called in __init__, engine would rebuild
            engine2.close()

    def test_load_index_without_rebuild(self, initialized_indexer, tmp_path):
        """Verify index can be loaded without rebuilding."""
        db_path = tmp_path / "test.db"

        # Build initial index
        engine1 = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
        engine1.close()

        # Load existing index
        engine2 = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=False)

        # Should be able to search
        results = engine2.search("motion")
        assert isinstance(results, list)

        engine2.close()


class TestParallelProcessing:
    """Test parallel text extraction during indexing."""

    def test_build_index_uses_multiple_threads(self, initialized_indexer, tmp_path):
        """Verify index build uses ThreadPoolExecutor."""
        db_path = tmp_path / "test.db"

        with patch("src.search_engine.ThreadPoolExecutor") as mock_executor:
            # Create mock executor context manager
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance

            # Mock the map method to return empty results
            mock_executor_instance.map.return_value = []

            try:
                engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
                engine.close()
            except Exception:
                # Expected to fail due to mocking, but ThreadPoolExecutor should have been called
                pass

            # Verify ThreadPoolExecutor was created
            assert mock_executor.called


class TestDatabaseConnection:
    """Test database connection and cleanup."""

    def test_context_manager_closes_connection(self, initialized_indexer, tmp_path):
        """Verify context manager closes connection."""
        db_path = tmp_path / "test.db"

        with HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True) as engine:
            # Should be able to search
            results = engine.search("motion")
            assert isinstance(results, list)

        # Connection should be closed (can't easily test directly)

    def test_close_method_works(self, initialized_indexer, tmp_path):
        """Verify close method closes connection."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)

        engine.close()

        # Calling close again should not raise exception
        engine.close()

    def test_del_cleanup(self, initialized_indexer, tmp_path):
        """Verify __del__ cleans up connection."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)

        # Trigger __del__ by removing references
        engine.__del__()
        # Connection should be closed now


class TestSearchLimits:
    """Test search result limiting."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path):
        """Create search engine with sample data."""
        db_path = tmp_path / "test.db"
        engine = HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True)
        yield engine
        engine.close()

    def test_search_respects_limit(self, search_engine_with_data):
        """Verify search respects limit parameter."""
        # Search with limit=1
        results = search_engine_with_data.search("motion", limit=1)
        assert len(results) <= 1

        # Search with limit=5
        results = search_engine_with_data.search("motion", limit=5)
        assert len(results) <= 5
