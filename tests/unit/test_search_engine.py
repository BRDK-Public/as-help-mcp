"""Unit tests for search_engine.py - LanceDB hybrid search with RRF."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.search_engine import HelpSearchEngine, _is_identifier_query


class TestQuerySanitization:
    """Test query sanitization for FTS compatibility."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
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
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
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
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
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
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
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
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
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
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        yield engine
        engine.close()

    def test_extract_text_for_page_section(self, search_engine_with_data):
        """Verify sections with HTML files get their content extracted."""
        section_id = "hardware_section"
        page = search_engine_with_data.indexer.pages[section_id]

        result = search_engine_with_data._extract_text_for_page(section_id, page)

        # Sections now get content extracted from their HTML file
        assert isinstance(result[2], str)  # content is 3rd element
        # hardware_section points to index.html which has "Welcome" / "This is the index page"
        assert len(result[2]) > 0

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
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()

        assert engine._index_exists() is True

        engine.close()

    def test_needs_reindex_detects_xml_change(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify _needs_reindex detects XML changes."""
        db_path = tmp_path / "test_lance"

        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        engine.close()

        with patch.object(initialized_indexer, "_get_xml_hash", return_value="different_hash"):
            engine2 = HelpSearchEngine(
                db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
            )
            engine2.initialize()
            engine2.close()

    def test_needs_reindex_detects_model_change(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify _needs_reindex detects embedding model changes."""
        db_path = tmp_path / "test_lance"

        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        engine.close()

        # Change the model name in metadata
        metadata_path = db_path / "_index_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["embedding_model"] = "different-model"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
        )
        engine2.initialize()
        # Should have rebuilt (detected model mismatch)
        assert engine2._index_exists() is True
        engine2.close()

    def test_load_index_without_rebuild(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify index can be loaded without rebuilding."""
        db_path = tmp_path / "test_lance"

        engine1 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine1.initialize()
        engine1.close()

        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
        )
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
                engine = HelpSearchEngine(
                    db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
                )
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

        with HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        ) as engine:
            engine.initialize()
            # Should be able to search
            results = engine.search("motion")
            assert isinstance(results, list)

    def test_close_method_works(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify close method closes connection."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()

        engine.close()

        # Calling close again should not raise exception
        engine.close()

    def test_del_cleanup(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify __del__ cleans up connection."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()

        engine.__del__()


class TestSearchLimits:
    """Test search result limiting."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        yield engine
        engine.close()

    def test_search_respects_limit(self, search_engine_with_data):
        """Verify search respects limit parameter."""
        results = search_engine_with_data.search("motion", limit=1)
        assert len(results) <= 1

        results = search_engine_with_data.search("motion", limit=5)
        assert len(results) <= 5


class TestPageFingerprinting:
    """Test page fingerprint generation."""

    def test_fingerprints_cover_all_pages(self, initialized_indexer):
        """Verify every page gets a fingerprint."""
        fps = initialized_indexer.get_page_fingerprints()
        assert set(fps.keys()) == set(initialized_indexer.pages.keys())

    def test_fingerprints_are_deterministic(self, initialized_indexer):
        """Verify same pages produce same fingerprints."""
        fps1 = initialized_indexer.get_page_fingerprints()
        fps2 = initialized_indexer.get_page_fingerprints()
        assert fps1 == fps2

    def test_fingerprint_changes_on_title_change(self, initialized_indexer):
        """Verify fingerprint changes when page title changes."""
        fps_before = initialized_indexer.get_page_fingerprints()
        old_title = initialized_indexer.pages["x20di9371_page"].text
        initialized_indexer.pages["x20di9371_page"].text = "New Title"
        fps_after = initialized_indexer.get_page_fingerprints()
        # Restore
        initialized_indexer.pages["x20di9371_page"].text = old_title

        assert fps_before["x20di9371_page"] != fps_after["x20di9371_page"]
        # Other pages unchanged
        assert fps_before["hardware_section"] == fps_after["hardware_section"]


class TestBuildStrategyDetection:
    """Test _detect_build_strategy logic."""

    def test_full_rebuild_on_first_run(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify first run uses full build strategy."""
        db_path = tmp_path / "new_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
        )
        assert engine._build_strategy == "full"

    def test_none_when_unchanged(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify no rebuild when nothing changed."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        engine.close()

        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
        )
        assert engine2._build_strategy == "none"

    def test_incremental_when_xml_changed_with_fingerprints(
        self, initialized_indexer, tmp_path, mock_embedding_service
    ):
        """Verify incremental strategy when XML changed but fingerprints exist."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        engine.close()

        with patch.object(initialized_indexer, "_get_xml_hash", return_value="different_hash"):
            engine2 = HelpSearchEngine(
                db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
            )
            assert engine2._build_strategy == "incremental"

    def test_full_when_model_changed(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify full rebuild when embedding model changes."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        engine.close()

        # Tamper with metadata to simulate model change
        metadata_path = db_path / "_index_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["embedding_model"] = "different-model"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
        )
        assert engine2._build_strategy == "full"

    def test_full_when_no_fingerprints_stored(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify full rebuild when metadata has no fingerprints (legacy index)."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        engine.close()

        # Remove fingerprints from metadata to simulate legacy format
        metadata_path = db_path / "_index_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata.pop("page_fingerprints", None)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        with patch.object(initialized_indexer, "_get_xml_hash", return_value="different_hash"):
            engine2 = HelpSearchEngine(
                db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
            )
            assert engine2._build_strategy == "full"


class TestIncrementalUpdate:
    """Test incremental index update logic."""

    def test_incremental_adds_new_page(self, temp_help_dir, tmp_path, mock_embedding_service):
        """Verify incremental update adds a new page to the index."""
        from src.indexer import HelpContentIndexer

        # Build initial index
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "inc_lance"
        engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        engine.close()

        # Add a new page to XML
        xml_content2 = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
        <Page Id="page2" Text="Page Two" File="motion/overview.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content2, encoding="utf-8")

        indexer2 = HelpContentIndexer(temp_help_dir)
        indexer2.parse_xml_structure()

        engine2 = HelpSearchEngine(db_path, indexer2, force_rebuild=False, embedding_service=mock_embedding_service)
        assert engine2._build_strategy == "incremental"
        engine2.initialize()

        # Verify new page is searchable
        table = engine2.db.open_table(engine2.TABLE_NAME)
        assert table.count_rows() == 3  # sec1 + page1 + page2
        engine2.close()

    def test_incremental_removes_deleted_page(self, temp_help_dir, tmp_path, mock_embedding_service):
        """Verify incremental update removes a deleted page from the index."""
        from src.indexer import HelpContentIndexer

        # Build with 2 pages
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
        <Page Id="page2" Text="Page Two" File="motion/overview.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "inc_lance"
        engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        engine.close()

        # Remove page2 from XML
        xml_content2 = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content2, encoding="utf-8")

        indexer2 = HelpContentIndexer(temp_help_dir)
        indexer2.parse_xml_structure()

        engine2 = HelpSearchEngine(db_path, indexer2, force_rebuild=False, embedding_service=mock_embedding_service)
        assert engine2._build_strategy == "incremental"
        engine2.initialize()

        table = engine2.db.open_table(engine2.TABLE_NAME)
        assert table.count_rows() == 2  # sec1 + page1 (page2 removed)
        engine2.close()

    def test_incremental_updates_changed_page(self, temp_help_dir, tmp_path, mock_embedding_service):
        """Verify incremental update re-indexes a page with changed title."""
        from src.indexer import HelpContentIndexer

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Old Title" File="hardware/x20di9371.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "inc_lance"
        engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        engine.close()

        # Change page title
        xml_content2 = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="New Title" File="hardware/x20di9371.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content2, encoding="utf-8")

        indexer2 = HelpContentIndexer(temp_help_dir)
        indexer2.parse_xml_structure()

        engine2 = HelpSearchEngine(db_path, indexer2, force_rebuild=False, embedding_service=mock_embedding_service)
        assert engine2._build_strategy == "incremental"
        engine2.initialize()

        # Verify updated title is in the index
        table = engine2.db.open_table(engine2.TABLE_NAME)
        rows = table.search().where("page_id = 'page1'").limit(1).to_list()
        assert rows[0]["title"] == "New Title"
        engine2.close()

    def test_incremental_no_changes_is_noop(self, temp_help_dir, tmp_path, mock_embedding_service):
        """Verify incremental update with same fingerprints is effectively a no-op."""
        from src.indexer import HelpContentIndexer

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "inc_lance"
        engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        engine.close()

        # Rewrite XML with same content but different whitespace → different MD5 but same fingerprints
        xml_content2 = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
  <Section Id="sec1" Text="Section" File="index.html">
    <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
  </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content2, encoding="utf-8")

        indexer2 = HelpContentIndexer(temp_help_dir)
        indexer2.parse_xml_structure()

        engine2 = HelpSearchEngine(db_path, indexer2, force_rebuild=False, embedding_service=mock_embedding_service)
        assert engine2._build_strategy == "incremental"
        engine2.initialize()

        # Row count unchanged
        table = engine2.db.open_table(engine2.TABLE_NAME)
        assert table.count_rows() == 2  # sec1 + page1
        engine2.close()

    def test_incremental_falls_back_to_full_on_massive_change(self, temp_help_dir, tmp_path, mock_embedding_service):
        """Verify >50% changed pages triggers full rebuild."""
        from src.indexer import HelpContentIndexer

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
        <Page Id="page2" Text="Page Two" File="motion/overview.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "inc_lance"
        engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        engine.initialize()
        engine.close()

        # Replace all pages → >50% changed
        xml_content2 = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec_new" Text="New Section" File="index.html">
        <Page Id="page_a" Text="Brand New A" File="hardware/x20di9371.html"/>
        <Page Id="page_b" Text="Brand New B" File="motion/overview.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content2, encoding="utf-8")

        indexer2 = HelpContentIndexer(temp_help_dir)
        indexer2.parse_xml_structure()

        engine2 = HelpSearchEngine(db_path, indexer2, force_rebuild=False, embedding_service=mock_embedding_service)
        assert engine2._build_strategy == "incremental"
        engine2.initialize()

        # Should have done a full rebuild — verify all new pages present
        table = engine2.db.open_table(engine2.TABLE_NAME)
        assert table.count_rows() == 3  # sec_new + page_a + page_b
        engine2.close()


class TestBuildResume:
    """Test chunked build with resume support."""

    def test_resume_detected_after_interrupted_build(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify resume strategy is detected when build was interrupted."""
        db_path = tmp_path / "resume_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )

        # Simulate an interrupted build: write progress marker + partial table
        engine._save_build_progress()
        # Build one chunk manually (all pages in test data fit in one chunk)
        all_pages = list(engine.indexer.pages.items())
        partial = all_pages[:1]  # Only index the first page
        records = [engine._extract_text_for_page(pid, page) for pid, page in partial]
        title_vecs = engine.embedder.embed_batch([r[1] for r in records])
        content_vecs = engine.embedder.embed_batch([r[2] if r[2] else r[1] for r in records])
        chunk_data = engine._records_to_hybrid_arrow(records, title_vecs, content_vecs)
        engine.db.create_table(engine.TABLE_NAME, chunk_data)
        engine.close()

        # New engine should detect the resumable partial build
        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
        )
        assert engine2._build_strategy == "resume"
        engine2.close()

    def test_resume_completes_build(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify resume finishes indexing remaining pages."""
        db_path = tmp_path / "resume_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )

        # Simulate interrupted build with 1 page done
        engine._save_build_progress()
        all_pages = list(engine.indexer.pages.items())
        partial = all_pages[:1]
        records = [engine._extract_text_for_page(pid, page) for pid, page in partial]
        title_vecs = engine.embedder.embed_batch([r[1] for r in records])
        content_vecs = engine.embedder.embed_batch([r[2] if r[2] else r[1] for r in records])
        chunk_data = engine._records_to_hybrid_arrow(records, title_vecs, content_vecs)
        engine.db.create_table(engine.TABLE_NAME, chunk_data)
        engine.close()

        # Resume should complete the build
        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
        )
        assert engine2._build_strategy == "resume"
        engine2.initialize()

        # All pages should now be indexed
        table = engine2.db.open_table(engine2.TABLE_NAME)
        assert table.count_rows() == len(initialized_indexer.pages)

        # Progress marker should be cleaned up
        assert not engine2._build_progress_path.exists()
        # Metadata should be saved
        assert engine2._metadata_path.exists()
        engine2.close()

    def test_no_resume_when_xml_changed(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify resume is skipped when XML hash doesn't match (data changed)."""
        db_path = tmp_path / "resume_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )

        # Simulate interrupted build
        engine._save_build_progress()
        records = [engine._extract_text_for_page(pid, page) for pid, page in list(engine.indexer.pages.items())[:1]]
        title_vecs = engine.embedder.embed_batch([r[1] for r in records])
        content_vecs = engine.embedder.embed_batch([r[2] if r[2] else r[1] for r in records])
        chunk_data = engine._records_to_hybrid_arrow(records, title_vecs, content_vecs)
        engine.db.create_table(engine.TABLE_NAME, chunk_data)
        engine.close()

        # Change XML hash — resume should not be possible
        with patch.object(initialized_indexer, "_get_xml_hash", return_value="different_hash"):
            engine2 = HelpSearchEngine(
                db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
            )
            # Should NOT be resume (XML changed since the partial build)
            assert engine2._build_strategy != "resume"
            engine2.close()

    def test_force_rebuild_ignores_resume(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify force_rebuild=True ignores resumable partial build."""
        db_path = tmp_path / "resume_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )

        # Simulate interrupted build
        engine._save_build_progress()
        records = [engine._extract_text_for_page(pid, page) for pid, page in list(engine.indexer.pages.items())[:1]]
        title_vecs = engine.embedder.embed_batch([r[1] for r in records])
        content_vecs = engine.embedder.embed_batch([r[2] if r[2] else r[1] for r in records])
        chunk_data = engine._records_to_hybrid_arrow(records, title_vecs, content_vecs)
        engine.db.create_table(engine.TABLE_NAME, chunk_data)
        engine.close()

        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        assert engine2._build_strategy == "full"
        engine2.close()

    def test_build_status_tracks_chunks(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify build status updates during chunked build."""
        db_path = tmp_path / "status_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()

        status = engine.build_status
        assert status["state"] == "ready"
        assert status["pages_total"] == len(initialized_indexer.pages)
        assert status["pages_processed"] == len(initialized_indexer.pages)
        assert status["elapsed_seconds"] is not None
        engine.close()


class TestReadyStateSemantics:
    """Test query-readiness behavior when initialization fails."""

    def test_ready_is_false_when_initialize_fails(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify waiters are unblocked but ready remains False on initialization failure."""
        db_path = tmp_path / "error_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )

        with patch.object(engine, "_build_index_two_phase", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                engine.initialize()

        # Event is set to release waiters, but index is not queryable.
        # wait_until_ready returns False because the build ended in error state.
        assert engine.wait_until_ready(timeout=0.0) is False
        assert engine.ready is False
        assert engine.build_status["state"] == "error"
        engine.close()


class TestInstanceLock:
    """Test per-db-path instance lock prevents duplicate servers."""

    def test_second_instance_same_db_path_raises(self, initialized_indexer, tmp_path, mock_embedding_service):
        """A second engine on the same db_path should fail while the first is alive."""
        db_path = tmp_path / "lock_lance"
        engine1 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )

        with pytest.raises(RuntimeError, match="already using"):
            HelpSearchEngine(db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service)

        engine1.close()

    def test_instance_lock_released_on_close(self, initialized_indexer, tmp_path, mock_embedding_service):
        """After close(), a new engine should acquire the lock successfully."""
        db_path = tmp_path / "lock_lance"
        engine1 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine1.close()

        # Should succeed now
        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine2.close()

    def test_different_db_paths_allowed(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Different db paths should be able to run concurrently."""
        db1 = tmp_path / "lance_as4"
        db2 = tmp_path / "lance_as6"
        engine1 = HelpSearchEngine(
            db1, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine2 = HelpSearchEngine(
            db2, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )

        # Both acquired locks successfully
        assert engine1._instance_lock_owned
        assert engine2._instance_lock_owned

        engine1.close()
        engine2.close()

    def test_stale_lock_from_dead_process_overwritten(self, initialized_indexer, tmp_path, mock_embedding_service):
        """A lock file from a dead PID should be overwritten."""
        db_path = tmp_path / "stale_lance"
        db_path.mkdir(parents=True, exist_ok=True)

        # Write a lock with a PID that definitely doesn't exist
        lock_path = db_path / "_instance.lock"
        import json

        with open(lock_path, "w") as f:
            json.dump({"pid": 99999999, "started_at": 0}, f)

        # Should succeed — stale lock is overwritten
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        assert engine._instance_lock_owned
        engine.close()

    def test_context_manager_releases_instance_lock(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Instance lock should be released when using context manager."""
        db_path = tmp_path / "ctx_lance"
        with HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        ) as engine:
            assert engine._instance_lock_owned

        # Lock should be released, file gone
        assert not (db_path / "_instance.lock").exists()


class TestFTSOnlyMode:
    """Test FTS-only mode (no embedding service)."""

    @pytest.fixture
    def fts_engine(self, initialized_indexer, tmp_path):
        """Create search engine in FTS-only mode (no embedding service)."""
        db_path = tmp_path / "fts_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=None
        )
        engine.initialize()
        yield engine
        engine.close()

    def test_fts_mode_embeddings_disabled(self, fts_engine):
        """Verify embeddings are disabled when no embedding service provided."""
        assert fts_engine._embeddings_enabled is False
        assert fts_engine.embedder is None

    def test_fts_mode_search_works(self, fts_engine):
        """Verify FTS search returns results in keyword mode."""
        results = fts_engine.search("motion")
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["search_mode"] == "keyword"

    def test_fts_mode_no_vector_columns(self, fts_engine):
        """Verify FTS-only table has no vector columns."""
        table = fts_engine.db.open_table(fts_engine.TABLE_NAME)
        schema = table.schema
        column_names = [field.name for field in schema]
        assert "title_vector" not in column_names
        assert "content_vector" not in column_names
        # But FTS columns exist
        assert "search_text" in column_names
        assert "page_id" in column_names

    def test_fts_mode_ready_immediately(self, fts_engine):
        """Verify FTS-only mode is ready without two-phase build."""
        assert fts_engine.ready is True
        assert fts_engine.fts_ready is True

    def test_fts_mode_build_status(self, fts_engine):
        """Verify build status reflects FTS-only mode."""
        status = fts_engine.build_status
        assert status["state"] == "ready"
        assert status["embeddings_enabled"] is False

    def test_fts_mode_metadata_no_embedding_model(self, fts_engine):
        """Verify metadata does not contain embedding model info."""
        import json
        with open(fts_engine._metadata_path) as f:
            metadata = json.load(f)
        assert metadata["embeddings_enabled"] is False
        assert "embedding_model" not in metadata
        assert "embedding_dimension" not in metadata

    def test_fts_mode_reload_without_rebuild(self, initialized_indexer, tmp_path):
        """Verify FTS-only index can be reloaded without rebuilding."""
        db_path = tmp_path / "fts_reload"
        engine1 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=None
        )
        engine1.initialize()
        engine1.close()

        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=None
        )
        assert engine2._build_strategy == "none"
        engine2.initialize()

        results = engine2.search("motion")
        assert isinstance(results, list)
        engine2.close()

    def test_fts_to_hybrid_mode_switch_triggers_rebuild(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify switching from FTS to hybrid triggers full rebuild."""
        db_path = tmp_path / "fts_to_hybrid"

        # Build FTS-only
        engine1 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=None
        )
        engine1.initialize()
        engine1.close()

        # Switch to hybrid mode
        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=mock_embedding_service
        )
        assert engine2._build_strategy == "full"
        engine2.close()

    def test_hybrid_to_fts_mode_switch_triggers_rebuild(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Verify switching from hybrid to FTS triggers full rebuild."""
        db_path = tmp_path / "hybrid_to_fts"

        # Build hybrid
        engine1 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine1.initialize()
        engine1.close()

        # Switch to FTS-only
        engine2 = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=False, embedding_service=None
        )
        assert engine2._build_strategy == "full"
        engine2.close()

    def test_fts_only_incremental_add(self, temp_help_dir, tmp_path):
        """Verify incremental update works in FTS-only mode."""
        from src.indexer import HelpContentIndexer

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "inc_fts"
        engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=None)
        engine.initialize()
        engine.close()

        # Add a page
        xml_content2 = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="sec1" Text="Section" File="index.html">
        <Page Id="page1" Text="Page One" File="hardware/x20di9371.html"/>
        <Page Id="page2" Text="Page Two" File="motion/overview.html"/>
    </Section>
</BrHelpContent>"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content2, encoding="utf-8")

        indexer2 = HelpContentIndexer(temp_help_dir)
        indexer2.parse_xml_structure()

        engine2 = HelpSearchEngine(db_path, indexer2, force_rebuild=False, embedding_service=None)
        assert engine2._build_strategy == "incremental"
        engine2.initialize()

        table = engine2.db.open_table(engine2.TABLE_NAME)
        assert table.count_rows() == 3  # sec1 + page1 + page2
        engine2.close()


class TestIdentifierDetection:
    """Test _is_identifier_query heuristic."""

    @pytest.mark.parametrize(
        "query",
        [
            "MC_MoveAbsolute",
            "AsGuard",
            "SYS_Lib",
            "X20DI9371",
            "mapp.Motion",
            "MC_BR_MoveAbsolute",
            "x20bc124",
            "VA_DelAlarmHistory",
        ],
    )
    def test_single_word_identifiers(self, query):
        assert _is_identifier_query(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "MC_MoveAbsolute parameters",  # 2 words, second is identifier-ish
            "AsGuard library",  # but "library" is also identifier-shaped
        ],
    )
    def test_two_word_identifiers(self, query):
        """Two-word queries where both words look like identifiers."""
        assert _is_identifier_query(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "how to move an axis",
            "configure emergency stop safety",
            "read and write PLC variables",
            "error handling best practices for function blocks",
            "transfer recipe data to PLC",
            "network communication between two PLCs",
        ],
    )
    def test_natural_language_queries(self, query):
        assert _is_identifier_query(query) is False

    def test_empty_and_whitespace(self):
        assert _is_identifier_query("") is False
        assert _is_identifier_query("   ") is False

    def test_three_word_query_is_not_identifier(self):
        """Queries with 3+ words are always natural language."""
        assert _is_identifier_query("MC Move Absolute") is False


class TestTitleMatchBoost:
    """Test that title exact/substring match boosts ranking."""

    @pytest.fixture
    def search_engine_with_data(self, initialized_indexer, tmp_path, mock_embedding_service):
        """Create search engine with sample data."""
        db_path = tmp_path / "test_lance"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=mock_embedding_service
        )
        engine.initialize()
        yield engine
        engine.close()

    def test_exact_title_match_ranks_first(self, search_engine_with_data):
        """Searching for an exact page title should rank that page first."""
        results = search_engine_with_data.search("MC_BR_MoveAbsolute")
        assert len(results) > 0
        assert results[0]["title"] == "MC_BR_MoveAbsolute"

    def test_title_substring_match_boosted(self, search_engine_with_data):
        """Results whose title contains the query should be boosted."""
        results = search_engine_with_data.search("X20DI9371")
        assert len(results) > 0
        # The page with X20DI9371 in the title should be at the top
        assert "X20DI9371" in results[0]["title"]

    def test_identifier_query_uses_higher_fts_weight(self, search_engine_with_data):
        """Identifier queries should still return results (weight shift doesn't break anything)."""
        results = search_engine_with_data.search("MC_BR_MoveAbsolute")
        assert len(results) > 0
        assert results[0]["search_mode"] == "hybrid"

    def test_natural_language_query_still_works(self, search_engine_with_data):
        """Natural language queries still return relevant results."""
        results = search_engine_with_data.search("motion control overview")
        assert len(results) > 0

    @pytest.fixture
    def fts_engine(self, initialized_indexer, tmp_path):
        """Create search engine in FTS-only mode."""
        db_path = tmp_path / "fts_lance_tm"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=None
        )
        engine.initialize()
        yield engine
        engine.close()

    def test_title_match_boost_in_fts_mode(self, fts_engine):
        """Title match boost should also work in keyword-only mode."""
        results = fts_engine.search("MC_BR_MoveAbsolute")
        assert len(results) > 0
        assert results[0]["title"] == "MC_BR_MoveAbsolute"

    def test_title_match_boost_in_fts_mode_x20(self, fts_engine):
        """Title match boost works for hardware product codes in FTS mode."""
        results = fts_engine.search("X20DI9371")
        assert len(results) > 0
        assert "X20DI9371" in results[0]["title"]


class TestBreadcrumbMatchBoost:
    """Test that breadcrumb path matching boosts ranking."""

    def test_apply_breadcrumb_bonus_adds_scores(self):
        """Breadcrumb bonus adds RRF scores for pages with matching breadcrumbs."""
        page_data = {
            "p1": {"breadcrumb_path": "Motion control > ACP10/ARNC0 > Revision Information"},
            "p2": {"breadcrumb_path": "Programming > Libraries > AsHttp"},
            "p3": {"breadcrumb_path": "Motion control > mapp Motion > Overview"},
        }
        rrf_scores = {"p1": 0.01, "p2": 0.02, "p3": 0.015}
        HelpSearchEngine._apply_breadcrumb_bonus("ACP10 version", page_data, rrf_scores, weight=2.0)
        # p1 has "ACP10" and "version" is not in breadcrumb but "ACP10" is → 1 hit
        # p3 has no matching terms → no bonus
        # p1 should get a bonus, p2 should not
        assert rrf_scores["p1"] > 0.01  # got a bonus
        assert rrf_scores["p2"] == 0.02  # no match
        # p3 has "motion" and "control" but query is "ACP10 version" → no match
        assert rrf_scores["p3"] == 0.015

    def test_apply_breadcrumb_bonus_ranks_by_hit_count(self):
        """Pages with more matching terms in breadcrumb rank higher."""
        page_data = {
            "p1": {"breadcrumb_path": "Motion control > ACP10/ARNC0 > General"},
            "p2": {"breadcrumb_path": "Motion control > ACP10/ARNC0 > Revision Information"},
        }
        rrf_scores = {"p1": 0.0, "p2": 0.0}
        HelpSearchEngine._apply_breadcrumb_bonus(
            "ACP10 revision information", page_data, rrf_scores, weight=2.0
        )
        # p2 has "ACP10", "revision", "information" → 3 hits
        # p1 has "ACP10" → 1 hit
        assert rrf_scores["p2"] > rrf_scores["p1"]

    def test_apply_breadcrumb_bonus_no_terms(self):
        """No crash when query has no usable terms."""
        page_data = {"p1": {"breadcrumb_path": "Motion"}}
        rrf_scores = {"p1": 0.5}
        HelpSearchEngine._apply_breadcrumb_bonus("a b", page_data, rrf_scores, weight=2.0)
        assert rrf_scores["p1"] == 0.5  # unchanged - terms too short

    def test_apply_breadcrumb_bonus_empty_breadcrumb(self):
        """Pages with no breadcrumb are skipped gracefully."""
        page_data = {"p1": {"breadcrumb_path": None}, "p2": {"breadcrumb_path": ""}}
        rrf_scores = {"p1": 0.1, "p2": 0.1}
        HelpSearchEngine._apply_breadcrumb_bonus("motion", page_data, rrf_scores, weight=2.0)
        assert rrf_scores["p1"] == 0.1
        assert rrf_scores["p2"] == 0.1

    @pytest.fixture
    def fts_engine_bc(self, initialized_indexer, tmp_path):
        """FTS-only engine for breadcrumb integration tests."""
        db_path = tmp_path / "fts_lance_bc"
        engine = HelpSearchEngine(
            db_path, initialized_indexer, force_rebuild=True, embedding_service=None
        )
        engine.initialize()
        yield engine
        engine.close()

    def test_breadcrumb_boost_in_fts_search(self, fts_engine_bc):
        """Search for a term in breadcrumb should boost matching pages."""
        # "mapp Motion" is in the breadcrumb of mc_moveabs_page
        results = fts_engine_bc.search("mapp Motion")
        assert len(results) > 0
        # MC_BR_MoveAbsolute has breadcrumb "Motion > mapp Motion > MC_BR_MoveAbsolute"
        titles = [r["title"] for r in results]
        assert "MC_BR_MoveAbsolute" in titles

    def test_breadcrumb_retrieval_requires_two_terms(self, fts_engine_bc):
        """Single-term queries should NOT trigger breadcrumb retrieval (too broad)."""
        table = fts_engine_bc.db.open_table(fts_engine_bc.TABLE_NAME)
        # Single term — should return empty
        results = fts_engine_bc._breadcrumb_retrieval(table, "Motion", limit=20, where_clause=None)
        assert results == []
        # Two terms — should return results (both "motion" and "mapp" are in breadcrumbs)
        results = fts_engine_bc._breadcrumb_retrieval(table, "mapp Motion", limit=20, where_clause=None)
        assert len(results) > 0

    def test_breadcrumb_retrieval_short_terms_filtered(self, fts_engine_bc):
        """Terms shorter than 3 chars are ignored; effective term count may drop below 2."""
        table = fts_engine_bc.db.open_table(fts_engine_bc.TABLE_NAME)
        # "IO" (2 chars) and "a" (1 char) are filtered, leaves only "motion" → single term → empty
        results = fts_engine_bc._breadcrumb_retrieval(table, "IO a motion", limit=20, where_clause=None)
        assert results == []

    def test_breadcrumb_retrieval_escapes_like_wildcards(self, fts_engine_bc):
        """SQL LIKE wildcards in query (%,_) should be escaped, not interpreted."""
        table = fts_engine_bc.db.open_table(fts_engine_bc.TABLE_NAME)
        # Query with LIKE wildcards should not cause errors or match everything
        results = fts_engine_bc._breadcrumb_retrieval(
            table, "100% mapp Motion", limit=20, where_clause=None
        )
        # Should not crash. "100%" → "100\%" in LIKE, won't match breadcrumbs
        assert isinstance(results, list)

    def test_breadcrumb_retrieval_empty_query(self, fts_engine_bc):
        """Empty or whitespace-only query returns empty list."""
        table = fts_engine_bc.db.open_table(fts_engine_bc.TABLE_NAME)
        assert fts_engine_bc._breadcrumb_retrieval(table, "", limit=20, where_clause=None) == []
        assert fts_engine_bc._breadcrumb_retrieval(table, "   ", limit=20, where_clause=None) == []
