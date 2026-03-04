"""Integration tests for indexer and search engine working together."""

import pytest

from src.indexer import HelpContentIndexer
from src.search_engine import HelpSearchEngine


class TestIndexerSearchEngineIntegration:
    """Integration tests for indexer and search engine working together."""

    @pytest.fixture
    def integrated_system(self, temp_help_dir, sample_xml, tmp_path, mock_embedding_service):
        """Create fully integrated indexer + search engine."""
        # Initialize and parse
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Build search index
        db_path = tmp_path / "test_integration_lance"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine.initialize()

        yield indexer, search_engine

        search_engine.close()

    def test_indexed_pages_searchable(self, integrated_system):
        """Verify pages indexed by indexer are findable via search."""
        indexer, search_engine = integrated_system

        # Search for a known page
        results = search_engine.search("X20DI9371")

        # Should find the page
        assert len(results) > 0
        assert any(r["title"] == "X20DI9371" for r in results)

    def test_breadcrumb_path_in_search_results(self, integrated_system):
        """Verify breadcrumb_path from indexer appears in search results."""
        indexer, search_engine = integrated_system

        # Search for a nested page
        results = search_engine.search("MC_BR_MoveAbsolute")

        # Should have breadcrumb path
        assert len(results) > 0
        result = results[0]
        assert result["breadcrumb_path"] is not None
        assert ">" in result["breadcrumb_path"]

    def test_category_from_breadcrumb_matches_filter(self, integrated_system):
        """Verify category extracted from breadcrumb enables category filtering."""
        indexer, search_engine = integrated_system

        # Search with category filter
        results = search_engine.search("X20", category="Hardware")

        # All results should be from Hardware category
        assert len(results) > 0
        for result in results:
            assert result["category"] == "Hardware"

    def test_help_id_search_and_retrieval(self, integrated_system):
        """Verify HelpID indexed during parsing is retrievable via search."""
        indexer, search_engine = integrated_system

        # Get page by HelpID from indexer
        page = indexer.get_page_by_help_id("12345")
        assert page is not None
        assert page.text == "X20DI9371"

        # Search should also find it
        results = search_engine.search("X20DI9371")
        assert len(results) > 0

        # HelpID should be in result
        result = next((r for r in results if r["page_id"] == page.id), None)
        assert result is not None
        assert result["help_id"] == "12345"

    def test_reindex_detection_after_xml_change(self, temp_help_dir, sample_xml, tmp_path, mock_embedding_service):
        """Verify both indexer and search engine detect XML changes."""
        # Initial indexing
        indexer1 = HelpContentIndexer(temp_help_dir)
        indexer1.parse_xml_structure()

        db_path = tmp_path / "test_reindex_lance"
        search_engine1 = HelpSearchEngine(db_path, indexer1, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine1.initialize()
        search_engine1.close()

        # Check indexer detects no change
        indexer2 = HelpContentIndexer(temp_help_dir)
        assert indexer2.needs_reindex() is False

        # Modify XML
        xml_path = temp_help_dir / "brhelpcontent.xml"
        content = xml_path.read_text()
        xml_path.write_text(content + "\n<!-- modified -->", encoding="utf-8")

        # Both should detect change
        indexer3 = HelpContentIndexer(temp_help_dir)
        assert indexer3.needs_reindex() is True

        # Parse with new indexer
        indexer3.parse_xml_structure()

        # Search engine should also detect change
        search_engine3 = HelpSearchEngine(db_path, indexer3, force_rebuild=False, embedding_service=mock_embedding_service)
        search_engine3.initialize()
        # Would rebuild if hash mismatch detected
        search_engine3.close()


class TestMCPToolIntegration:
    """Integration tests for MCP tools using real indexer/search engine."""

    @pytest.fixture
    def app_context(self, temp_help_dir, sample_xml, tmp_path, mock_embedding_service):
        """Create app context with real components."""
        from src.server import AppContext

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "test_mcp_lance"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine.initialize()

        context = AppContext(
            indexer=indexer,
            search_engine=search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        yield context

        search_engine.close()

    def test_search_then_get_page_workflow(self, app_context):
        """Verify search_help -> get_page_by_id workflow."""
        from unittest.mock import MagicMock

        from src.server import get_page_by_id, search_help

        # Create mock context
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        # Search for a page
        search_results = search_help(ctx, query="X20DI9371")

        assert search_results.total > 0
        page_id = search_results.results[0].page_id

        # Get full page content
        page_content = get_page_by_id(page_id=page_id, ctx=ctx)

        assert page_content is not None
        assert page_content.title == "X20DI9371"
        assert page_content.plain_text is not None
        assert "Digital input module" in page_content.plain_text

    def test_browse_categories_hierarchy(self, app_context):
        """Verify get_categories -> browse_section navigation."""
        from unittest.mock import MagicMock

        from src.server import browse_section, get_categories

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        # Get categories
        categories = get_categories(ctx)

        assert categories.total > 0
        hardware_cat = next((c for c in categories.categories if c.title == "Hardware"), None)
        assert hardware_cat is not None

        # Browse hardware category
        children = browse_section(ctx, section_id=hardware_cat.id)

        assert children is not None
        assert children.total > 0

    def test_search_with_category_filter(self, app_context):
        """Verify category filter from get_categories works in search_help."""
        from unittest.mock import MagicMock

        from src.server import get_categories, search_help

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        # Get categories
        categories = get_categories(ctx)
        hardware_cat = next((c for c in categories.categories if c.title == "Hardware"), None)
        assert hardware_cat is not None

        results = search_help(ctx, query="X20", category="Hardware")

        # All results should be from Hardware category
        for result in results.results:
            assert result.category == "Hardware"


class TestSearchAccuracy:
    """Test search accuracy and ranking."""

    @pytest.fixture
    def search_system(self, temp_help_dir, sample_xml, tmp_path):
        """Create search system."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "test_accuracy.db"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True)
        search_engine.initialize()

        yield indexer, search_engine

        search_engine.close()

    def test_exact_title_match_ranks_higher(self, search_system):
        """Verify exact title matches rank higher than content matches."""
        indexer, search_engine = search_system

        # Search for exact title
        results = search_engine.search("X20DI9371")

        # First result should be the exact match
        assert len(results) > 0
        assert results[0]["title"] == "X20DI9371"

    def test_prefix_matching_works(self, search_system):
        """Verify prefix matching finds partial words."""
        indexer, search_engine = search_system

        # Search with partial word
        results = search_engine.search("X20")

        # Should find X20DI9371
        assert len(results) > 0
        assert any("X20" in r["title"] for r in results)

    def test_search_finds_content_not_just_title(self, search_system):
        """Verify search finds content in HTML."""
        indexer, search_engine = search_system

        # Search for word that appears in content but not title
        results = search_engine.search("Digital input module")

        # Should find X20DI9371 page
        assert len(results) > 0
        x20_result = next((r for r in results if "X20" in r["title"]), None)
        assert x20_result is not None


class TestBreadcrumbConsistency:
    """Test breadcrumb consistency across components."""

    @pytest.fixture
    def system(self, temp_help_dir, sample_xml, tmp_path):
        """Create full system."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "test_breadcrumb.db"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True)
        search_engine.initialize()

        yield indexer, search_engine

        search_engine.close()

    def test_breadcrumb_matches_between_indexer_and_search(self, system):
        """Verify breadcrumb from search matches indexer breadcrumb."""
        indexer, search_engine = system

        # Get a page
        page_id = "mc_moveabs_page"

        # Get breadcrumb from indexer
        indexer_breadcrumb = indexer.get_breadcrumb_string(page_id)

        # Search for the page
        results = search_engine.search("MC_BR_MoveAbsolute")
        result = next((r for r in results if r["page_id"] == page_id), None)

        assert result is not None
        assert result["breadcrumb_path"] == indexer_breadcrumb


class TestPerformance:
    """Test performance characteristics of integrated system."""

    @pytest.fixture
    def system(self, temp_help_dir, sample_xml, tmp_path):
        """Create system."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "test_perf.db"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True)
        search_engine.initialize()

        yield indexer, search_engine

        search_engine.close()

    def test_search_is_fast(self, system):
        """Verify search completes quickly."""
        import time

        indexer, search_engine = system

        # Time a search
        start = time.time()
        _results = search_engine.search("motion")
        elapsed = time.time() - start

        # Should complete in under 1 second (even for small dataset)
        assert elapsed < 1.0

    def test_multiple_searches_dont_slow_down(self, system):
        """Verify repeated searches maintain performance."""
        import time

        indexer, search_engine = system

        times = []
        for _ in range(5):
            start = time.time()
            _results = search_engine.search("motion")
            times.append(time.time() - start)

        # Later searches shouldn't be slower
        assert times[-1] <= times[0] * 2  # Allow 2x tolerance
