"""Unit tests for server.py - MCP tool implementations."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.server import (
    AppContext,
    browse_section,
    get_as_version_config,
    get_breadcrumb,
    get_categories,
    get_help_page_resource,
    get_help_statistics,
    get_page_by_help_id,
    get_page_by_id,
    get_page_html,
    search_help,
)


class TestASVersionConfiguration:
    """Test AS version configuration from environment variables."""

    def test_get_as_version_config_default(self):
        """Verify default is AS4 when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            version, url = get_as_version_config()
            assert version == "4"
            assert url == "https://help.br-automation.com/#/en/4/"

    def test_get_as_version_config_version_6(self):
        """Verify AS6 configuration when env var is '6'."""
        with patch.dict(os.environ, {"AS_HELP_VERSION": "6"}):
            version, url = get_as_version_config()
            assert version == "6"
            assert url == "https://help.br-automation.com/#/en/6/"

    def test_get_as_version_config_strips_whitespace(self):
        """Verify whitespace is stripped from env var."""
        with patch.dict(os.environ, {"AS_HELP_VERSION": "  6  "}):
            version, url = get_as_version_config()
            assert version == "6"
            assert url == "https://help.br-automation.com/#/en/6/"

    def test_get_as_version_config_falls_back_to_4(self):
        """Verify unknown version falls back to AS4."""
        with patch.dict(os.environ, {"AS_HELP_VERSION": "5"}):
            version, url = get_as_version_config()
            assert version == "4"
            assert url == "https://help.br-automation.com/#/en/4/"


class TestSearchHelpTool:
    """Test search_help MCP tool."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context with indexer and search engine."""
        mock_search_engine = MagicMock()

        # Mock search results
        mock_search_engine.search.return_value = [
            {
                "page_id": "page1",
                "title": "Test Page",
                "file_path": "test.html",
                "help_id": "12345",
                "is_section": False,
                "breadcrumb_path": "Test Section > Test Page",
                "category": "Test",
                "score": 10.5,
                "snippet": "Test snippet",
            }
        ]

        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=mock_search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        # Create mock MCP context
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        return ctx

    def test_search_help_truncates_preview(self, mock_context):
        """Verify content_preview is truncated to ~100 chars."""
        # Add plain text to mock indexer
        mock_context.request_context.lifespan_context.indexer.extract_plain_text.return_value = "A" * 500

        result = search_help(mock_context, query="test")

        assert len(result.results) > 0
        if result.results[0].content_preview:
            assert len(result.results[0].content_preview) < 200
            assert "[TRUNCATED" in result.results[0].content_preview

    def test_search_help_builds_online_url(self, mock_context):
        """Verify online_help_url is constructed from file_path."""
        # Mock extract_plain_text to return None (for sections or pages without content)
        mock_context.request_context.lifespan_context.indexer.extract_plain_text.return_value = None

        result = search_help(mock_context, query="test")

        assert len(result.results) > 0
        assert result.results[0].online_help_url is not None
        assert "https://help.br-automation.com/#/en/4/test.html" == result.results[0].online_help_url

    def test_search_help_normalizes_path_separators(self, mock_context):
        """Verify backslashes in file_path are converted to forward slashes."""
        # Modify mock to return backslashes
        mock_context.request_context.lifespan_context.search_engine.search.return_value = [
            {
                "page_id": "page1",
                "title": "Test",
                "file_path": "motion\\axis.html",
                "help_id": None,
                "is_section": False,
                "breadcrumb_path": "Motion > Axis",
                "category": "Motion",
                "score": 10.0,
                "snippet": "test",
            }
        ]

        # Mock extract_plain_text to return None
        mock_context.request_context.lifespan_context.indexer.extract_plain_text.return_value = None

        result = search_help(mock_context, query="test")

        assert len(result.results) > 0
        assert "motion/axis.html" in result.results[0].online_help_url
        assert "\\" not in result.results[0].online_help_url

    def test_search_help_sections_no_preview(self, mock_context, mock_indexer):
        """Verify sections (is_section=True) have no content_preview."""
        # Mock section result
        mock_context.request_context.lifespan_context.search_engine.search.return_value = [
            {
                "page_id": "section1",
                "title": "Test Section",
                "file_path": "section.html",
                "help_id": None,
                "is_section": True,
                "breadcrumb_path": "Test Section",
                "category": "Test",
                "score": 10.0,
                "snippet": None,
            }
        ]

        result = search_help(mock_context, query="test")

        assert len(result.results) > 0
        assert result.results[0].content_preview is None


class TestGetPageByIDTool:
    """Test get_page_by_id MCP tool."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context."""
        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=MagicMock(),
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        # Mock content extraction
        mock_indexer.extract_html_content.return_value = "<html>Test HTML</html>"
        mock_indexer.extract_plain_text.return_value = "Test plain text"

        return ctx

    def test_get_page_by_id_include_flags(self, mock_context):
        """Verify include_html, include_text, include_breadcrumb flags work."""
        # Test with all flags
        result = get_page_by_id(
            page_id="page1", include_html=True, include_text=True, include_breadcrumb=True, ctx=mock_context
        )

        assert result is not None
        assert result.html_content == "<html>Test HTML</html>"
        assert result.plain_text == "Test plain text"
        assert len(result.breadcrumb) > 0

    def test_get_page_by_id_not_found(self, mock_context, mock_indexer):
        """Verify None returned for non-existent page_id."""
        mock_indexer.get_page_by_id.return_value = None

        result = get_page_by_id(page_id="nonexistent", ctx=mock_context)
        assert result is None

    def test_get_page_by_id_url_building(self, mock_context):
        """Verify online_help_url is correctly built."""
        result = get_page_by_id(page_id="page1", ctx=mock_context)

        assert result is not None
        assert result.online_help_url is not None
        assert "https://help.br-automation.com/#/en/4/" in result.online_help_url


class TestGetPageByHelpIDTool:
    """Test get_page_by_help_id MCP tool."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context."""
        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=MagicMock(),
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        # Mock content extraction
        mock_indexer.extract_html_content.return_value = "<html>Test</html>"
        mock_indexer.extract_plain_text.return_value = "Test"

        return ctx

    def test_get_page_by_help_id_lookup(self, mock_context, mock_indexer):
        """Verify HelpID is resolved to page_id via indexer."""
        result = get_page_by_help_id(mock_context, help_id="12345")

        assert result is not None
        assert result.page_id == "page1"
        assert result.title == "Test Page"

    def test_get_page_by_help_id_not_found(self, mock_context, mock_indexer):
        """Verify None returned for non-existent HelpID."""
        mock_indexer.get_page_by_help_id.return_value = None

        result = get_page_by_help_id(mock_context, help_id="99999")
        assert result is None


class TestGetCategoriesTool:
    """Test get_categories MCP tool."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context."""
        # Mock categories
        mock_indexer.get_top_level_categories.return_value = [
            {"id": "hardware", "title": "Hardware", "file_path": "hardware.html"},
            {"id": "motion", "title": "Motion", "file_path": "motion.html"},
        ]

        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=MagicMock(),
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        return ctx

    def test_get_categories_returns_all(self, mock_context):
        """Verify all categories are returned."""
        result = get_categories(mock_context)

        assert result.total == 2
        assert len(result.categories) == 2
        assert result.categories[0].title == "Hardware"
        assert result.categories[1].title == "Motion"


class TestBrowseSectionTool:
    """Test browse_section MCP tool."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context."""
        # Mock section children
        mock_indexer.get_section_children.return_value = [
            {"id": "child1", "title": "Child 1", "file_path": "c1.html", "is_section": True},
            {"id": "child2", "title": "Child 2", "file_path": "c2.html", "is_section": False},
        ]

        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=MagicMock(),
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        return ctx

    def test_browse_section_returns_children(self, mock_context):
        """Verify children are returned."""
        result = browse_section(mock_context, section_id="section1")

        assert result is not None
        assert result.total == 2
        assert len(result.children) == 2
        assert result.children[0].is_section is True
        assert result.children[1].is_section is False

    def test_browse_section_nonexistent(self, mock_context, mock_indexer):
        """Verify None returned for non-existent section."""
        mock_indexer.get_page_by_id.return_value = None

        result = browse_section(mock_context, section_id="nonexistent")
        assert result is None


class TestGetBreadcrumbTool:
    """Test get_breadcrumb MCP tool."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context."""
        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=MagicMock(),
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        return ctx

    @pytest.mark.asyncio
    async def test_get_breadcrumb_returns_path(self, mock_context):
        """Verify breadcrumb path is returned."""
        result = await get_breadcrumb(mock_context, page_id="page1")

        assert len(result) == 2
        assert result[0].title == "Test Section"
        assert result[1].title == "Test Page"


class TestGetHelpStatisticsTool:
    """Test get_help_statistics MCP tool."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context."""
        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=MagicMock(),
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        from unittest.mock import AsyncMock

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context
        ctx.info = AsyncMock()  # Mock async info method

        return ctx

    @pytest.mark.asyncio
    async def test_get_help_statistics_counts(self, mock_context):
        """Verify all statistics are calculated correctly."""
        result = await get_help_statistics(mock_context)

        assert "total_pages" in result
        assert "total_sections" in result
        assert "regular_pages" in result
        assert "help_id_mappings" in result
        assert "pages_with_parents" in result
        assert "root_items" in result

        assert result["total_pages"] == 3  # from mock_indexer
        assert result["help_id_mappings"] == 1

    @pytest.mark.asyncio
    async def test_get_help_statistics_regular_pages_calculation(self, mock_context):
        """Verify regular_pages = total_pages - total_sections."""
        result = await get_help_statistics(mock_context)

        assert result["regular_pages"] == result["total_pages"] - result["total_sections"]


class TestURLBuilding:
    """Test URL building logic across tools."""

    def test_url_with_as_version_4(self):
        """Verify URLs are built correctly for AS4."""
        with patch.dict(os.environ, {"AS_HELP_VERSION": "4"}):
            version, base_url = get_as_version_config()
            file_path = "motion/axis.html"

            expected_url = f"{base_url}{file_path}"
            assert expected_url == "https://help.br-automation.com/#/en/4/motion/axis.html"

    def test_url_with_as_version_6(self):
        """Verify URLs are built correctly for AS6."""
        with patch.dict(os.environ, {"AS_HELP_VERSION": "6"}):
            version, base_url = get_as_version_config()
            file_path = "motion/axis.html"

            expected_url = f"{base_url}{file_path}"
            assert expected_url == "https://help.br-automation.com/#/en/6/motion/axis.html"


class TestSearchResultTransformation:
    """Test search result transformation logic."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context."""
        mock_search_engine = MagicMock()
        mock_search_engine.search.return_value = []

        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=mock_search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        return ctx

    def test_empty_search_results(self, mock_context):
        """Verify empty search returns empty SearchResults."""
        result = search_help(mock_context, query="nonexistent")

        assert result.total == 0
        assert len(result.results) == 0
        assert result.query == "nonexistent"


class TestResourceHandlers:
    """Test MCP resource handlers."""

    @pytest.fixture
    def mock_context(self, mock_indexer):
        """Create mock context."""
        app_context = AppContext(
            indexer=mock_indexer,
            search_engine=MagicMock(),
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        return ctx

    def test_get_help_page_resource_returns_text(self, mock_context, mock_indexer):
        """Verify get_help_page_resource returns plain text content."""
        mock_indexer.extract_plain_text.return_value = "This is plain text content"
        mock_indexer.extract_html_content.return_value = "<html>HTML</html>"

        result = get_help_page_resource(page_id="page1", ctx=mock_context)

        assert result == "This is plain text content"

    def test_get_help_page_resource_fallback_to_html(self, mock_context, mock_indexer):
        """Verify get_help_page_resource falls back to HTML when text is empty."""
        mock_indexer.extract_plain_text.return_value = None
        mock_indexer.extract_html_content.return_value = "<html>Fallback HTML</html>"

        result = get_help_page_resource(page_id="page1", ctx=mock_context)

        assert result == "<html>Fallback HTML</html>"

    def test_get_help_page_resource_not_found_raises(self, mock_context, mock_indexer):
        """Verify get_help_page_resource raises ValueError for missing page."""
        mock_indexer.extract_plain_text.return_value = None
        mock_indexer.extract_html_content.return_value = None

        with pytest.raises(ValueError, match="Page nonexistent not found"):
            get_help_page_resource(page_id="nonexistent", ctx=mock_context)

    def test_get_page_html_returns_html(self, mock_context, mock_indexer):
        """Verify get_page_html returns HTML content."""
        mock_indexer.extract_html_content.return_value = "<html><body>Content</body></html>"

        result = get_page_html(page_id="page1", ctx=mock_context)

        assert result == "<html><body>Content</body></html>"

    def test_get_page_html_not_found_returns_message(self, mock_context, mock_indexer):
        """Verify get_page_html returns error message for missing page."""
        mock_indexer.extract_html_content.return_value = None

        result = get_page_html(page_id="nonexistent", ctx=mock_context)

        assert result == "Page not found: nonexistent"
