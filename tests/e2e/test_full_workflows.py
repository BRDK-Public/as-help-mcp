"""End-to-end tests with sample help content."""

from unittest.mock import MagicMock

import pytest

from src.indexer import HelpContentIndexer
from src.search_engine import HelpSearchEngine
from src.server import AppContext, browse_section, get_categories, get_page_by_help_id, get_page_by_id, search_help


class TestEndToEnd:
    """End-to-end tests with sample help content."""

    @pytest.fixture
    def help_server(self, temp_help_dir, sample_xml, tmp_path, mock_embedding_service):
        """Create server with sample help content."""
        # Initialize indexer
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Build search index
        db_path = tmp_path / "e2e_test_lance"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine.initialize()

        # Create app context
        app_context = AppContext(
            indexer=indexer,
            search_engine=search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        # Create mock MCP context
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        yield ctx

        search_engine.close()

    def test_full_search_workflow(self, help_server):
        """Test: search -> get results -> retrieve page content."""
        # 1. Call search_help with query
        search_results = search_help(help_server, query="X20DI9371")

        # 2. Verify results contain expected pages
        assert search_results.total > 0
        assert any("X20DI9371" in r.title for r in search_results.results)

        # 3. Call get_page_by_id for top result
        top_result = search_results.results[0]
        page_content = get_page_by_id(page_id=top_result.page_id, ctx=help_server)

        # 4. Verify page content is returned
        assert page_content is not None
        assert page_content.title == top_result.title
        assert page_content.plain_text is not None
        assert len(page_content.plain_text) > 0

    def test_category_navigation_workflow(self, help_server):
        """Test: get_categories -> browse_section -> get_page_by_id."""
        # 1. Call get_categories
        categories = get_categories(help_server)

        # 2. Verify Hardware and Motion categories exist
        assert categories.total >= 2
        hardware = next((c for c in categories.categories if c.title == "Hardware"), None)
        motion = next((c for c in categories.categories if c.title == "Motion"), None)
        assert hardware is not None
        assert motion is not None

        # 3. Browse Hardware category
        hardware_children = browse_section(help_server, section_id=hardware.id)

        # 4. Verify children appear
        assert hardware_children is not None
        assert hardware_children.total > 0

        # 5. Get a page from children
        page_child = next((c for c in hardware_children.children if not c.is_section), None)
        if page_child:
            page_content = get_page_by_id(page_id=page_child.id, ctx=help_server)
            assert page_content is not None

    def test_help_id_lookup_workflow(self, help_server):
        """Test: get_page_by_help_id returns correct page."""
        # 1. Call get_page_by_help_id with known HelpID
        page_content = get_page_by_help_id(help_server, help_id="12345")

        # 2. Verify correct page is returned
        assert page_content is not None
        assert page_content.title == "X20DI9371"
        assert page_content.help_id == "12345"

    def test_breadcrumb_accuracy(self, help_server):
        """Test: breadcrumb correctly represents hierarchy."""
        # 1. Get page at motion/mapp_motion/mc_br_moveabsolute.html
        page_id = "mc_moveabs_page"
        page_content = get_page_by_id(page_id=page_id, include_breadcrumb=True, ctx=help_server)

        # 2. Verify breadcrumb is correct
        assert page_content is not None
        assert len(page_content.breadcrumb) == 3
        assert page_content.breadcrumb[0] == "Motion"
        assert page_content.breadcrumb[1] == "mapp Motion"
        assert page_content.breadcrumb[2] == "MC_BR_MoveAbsolute"

    def test_incremental_reindex(self, help_server, temp_help_dir, tmp_path, mock_embedding_service):
        """Test: modifying XML triggers reindex."""
        # 1. Verify initial index works
        initial_results = search_help(help_server, query="motion")
        assert initial_results.total > 0

        # 2. Modify brhelpcontent.xml (add new page)
        xml_path = temp_help_dir / "brhelpcontent.xml"
        content = xml_path.read_text()
        new_page = '<Page Id="new_page" Text="New Test Page" File="new.html"/>'
        content = content.replace("</BrHelpContent>", f"{new_page}</BrHelpContent>")
        xml_path.write_text(content, encoding="utf-8")

        # Create new HTML file
        (temp_help_dir / "new.html").write_text(
            """
            <html><head><title>New Page</title></head>
            <body><h1>New Page</h1><p>This is a new test page.</p></body></html>
        """,
            encoding="utf-8",
        )

        # 3. Create new server instance
        indexer = HelpContentIndexer(temp_help_dir)
        assert indexer.needs_reindex() is True  # Should detect change

        indexer.parse_xml_structure()

        db_path = tmp_path / "e2e_reindex_lance"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine.initialize()

        # 4. Verify new page is searchable
        app_context = AppContext(
            indexer=indexer,
            search_engine=search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        results = search_help(ctx, query="New Test Page")
        assert results.total > 0
        assert any("New Test Page" in r.title for r in results.results)

        search_engine.close()


class TestCompleteUserJourneys:
    """Test complete user journeys through the system."""

    @pytest.fixture
    def help_server(self, temp_help_dir, sample_xml, tmp_path, mock_embedding_service):
        """Create server with sample help content."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "journey_test_lance"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine.initialize()

        app_context = AppContext(
            indexer=indexer,
            search_engine=search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        yield ctx

        search_engine.close()

    def test_hardware_specialist_journey(self, help_server):
        """Simulate a hardware specialist looking for module information."""
        # User searches for specific module
        results = search_help(help_server, query="X20DI9371")
        assert results.total > 0

        # User selects top result and reads full documentation
        page = get_page_by_id(page_id=results.results[0].page_id, ctx=help_server)
        assert page is not None
        assert "Digital input module" in page.plain_text

        # User checks online help URL
        assert page.online_help_url is not None
        assert "hardware" in page.online_help_url.lower()

    def test_motion_engineer_journey(self, help_server):
        """Simulate a motion engineer looking for function block docs."""
        # User browses categories
        categories = get_categories(help_server)
        motion_cat = next((c for c in categories.categories if c.title == "Motion"), None)
        assert motion_cat is not None

        # User explores motion category
        motion_children = browse_section(help_server, section_id=motion_cat.id)
        assert motion_children is not None

        # User finds mapp Motion section
        mapp_section = next((c for c in motion_children.children if "mapp" in c.title.lower()), None)
        assert mapp_section is not None

        # User browses deeper
        mapp_children = browse_section(help_server, section_id=mapp_section.id)
        assert mapp_children is not None

        # User searches for specific FB
        results = search_help(help_server, query="MC_BR_MoveAbsolute", category="Motion")
        assert results.total > 0

        # User reads documentation
        page = get_page_by_id(page_id=results.results[0].page_id, ctx=help_server)
        assert page is not None
        assert "absolute position" in page.plain_text.lower()

    def test_help_context_lookup_journey(self, help_server):
        """Simulate context-sensitive help lookup via HelpID."""
        # Application calls help with HelpID (from F1 key press)
        page = get_page_by_help_id(help_server, help_id="12345")
        assert page is not None
        assert page.title == "X20DI9371"

        # Help window shows breadcrumb for navigation
        assert len(page.breadcrumb) > 0
        assert page.breadcrumb[0] == "Hardware"

    def test_search_refinement_journey(self, help_server):
        """Simulate user refining search with different keywords."""
        # Initial broad search
        results1 = search_help(help_server, query="motion")
        initial_count = results1.total

        # Refine with more specific term
        results2 = search_help(help_server, query="move absolute")
        assert results2.total <= initial_count

        # Further refine with category
        results3 = search_help(help_server, query="move", category="Motion")
        assert results3.total > 0

        # All results should be from Motion category
        for result in results3.results:
            assert result.category == "Motion"


class TestErrorRecovery:
    """Test error handling and recovery in E2E scenarios."""

    @pytest.fixture
    def help_server(self, temp_help_dir, sample_xml, tmp_path, mock_embedding_service):
        """Create server with sample help content."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "error_test_lance"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine.initialize()

        app_context = AppContext(
            indexer=indexer,
            search_engine=search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        yield ctx

        search_engine.close()

    def test_nonexistent_page_id(self, help_server):
        """Test handling of non-existent page ID."""
        page = get_page_by_id(page_id="nonexistent_id", ctx=help_server)
        assert page is None

    def test_nonexistent_help_id(self, help_server):
        """Test handling of non-existent HelpID."""
        page = get_page_by_help_id(help_server, help_id="99999")
        assert page is None

    def test_nonexistent_section_id(self, help_server):
        """Test handling of non-existent section ID."""
        children = browse_section(help_server, section_id="nonexistent_section")
        assert children is None

    def test_empty_search_query(self, help_server):
        """Test handling of empty search query."""
        results = search_help(help_server, query="")
        assert results.total == 0
        assert len(results.results) == 0

    def test_search_no_results(self, help_server):
        """Test handling of search with no results."""
        results = search_help(help_server, query="zzzznonexistentzzzzz")
        # With hybrid search, vector search may return low-score results
        # for nonsensical queries, but FTS won't contribute
        for result in results.results:
            assert result.score < 0.1  # Low relevance scores expected


class TestDataConsistency:
    """Test data consistency across all operations."""

    @pytest.fixture
    def help_server(self, temp_help_dir, sample_xml, tmp_path, mock_embedding_service):
        """Create server with sample help content."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "consistency_test_lance"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine.initialize()

        app_context = AppContext(
            indexer=indexer,
            search_engine=search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        yield ctx

        search_engine.close()

    def test_page_id_consistency(self, help_server):
        """Verify page_id is consistent across search and retrieval."""
        # Search for page
        results = search_help(help_server, query="X20DI9371")
        assert results.total > 0

        page_id = results.results[0].page_id

        # Retrieve by page_id
        page = get_page_by_id(page_id=page_id, ctx=help_server)
        assert page is not None
        assert page.page_id == page_id

    def test_help_id_consistency(self, help_server):
        """Verify HelpID is consistent across operations."""
        # Get page by HelpID
        page_by_help_id = get_page_by_help_id(help_server, help_id="12345")
        assert page_by_help_id is not None

        # Get same page by page_id
        page_by_id = get_page_by_id(page_id=page_by_help_id.page_id, ctx=help_server)
        assert page_by_id is not None

        # Should be same page
        assert page_by_help_id.title == page_by_id.title
        assert page_by_help_id.help_id == page_by_id.help_id

    def test_breadcrumb_consistency(self, help_server):
        """Verify breadcrumb is consistent across search and page retrieval."""
        # Search for deeply nested page
        results = search_help(help_server, query="MC_BR_MoveAbsolute")
        assert results.total > 0

        search_breadcrumb = results.results[0].breadcrumb_path

        # Get full page
        page = get_page_by_id(page_id=results.results[0].page_id, include_breadcrumb=True, ctx=help_server)
        page_breadcrumb = " > ".join(page.breadcrumb)

        # Should match
        assert search_breadcrumb == page_breadcrumb


@pytest.mark.slow
class TestLargeDataset:
    """Test with larger synthetic dataset (marked as slow)."""

    @pytest.fixture
    def large_help_server(self, temp_help_dir, tmp_path, mock_embedding_service):
        """Create server with larger synthetic dataset."""
        # Generate 100 pages across 10 categories
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>\n<BrHelpContent>']

        for cat_num in range(10):
            cat_id = f"cat_{cat_num}"
            cat_name = f"Category {cat_num}"
            xml_parts.append(f'  <Section Id="{cat_id}" Text="{cat_name}" File="cat{cat_num}.html">')

            # Create HTML for category
            cat_html = f"<html><body><h1>{cat_name}</h1></body></html>"
            (temp_help_dir / f"cat{cat_num}.html").write_text(cat_html, encoding="utf-8")

            for page_num in range(10):
                page_id = f"page_{cat_num}_{page_num}"
                page_name = f"Page {cat_num}-{page_num}"
                page_file = f"page_{cat_num}_{page_num}.html"

                xml_parts.append(f'    <Page Id="{page_id}" Text="{page_name}" File="{page_file}">')
                xml_parts.append(f'      <Identifiers><HelpID Value="{cat_num * 100 + page_num}"/></Identifiers>')
                xml_parts.append("    </Page>")

                # Create HTML for page
                page_html = f"<html><body><h1>{page_name}</h1><p>Content for {page_name}</p></body></html>"
                (temp_help_dir / page_file).write_text(page_html, encoding="utf-8")

            xml_parts.append("  </Section>")

        xml_parts.append("</BrHelpContent>")

        xml_path = temp_help_dir / "brhelpcontent.xml"
        xml_path.write_text("\n".join(xml_parts), encoding="utf-8")

        # Index the data
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        db_path = tmp_path / "large_test_lance"
        search_engine = HelpSearchEngine(db_path, indexer, force_rebuild=True, embedding_service=mock_embedding_service)
        search_engine.initialize()

        app_context = AppContext(
            indexer=indexer,
            search_engine=search_engine,
            as_version="4",
            online_help_base_url="https://help.br-automation.com/#/en/4/",
        )

        ctx = MagicMock()
        ctx.request_context.lifespan_context = app_context

        yield ctx

        search_engine.close()

    def test_large_dataset_search(self, large_help_server):
        """Test search performance with larger dataset."""
        results = search_help(large_help_server, query="Page")

        # Should find many pages
        assert results.total > 0

    def test_large_dataset_pagination(self, large_help_server):
        """Test search with different limits."""
        results_5 = search_help(large_help_server, query="Page", limit=5)
        results_10 = search_help(large_help_server, query="Page", limit=10)

        assert len(results_5.results) <= 5
        assert len(results_10.results) <= 10
