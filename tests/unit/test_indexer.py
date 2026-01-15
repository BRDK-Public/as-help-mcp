"""Unit tests for indexer.py - XML parsing and breadcrumb logic."""

import json

from src.indexer import HelpContentIndexer


class TestXMLAttributeExtraction:
    """Test XML attribute extraction for both full and abbreviated formats."""

    def test_process_section_full_format(self, temp_help_dir, sample_xml):
        """Verify _process_section handles full XML attribute format."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check that hardware section was parsed
        assert "hardware_section" in indexer.pages
        section = indexer.pages["hardware_section"]
        assert section.text == "Hardware"
        assert section.file_path == "index.html"
        assert section.is_section is True

    def test_process_page_full_format(self, temp_help_dir, sample_xml):
        """Verify _process_page handles full XML attribute format."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check that x20 page was parsed
        assert "x20di9371_page" in indexer.pages
        page = indexer.pages["x20di9371_page"]
        assert page.text == "X20DI9371"
        assert page.file_path == "hardware/x20di9371.html"
        assert page.is_section is False
        assert page.parent_id == "hardware_section"

    def test_process_section_abbreviated_format(self, temp_help_dir, sample_xml_abbreviated):
        """Verify _process_section handles abbreviated XML format."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check that hardware section was parsed with abbreviated tags
        assert "hardware_section" in indexer.pages
        section = indexer.pages["hardware_section"]
        assert section.text == "Hardware"
        assert section.file_path == "index.html"
        assert section.is_section is True

    def test_process_page_abbreviated_format(self, temp_help_dir, sample_xml_abbreviated):
        """Verify _process_page handles abbreviated XML format."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check that x20 page was parsed with abbreviated tags
        assert "x20di9371_page" in indexer.pages
        page = indexer.pages["x20di9371_page"]
        assert page.text == "X20DI9371"
        assert page.file_path == "hardware/x20di9371.html"
        assert page.is_section is False


class TestHelpIDExtraction:
    """Test HelpID extraction from XML."""

    def test_help_id_extraction_full_format(self, temp_help_dir, sample_xml):
        """Verify HelpID extraction with full format."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check page with HelpID
        page = indexer.pages["x20di9371_page"]
        assert page.help_id == "12345"

    def test_help_id_extraction_abbreviated_format(self, temp_help_dir, sample_xml_abbreviated):
        """Verify HelpID extraction with abbreviated format."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check page with HelpID (abbreviated)
        page = indexer.pages["x20di9371_page"]
        assert page.help_id == "12345"

    def test_help_id_mapping(self, temp_help_dir, sample_xml):
        """Verify help_id_map is populated correctly."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Verify mapping
        assert "12345" in indexer.help_id_map
        assert indexer.help_id_map["12345"] == "x20di9371_page"

        assert "20000" in indexer.help_id_map
        assert indexer.help_id_map["20000"] == "motion_section"

        assert "20100" in indexer.help_id_map
        assert indexer.help_id_map["20100"] == "mc_moveabs_page"

    def test_get_page_by_help_id(self, temp_help_dir, sample_xml):
        """Verify get_page_by_help_id returns correct page."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        page = indexer.get_page_by_help_id("12345")
        assert page is not None
        assert page.id == "x20di9371_page"
        assert page.text == "X20DI9371"

    def test_get_page_by_help_id_not_found(self, temp_help_dir, sample_xml):
        """Verify get_page_by_help_id returns None for non-existent HelpID."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        page = indexer.get_page_by_help_id("99999")
        assert page is None


class TestBreadcrumbComputation:
    """Test breadcrumb computation logic."""

    def test_breadcrumb_simple_hierarchy(self, temp_help_dir, sample_xml):
        """Verify breadcrumb for a 3-level hierarchy."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Get breadcrumb for deeply nested page
        breadcrumb = indexer.get_breadcrumb("mc_moveabs_page")
        assert len(breadcrumb) == 3
        assert breadcrumb[0].text == "Motion"
        assert breadcrumb[1].text == "mapp Motion"
        assert breadcrumb[2].text == "MC_BR_MoveAbsolute"

    def test_breadcrumb_cycle_detection(self, temp_help_dir):
        """Verify breadcrumb stops on cycle."""
        # Create XML with duplicate ID causing apparent cycle
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="section1" Text="Section 1" File="s1.html">
        <Page Id="page1" Text="Page 1" File="p1.html"/>
    </Section>
    <Section Id="section1" Text="Section 1 Duplicate" File="s2.html">
        <Page Id="page2" Text="Page 2" File="p2.html"/>
    </Section>
</BrHelpContent>
"""
        xml_path = temp_help_dir / "brhelpcontent.xml"
        xml_path.write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Breadcrumb should terminate without infinite loop
        # The second occurrence of section1 overwrites the first
        breadcrumb = indexer.get_breadcrumb("page2")
        assert len(breadcrumb) < 100  # Should not hit depth limit

    def test_breadcrumb_depth_limit(self, temp_help_dir):
        """Verify breadcrumb stops at 100 levels."""
        # Create deeply nested hierarchy
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>\n<BrHelpContent>']

        # Create 150-level deep hierarchy
        for i in range(150):
            indent = "  " * (i + 1)
            xml_parts.append(f'{indent}<Section Id="s{i}" Text="Section {i}" File="s{i}.html">')

        # Close all sections
        for i in range(149, -1, -1):
            indent = "  " * (i + 1)
            xml_parts.append(f"{indent}</Section>")

        xml_parts.append("</BrHelpContent>")
        xml_content = "\n".join(xml_parts)

        xml_path = temp_help_dir / "brhelpcontent.xml"
        xml_path.write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Breadcrumb should be limited to 100 levels
        breadcrumb = indexer.get_breadcrumb("s149")
        assert len(breadcrumb) < 150  # Should stop before reaching all 150 levels

    def test_breadcrumb_missing_parent(self, temp_help_dir):
        """Verify breadcrumb handles missing parent gracefully."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Page Id="orphan_page" Text="Orphan Page" File="orphan.html"/>
</BrHelpContent>
"""
        xml_path = temp_help_dir / "brhelpcontent.xml"
        xml_path.write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Manually set invalid parent_id
        indexer.pages["orphan_page"].parent_id = "nonexistent"

        # Breadcrumb should only contain the page itself
        breadcrumb = indexer._compute_breadcrumb("orphan_page")
        assert len(breadcrumb) == 1
        assert breadcrumb[0].text == "Orphan Page"

    def test_breadcrumb_string_format(self, temp_help_dir, sample_xml):
        """Verify get_breadcrumb_string returns ' > ' separated path."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        breadcrumb_str = indexer.get_breadcrumb_string("mc_moveabs_page")
        assert breadcrumb_str == "Motion > mapp Motion > MC_BR_MoveAbsolute"


class TestCategoryAndChildrenFunctions:
    """Test category and children retrieval functions."""

    def test_get_top_level_categories(self, temp_help_dir, sample_xml):
        """Verify only root sections are returned."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        categories = indexer.get_top_level_categories()

        # Should have 2 root sections
        assert len(categories) == 2

        # Verify they're all sections with no parent
        for cat in categories:
            page = indexer.pages[cat["id"]]
            assert page.is_section is True
            assert page.parent_id is None

    def test_get_top_level_categories_sorted(self, temp_help_dir, sample_xml):
        """Verify categories are sorted alphabetically."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        categories = indexer.get_top_level_categories()
        titles = [cat["title"] for cat in categories]

        # Should be sorted: Hardware, Motion
        assert titles == sorted(titles, key=str.lower)

    def test_get_section_children_separates_sections_and_pages(self, temp_help_dir, sample_xml):
        """Verify sections listed first, then pages."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Motion section has both child sections and pages
        children = indexer.get_section_children("motion_section")

        # First child should be a section
        assert len(children) > 0
        assert children[0]["is_section"] is True
        assert children[0]["title"] == "mapp Motion"

    def test_get_section_children_nonexistent_section(self, temp_help_dir, sample_xml):
        """Verify empty list returned for non-existent section."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        children = indexer.get_section_children("nonexistent")
        assert children == []


class TestDuplicateIDHandling:
    """Test duplicate ID detection and handling."""

    def test_duplicate_id_detection(self, temp_help_dir):
        """Verify duplicate IDs are tracked in _duplicate_ids dict."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="dup_id" Text="First Instance" File="first.html"/>
    <Section Id="dup_id" Text="Second Instance" File="second.html"/>
</BrHelpContent>
"""
        xml_path = temp_help_dir / "brhelpcontent.xml"
        xml_path.write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check that duplicate was detected
        assert "dup_id" in indexer._duplicate_ids
        assert len(indexer._duplicate_ids["dup_id"]) == 2
        assert "First Instance" in indexer._duplicate_ids["dup_id"]
        assert "Second Instance" in indexer._duplicate_ids["dup_id"]

    def test_duplicate_id_does_not_crash(self, temp_help_dir):
        """Verify parsing continues when duplicate IDs encountered."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="dup_id" Text="First" File="first.html">
        <Page Id="page1" Text="Page 1" File="p1.html"/>
    </Section>
    <Section Id="dup_id" Text="Second" File="second.html">
        <Page Id="page2" Text="Page 2" File="p2.html"/>
    </Section>
</BrHelpContent>
"""
        xml_path = temp_help_dir / "brhelpcontent.xml"
        xml_path.write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        # Should not raise exception
        indexer.parse_xml_structure()

        # Both child pages should be parsed
        assert "page1" in indexer.pages
        assert "page2" in indexer.pages

    def test_duplicate_page_id_detection(self, temp_help_dir):
        """Verify duplicate page IDs are tracked in _duplicate_ids dict."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="section1" Text="Section 1" File="section1.html">
        <Page Id="dup_page_id" Text="First Page" File="first_page.html"/>
        <Page Id="dup_page_id" Text="Second Page" File="second_page.html"/>
    </Section>
</BrHelpContent>
"""
        xml_path = temp_help_dir / "brhelpcontent.xml"
        xml_path.write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check that duplicate was detected
        assert "dup_page_id" in indexer._duplicate_ids
        assert len(indexer._duplicate_ids["dup_page_id"]) == 2
        assert "First Page" in indexer._duplicate_ids["dup_page_id"]
        assert "Second Page" in indexer._duplicate_ids["dup_page_id"]

    def test_duplicate_page_id_does_not_crash(self, temp_help_dir):
        """Verify parsing continues when duplicate page IDs encountered."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="section1" Text="Section 1" File="section1.html">
        <Page Id="dup_page_id" Text="First Page" File="first.html"/>
    </Section>
    <Section Id="section2" Text="Section 2" File="section2.html">
        <Page Id="dup_page_id" Text="Duplicate Page" File="dup.html"/>
        <Page Id="page2" Text="Page 2" File="p2.html"/>
    </Section>
</BrHelpContent>
"""
        xml_path = temp_help_dir / "brhelpcontent.xml"
        xml_path.write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        # Should not raise exception
        indexer.parse_xml_structure()

        # Both sections and the unique page should be parsed
        assert "section1" in indexer.pages
        assert "section2" in indexer.pages
        assert "page2" in indexer.pages
        # The duplicate page ID exists (last occurrence wins)
        assert "dup_page_id" in indexer.pages


class TestMetadataOperations:
    """Test metadata loading and saving."""

    def test_needs_reindex_no_metadata(self, temp_help_dir, sample_xml):
        """Verify needs_reindex returns True when no metadata exists."""
        indexer = HelpContentIndexer(temp_help_dir)

        # Before parsing, no metadata exists
        assert indexer.needs_reindex() is True

    def test_needs_reindex_hash_match(self, temp_help_dir, sample_xml):
        """Verify needs_reindex returns False when hash matches."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # After parsing, metadata is saved
        # Create new indexer to check
        indexer2 = HelpContentIndexer(temp_help_dir)
        assert indexer2.needs_reindex() is False

    def test_needs_reindex_hash_mismatch(self, temp_help_dir, sample_xml):
        """Verify needs_reindex returns True when hash differs."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Modify XML file
        xml_path = temp_help_dir / "brhelpcontent.xml"
        content = xml_path.read_text()
        xml_path.write_text(content + "\n<!-- modified -->", encoding="utf-8")

        # Create new indexer - should detect change
        indexer2 = HelpContentIndexer(temp_help_dir)
        assert indexer2.needs_reindex() is True

    def test_save_metadata_content(self, temp_help_dir, sample_xml):
        """Verify _save_metadata writes correct JSON structure."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Check metadata file exists
        metadata_path = indexer.metadata_path
        assert metadata_path.exists()

        # Load and verify content
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert "xml_hash" in metadata
        assert "indexed_at" in metadata
        assert "page_count" in metadata
        assert "help_id_count" in metadata
        assert metadata["page_count"] > 0
        assert metadata["help_id_count"] > 0

    def test_get_xml_hash_consistency(self, temp_help_dir, sample_xml):
        """Verify _get_xml_hash returns consistent hash for same file."""
        indexer = HelpContentIndexer(temp_help_dir)

        hash1 = indexer._get_xml_hash()
        hash2 = indexer._get_xml_hash()

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length


class TestContentExtraction:
    """Test HTML and plain text extraction."""

    def test_extract_html_content(self, temp_help_dir, sample_xml):
        """Verify HTML content extraction."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        html = indexer.extract_html_content("x20di9371_page")
        assert html is not None
        assert "X20DI9371" in html
        assert "Digital input module" in html

    def test_extract_html_content_cached(self, temp_help_dir, sample_xml):
        """Verify HTML content is cached."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # First extraction
        html1 = indexer.extract_html_content("x20di9371_page")

        # Second extraction should return cached
        page = indexer.pages["x20di9371_page"]
        assert page.html_content is not None

        html2 = indexer.extract_html_content("x20di9371_page")
        assert html1 is html2  # Same object reference

    def test_extract_plain_text(self, temp_help_dir, sample_xml):
        """Verify plain text extraction."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        text = indexer.extract_plain_text("x20di9371_page")
        assert text is not None
        assert "X20DI9371" in text
        assert "Digital input module" in text
        # Should not contain HTML tags
        assert "<" not in text
        assert ">" not in text

    def test_extract_plain_text_removes_script_style(self, temp_help_dir):
        """Verify script and style tags are removed."""
        # Create HTML with script/style
        html_content = """<html>
<head>
    <style>body { color: red; }</style>
    <script>console.log('test');</script>
</head>
<body>
    <h1>Title</h1>
    <p>Content</p>
</body>
</html>"""

        (temp_help_dir / "test.html").write_text(html_content, encoding="utf-8")

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Page Id="test_page" Text="Test" File="test.html"/>
</BrHelpContent>
"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        text = indexer.extract_plain_text("test_page")
        assert text is not None
        # Should not contain script/style content
        assert "console.log" not in text
        assert "color: red" not in text
        # Should contain actual content
        assert "Title" in text
        assert "Content" in text

    def test_extract_text_for_page_no_cache(self, temp_help_dir, sample_xml):
        """Verify _extract_plain_text_no_cache doesn't cache."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        page = indexer.pages["x20di9371_page"]
        text = indexer._extract_plain_text_no_cache(page)

        # Verify text was extracted
        assert text is not None
        assert "X20DI9371" in text

        # Verify it was NOT cached
        assert page.plain_text is None

    def test_extract_content_file_not_found(self, temp_help_dir):
        """Verify graceful handling of missing HTML files."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Page Id="missing_page" Text="Missing" File="nonexistent.html"/>
</BrHelpContent>
"""
        (temp_help_dir / "brhelpcontent.xml").write_text(xml_content, encoding="utf-8")

        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        # Should return None without crashing
        html = indexer.extract_html_content("missing_page")
        assert html is None

        text = indexer.extract_plain_text("missing_page")
        assert text is None


class TestPageRetrieval:
    """Test page retrieval methods."""

    def test_get_page_by_id(self, temp_help_dir, sample_xml):
        """Verify get_page_by_id returns correct page."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        page = indexer.get_page_by_id("x20di9371_page")
        assert page is not None
        assert page.text == "X20DI9371"
        assert page.file_path == "hardware/x20di9371.html"

    def test_get_page_by_id_not_found(self, temp_help_dir, sample_xml):
        """Verify get_page_by_id returns None for non-existent page."""
        indexer = HelpContentIndexer(temp_help_dir)
        indexer.parse_xml_structure()

        page = indexer.get_page_by_id("nonexistent")
        assert page is None
