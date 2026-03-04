"""Shared test fixtures and utilities."""

import hashlib
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import pytest

from src.embeddings import EmbeddingService
from src.indexer import HelpContentIndexer, HelpPage


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service that returns deterministic vectors without loading a real model.

    Uses MD5 hash of input text to generate reproducible vectors. Identical texts
    produce identical vectors, while different texts produce different vectors.
    Uses dimension=8 for fast tests.
    """

    def __init__(self, dimension: int = 8):
        # Skip parent __init__ entirely — no model loading
        self.model_name = "mock-model"
        self._model = None
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).digest()
        vector = [float(h[i % len(h)]) / 255.0 for i in range(self._dimension)]
        # Normalize to unit vector for cosine similarity
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_batch(self, texts: list[str], batch_size: int = 256) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def mock_embedding_service():
    """Provide a MockEmbeddingService instance for tests."""
    return MockEmbeddingService()


@pytest.fixture
def temp_help_dir(tmp_path):
    """Create temporary help directory with sample content."""
    help_dir = tmp_path / "help"
    help_dir.mkdir()

    # Create sample pages directory
    (help_dir / "hardware").mkdir()
    (help_dir / "motion").mkdir()
    (help_dir / "motion" / "mapp_motion").mkdir()

    # Create sample HTML files
    (help_dir / "index.html").write_text(
        """
        <html><head><title>Index</title></head>
        <body><h1>Welcome</h1><p>This is the index page.</p></body></html>
    """,
        encoding="utf-8",
    )

    (help_dir / "hardware" / "x20di9371.html").write_text(
        """
        <html><head><title>X20DI9371</title></head>
        <body><h1>X20DI9371</h1><p>Digital input module with 12 channels.</p></body></html>
    """,
        encoding="utf-8",
    )

    (help_dir / "motion" / "overview.html").write_text(
        """
        <html><head><title>Motion Overview</title></head>
        <body><h1>Motion</h1><p>Motion control system overview.</p></body></html>
    """,
        encoding="utf-8",
    )

    (help_dir / "motion" / "mapp_motion" / "mc_br_moveabsolute.html").write_text(
        """
        <html><head><title>MC_BR_MoveAbsolute</title></head>
        <body><h1>MC_BR_MoveAbsolute</h1><p>Moves axis to absolute position.</p></body></html>
    """,
        encoding="utf-8",
    )

    return help_dir


@pytest.fixture
def sample_xml(temp_help_dir):
    """Create a sample brhelpcontent.xml file."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <Section Id="hardware_section" Text="Hardware" File="index.html">
        <Page Id="x20di9371_page" Text="X20DI9371" File="hardware/x20di9371.html">
            <Identifiers>
                <HelpID Value="12345"/>
            </Identifiers>
        </Page>
    </Section>
    <Section Id="motion_section" Text="Motion" File="motion/overview.html">
        <Identifiers>
            <HelpID Value="20000"/>
        </Identifiers>
        <Section Id="mapp_motion_section" Text="mapp Motion" File="motion/overview.html">
            <Page Id="mc_moveabs_page" Text="MC_BR_MoveAbsolute" File="motion/mapp_motion/mc_br_moveabsolute.html">
                <Identifiers>
                    <HelpID Value="20100"/>
                </Identifiers>
            </Page>
        </Section>
    </Section>
</BrHelpContent>
"""
    xml_path = temp_help_dir / "brhelpcontent.xml"
    xml_path.write_text(xml_content, encoding="utf-8")
    return xml_path


@pytest.fixture
def sample_xml_abbreviated(temp_help_dir):
    """Create a sample brhelpcontent.xml with abbreviated tags."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<BrHelpContent>
    <S Id="hardware_section" t="Hardware" p="index.html">
        <P Id="x20di9371_page" t="X20DI9371" p="hardware/x20di9371.html">
            <I>
                <H v="12345"/>
            </I>
        </P>
    </S>
    <S Id="motion_section" t="Motion" p="motion/overview.html">
        <I>
            <H v="20000"/>
        </I>
        <S Id="mapp_motion_section" t="mapp Motion" p="motion/overview.html">
            <P Id="mc_moveabs_page" t="MC_BR_MoveAbsolute" p="motion/mapp_motion/mc_br_moveabsolute.html">
                <I>
                    <H v="20100"/>
                </I>
            </P>
        </S>
    </S>
</BrHelpContent>
"""
    xml_path = temp_help_dir / "brhelpcontent.xml"
    xml_path.write_text(xml_content, encoding="utf-8")
    return xml_path


@pytest.fixture
def mock_indexer():
    """Create indexer with in-memory test data (no file system)."""
    indexer = MagicMock(spec=HelpContentIndexer)

    # Create sample pages
    page1 = HelpPage(
        id="page1", text="Test Page", file_path="test.html", is_section=False, help_id="12345", parent_id="section1"
    )

    section1 = HelpPage(id="section1", text="Test Section", file_path="section.html", is_section=True, parent_id=None)

    page2 = HelpPage(id="page2", text="Child Page", file_path="child.html", is_section=False, parent_id="section1")

    indexer.pages = {"page1": page1, "section1": section1, "page2": page2}

    indexer.help_id_map = {"12345": "page1"}
    indexer._breadcrumb_cache = {"page1": [section1, page1], "page2": [section1, page2], "section1": [section1]}

    indexer.get_page_by_id = lambda pid: indexer.pages.get(pid)
    indexer.get_page_by_help_id = lambda hid: indexer.pages.get(indexer.help_id_map.get(hid))
    indexer.get_breadcrumb = lambda pid: indexer._breadcrumb_cache.get(pid, [])
    indexer.get_breadcrumb_string = lambda pid: " > ".join(p.text for p in indexer.get_breadcrumb(pid))

    return indexer


@pytest.fixture
def initialized_indexer(sample_xml):
    """Create a fully initialized indexer with parsed content."""
    indexer = HelpContentIndexer(sample_xml.parent)
    indexer.parse_xml_structure()
    return indexer


def create_sample_xml_string(pages: list[dict]) -> str:
    """Generate brhelpcontent.xml from page definitions.

    Args:
        pages: List of dicts with keys: id, text, file_path, is_section, parent_id, help_id

    Returns:
        XML string
    """
    root = ET.Element("BrHelpContent")

    # Build hierarchy
    def add_element(parent_elem, page_dict):
        tag = "Section" if page_dict.get("is_section") else "Page"
        elem = ET.SubElement(parent_elem, tag)
        elem.set("Id", page_dict["id"])
        elem.set("Text", page_dict["text"])
        elem.set("File", page_dict["file_path"])

        if page_dict.get("help_id"):
            identifiers = ET.SubElement(elem, "Identifiers")
            help_id_elem = ET.SubElement(identifiers, "HelpID")
            help_id_elem.set("Value", page_dict["help_id"])

        return elem

    # First pass: create root elements
    page_elements = {}
    for page in pages:
        if page.get("parent_id") is None:
            elem = add_element(root, page)
            page_elements[page["id"]] = elem

    # Second pass: add children
    for page in pages:
        if page.get("parent_id") is not None:
            parent_elem = page_elements.get(page["parent_id"])
            if parent_elem is not None:
                elem = add_element(parent_elem, page)
                page_elements[page["id"]] = elem

    return ET.tostring(root, encoding="unicode")


@pytest.fixture
def create_xml_helper():
    """Helper function to create XML content."""
    return create_sample_xml_string
