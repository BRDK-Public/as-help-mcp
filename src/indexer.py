"""Optimized help content indexer with incremental indexing support."""

import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import defusedxml.ElementTree as DefusedET

logger = logging.getLogger(__name__)


class SectionChild(TypedDict):
    """Type definition for section children."""

    id: str
    title: str
    file_path: str
    is_section: bool


class Category(TypedDict):
    """Type definition for top-level categories."""

    id: str
    title: str
    file_path: str


@dataclass
class HelpPage:
    """Represents a help page or section."""

    id: str
    text: str
    file_path: str
    help_id: str | None = None
    parent_id: str | None = None
    is_section: bool = False
    html_content: str | None = None
    plain_text: str | None = None


class HelpContentIndexer:
    """Indexes B&R Automation Studio help content with incremental update support."""

    def __init__(self, help_root: Path, metadata_dir: Path | None = None):
        """Initialize indexer with path to help root directory.
        Args:
            help_root: Path to directory containing brhelpcontent.xml and HTML files
            metadata_dir: Directory to store metadata (defaults to help_root/.ashelp_metadata)
        """
        self.help_root = Path(help_root)
        self.xml_path = self.help_root / "brhelpcontent.xml"
        self.metadata_dir = Path(metadata_dir) if metadata_dir else self.help_root / ".ashelp_metadata"
        self.metadata_path = self.metadata_dir / "index_metadata.json"

        self.pages: dict[str, HelpPage] = {}
        self.help_id_map: dict[str, str] = {}  # Maps HelpID -> page ID
        self._breadcrumb_cache: dict[str, list[HelpPage]] = {}  # Cache breadcrumbs to avoid recomputation
        self._duplicate_ids: dict[str, list[str]] = {}  # Track duplicate IDs: id -> [first_title, second_title, ...]

        # Ensure directories exist
        self.help_root.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        if not self.xml_path.exists():
            raise ValueError(  # pragma: no cover
                f"brhelpcontent.xml not found at: {self.xml_path}. "  # pragma: no cover
                "Please ensure you have copied the B&R Help 'Data' folder content to this directory."  # pragma: no cover
            )  # pragma: no cover

    def _get_xml_hash(self) -> str:
        """Calculate MD5 hash of brhelpcontent.xml for change detection."""
        return hashlib.md5(self.xml_path.read_bytes(), usedforsecurity=False).hexdigest()

    def _load_metadata(self) -> dict[str, str | int] | None:
        """Load index metadata if it exists."""
        if self.metadata_path.exists():
            try:
                data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else None
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to load metadata: {e}")  # pragma: no cover
        return None

    def _save_metadata(self):
        """Save index metadata."""
        metadata = {
            "xml_hash": self._get_xml_hash(),
            "indexed_at": datetime.now().isoformat(),
            "page_count": len(self.pages),
            "help_id_count": len(self.help_id_map),
            "help_root": str(self.help_root),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(f"Saved metadata: {metadata['page_count']} pages, {metadata['help_id_count']} HelpIDs")

    def needs_reindex(self) -> bool:
        """Check if XML has changed and reindexing is needed.

        Returns:
            True if brhelpcontent.xml has changed or no metadata exists
        """
        metadata = self._load_metadata()
        if not metadata:
            logger.info("No metadata found - full index required")
            return True

        current_hash = self._get_xml_hash()
        has_changed = metadata.get("xml_hash") != current_hash

        if has_changed:
            logger.info("XML file has changed - reindex required")
        else:
            logger.info("XML file unchanged - can use existing index")

        return has_changed

    def parse_xml_structure(self) -> None:
        """Parse brhelpcontent.xml to extract structure and metadata."""
        logger.info(f"Parsing {self.xml_path}")
        start_time = datetime.now()

        try:
            tree = DefusedET.parse(self.xml_path)
            root = tree.getroot()

            if root is None:
                raise ValueError("Failed to parse XML: root element is None")  # pragma: no cover

            logger.info(f"Root element: {root.tag}")

            # Process sections and pages - tags may be abbreviated (S=Section, P=Page)
            for child in root:
                if child.tag in ("Section", "S"):  # S is abbreviated Section
                    self._process_section(child, None)
                elif child.tag in ("Page", "P"):  # P is abbreviated Page
                    self._process_page(child, None)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Indexed {len(self.pages)} pages and sections in {elapsed:.2f}s")
            logger.info(f"Found {len(self.help_id_map)} HelpID mappings")

            # Report duplicate IDs found in XML (these cause breadcrumb issues)
            if self._duplicate_ids:
                logger.warning(f"Found {len(self._duplicate_ids)} duplicate IDs in brhelpcontent.xml (B&R data issue)")
                for dup_id, titles in list(self._duplicate_ids.items())[:5]:
                    logger.warning(f"  Duplicate ID '{dup_id}': used by {titles}")
                if len(self._duplicate_ids) > 5:
                    logger.warning(f"  ... and {len(self._duplicate_ids) - 5} more duplicates")  # pragma: no cover

            if len(self.help_id_map) == 0:
                logger.warning("No HelpIDs found! Checking first 5 pages for debugging...")
                count = 0
                for page_id, page in list(self.pages.items())[:5]:
                    logger.warning(f"  Page '{page_id}': title='{page.text}', help_id={page.help_id}")
                    count += 1

            # Pre-compute breadcrumbs for all pages (avoids repeated computation during search)
            logger.info("Pre-computing breadcrumbs for all pages...")
            self._precompute_breadcrumbs()
            logger.info(f"Breadcrumb cache populated: {len(self._breadcrumb_cache)} entries")

            # Log top-level categories for visibility at startup
            categories = self.get_top_level_categories()
            logger.info(f"Found {len(categories)} top-level categories:")
            for cat in categories:
                logger.info(f"  - {cat['title']} (id: {cat['id']})")

            # Save metadata for future incremental checks
            self._save_metadata()

        except ET.ParseError as e:  # pragma: no cover
            logger.error(f"Failed to parse XML: {e}")  # pragma: no cover
            raise  # pragma: no cover

    def _process_section(self, section_elem: ET.Element, parent_id: str | None = None) -> None:
        """Process a Section element recursively.

        Handles both full and abbreviated XML formats:
        - Full: <Section Text="..." File="..." Id="...">
        - Abbreviated: <S t="..." p="..." Id="...">
        """
        section_id = section_elem.get("Id")
        # Handle both full (Text) and abbreviated (t) attribute names
        text = section_elem.get("Text", section_elem.get("t", ""))
        file_path = section_elem.get("File", section_elem.get("p", ""))

        if not section_id:
            return  # pragma: no cover

        # Check for duplicate ID (B&R XML data issue)
        if section_id in self.pages:
            existing = self.pages[section_id]
            if section_id not in self._duplicate_ids:
                self._duplicate_ids[section_id] = [existing.text]
            self._duplicate_ids[section_id].append(text)

        # Create section entry
        page = HelpPage(id=section_id, text=text, file_path=file_path, parent_id=parent_id, is_section=True)

        # Check for HelpID - may be in <Identifiers> or <I> element
        identifiers = section_elem.find("Identifiers")
        if identifiers is None:
            identifiers = section_elem.find("I")
        if identifiers is not None:
            # Find HelpID (NOTE: must use 'is not None' because empty elements are falsy!)
            help_id_elem = identifiers.find("HelpID")
            if help_id_elem is None:
                help_id_elem = identifiers.find("H")
            if help_id_elem is not None:
                help_id = help_id_elem.get("Value")
                if help_id is None:
                    help_id = help_id_elem.get("v")
                if help_id:
                    page.help_id = help_id
                    self.help_id_map[help_id] = section_id

        self.pages[section_id] = page

        # Process child sections and pages (handle both full and abbreviated tags)
        for child in section_elem:
            if child.tag in ("Section", "S"):
                self._process_section(child, section_id)
            elif child.tag in ("Page", "P"):
                self._process_page(child, section_id)

    def _process_page(self, page_elem: ET.Element, parent_id: str | None = None) -> None:
        """Process a Page element.

        Handles both full and abbreviated XML formats:
        - Full: <Page Text="..." File="..." Id="...">
        - Abbreviated: <P t="..." p="..." Id="...">
        """
        page_id = page_elem.get("Id")
        # Handle both full (Text) and abbreviated (t) attribute names
        text = page_elem.get("Text", page_elem.get("t", ""))
        file_path = page_elem.get("File", page_elem.get("p", ""))

        if not page_id:
            return  # pragma: no cover

        # Check for duplicate ID (B&R XML data issue)
        if page_id in self.pages:
            existing = self.pages[page_id]
            if page_id not in self._duplicate_ids:
                self._duplicate_ids[page_id] = [existing.text]
            self._duplicate_ids[page_id].append(text)

        page = HelpPage(id=page_id, text=text, file_path=file_path, parent_id=parent_id, is_section=False)

        # Check for HelpID - may be in <Identifiers> or <I> element
        identifiers = page_elem.find("Identifiers")
        if identifiers is None:
            identifiers = page_elem.find("I")
        if identifiers is not None:
            # Find HelpID (NOTE: must use 'is not None' because empty elements are falsy!)
            help_id_elem = identifiers.find("HelpID")
            if help_id_elem is None:
                help_id_elem = identifiers.find("H")
            if help_id_elem is not None:
                help_id = help_id_elem.get("Value")
                if help_id is None:
                    help_id = help_id_elem.get("v")
                if help_id:
                    page.help_id = help_id
                    self.help_id_map[help_id] = page_id

        self.pages[page_id] = page

    def extract_html_content(self, page_id: str) -> str | None:
        """Extract and cache HTML content for a page.

        Args:
            page_id: The unique ID of the page

        Returns:
            The HTML content as a string, or None if file not found
        """
        if page_id not in self.pages:
            return None

        page = self.pages[page_id]

        # Return cached content if available
        if page.html_content is not None:
            return page.html_content

        if not page.file_path:
            return None  # pragma: no cover

        html_file = self.help_root / page.file_path

        if not html_file.exists():
            logger.debug(f"HTML file not found: {html_file}")
            return None

        try:
            with open(html_file, encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
                page.html_content = html_content
                return html_content
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to read HTML file {html_file}: {e}")  # pragma: no cover
            return None  # pragma: no cover

    def _extract_plain_text_no_cache(self, page: "HelpPage") -> str | None:
        """Extract plain text without caching (for bulk indexing).

        Uses lxml parser for 2-3x faster parsing than html.parser.
        Extracts text with proper spacing to preserve word boundaries.

        Args:
            page: The HelpPage object

        Returns:
            Plain text content, or None if extraction fails
        """
        if not page.file_path:
            return None  # pragma: no cover

        html_file = self.help_root / page.file_path

        if not html_file.exists():
            return None  # pragma: no cover

        try:
            # Use lxml directly for maximum speed (bypasses BeautifulSoup overhead)
            from lxml import html as lxml_html

            with open(html_file, "rb") as f:  # Read as bytes for lxml
                tree = lxml_html.parse(f)

            root = tree.getroot()
            if root is None:
                return None  # pragma: no cover

            # Remove script and style elements using XPath (faster than Cleaner)
            # Cast to lxml HtmlElement to access xpath method
            from typing import cast
            from lxml.html import HtmlElement
            root_elem = cast(HtmlElement, root)
            for element in root_elem.xpath(".//script | .//style"):
                element.getparent().remove(element)

            # Extract text with proper spacing between block elements
            # Using get_text() with separator to preserve word boundaries
            text_parts: list[str] = []
            for elem in root.iter():
                if elem.text:
                    text_parts.append(elem.text)
                # Add space after block-level elements to preserve word boundaries
                if elem.tag in (
                    "p",
                    "div",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "li",
                    "td",
                    "th",
                    "tr",
                    "table",
                    "blockquote",
                    "pre",
                ):
                    text_parts.append(" ")
                if elem.tail:
                    text_parts.append(elem.tail)

            text = "".join(text_parts) if text_parts else ""

            if text:
                text_result = " ".join(text.split())
                return text_result
            return None  # pragma: no cover
        except Exception as e:  # pragma: no cover
            logger.debug(f"Failed to extract text from {html_file}: {e}")  # pragma: no cover
            return None  # pragma: no cover

    def extract_plain_text(self, page_id: str) -> str | None:
        """Extract plain text from HTML content for searching.

        Uses lxml parser for faster parsing.
        Extracts text with proper spacing to preserve word boundaries.

        Args:
            page_id: The unique ID of the page

        Returns:
            Plain text content, or None if extraction fails
        """
        if page_id not in self.pages:
            return None  # pragma: no cover

        page = self.pages[page_id]

        # Return cached text if available
        if page.plain_text is not None:
            return page.plain_text

        html_content = self.extract_html_content(page_id)
        if not html_content:
            return None

        try:
            # Use lxml for faster parsing
            from lxml import html as lxml_html

            root = lxml_html.fromstring(html_content)

            # Remove script and style elements using XPath (faster than Cleaner)
            # Cast to lxml HtmlElement to access xpath method
            from typing import cast
            from lxml.html import HtmlElement
            root_elem = cast(HtmlElement, root)
            for element in root_elem.xpath(".//script | .//style"):
                element.getparent().remove(element)

            # Extract text with proper spacing between block elements
            text_parts = []
            for elem in root.iter():
                if elem.text:
                    text_parts.append(elem.text)
                # Add space after block-level elements to preserve word boundaries
                if elem.tag in (
                    "p",
                    "div",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "li",
                    "td",
                    "th",
                    "tr",
                    "table",
                    "blockquote",
                    "pre",
                ):
                    text_parts.append(" ")
                if elem.tail:
                    text_parts.append(elem.tail)

            text = "".join(text_parts) if text_parts else ""

            if text:
                text = " ".join(text.split())

            page.plain_text = text
            return text

        except Exception as e:  # pragma: no cover
            logger.debug(f"Failed to extract text from HTML: {e}")  # pragma: no cover
            return None  # pragma: no cover

    def get_page_by_help_id(self, help_id: str) -> HelpPage | None:
        """Get a page by its HelpID.

        Args:
            help_id: The HelpID value

        Returns:
            The HelpPage object, or None if not found
        """
        page_id = self.help_id_map.get(help_id)
        if page_id:
            return self.pages.get(page_id)
        return None

    def get_page_by_id(self, page_id: str) -> HelpPage | None:
        """Get a page by its unique ID.

        Args:
            page_id: The unique page ID

        Returns:
            The HelpPage object, or None if not found
        """
        return self.pages.get(page_id)

    def _precompute_breadcrumbs(self) -> None:
        """Pre-compute breadcrumbs for all pages during indexing.

        This avoids repeated breadcrumb computation during search operations
        and ensures consistent results even with duplicate IDs in the XML.
        """
        computed = 0
        for page_id in self.pages:
            if page_id not in self._breadcrumb_cache:
                self._compute_breadcrumb(page_id)
                computed += 1
        logger.debug(f"Computed {computed} breadcrumbs")

    def _compute_breadcrumb(self, page_id: str) -> list[HelpPage]:
        """Compute breadcrumb for a single page (internal method).

        Args:
            page_id: The unique page ID

        Returns:
            List of HelpPage objects from root to current page
        """
        breadcrumb: list[HelpPage] = []
        current_id: str | None = page_id
        visited = set()  # Prevent infinite loops from duplicate IDs in XML

        while current_id:
            # Check for cycles (caused by duplicate IDs in B&R XML, not true circular refs)
            if current_id in visited:
                # This happens when B&R XML has duplicate IDs - the same ID appears
                # in different parts of the tree (e.g., 'General information' and 'Status numbers')
                # The later occurrence overwrites the earlier one, causing apparent cycles.
                # This is a B&R XML data quality issue, not a bug in our code.
                logger.debug(f"Duplicate ID detected in breadcrumb for '{page_id}': stopping at {current_id}")
                break

            visited.add(current_id)

            page = self.pages.get(current_id)
            if not page:
                logger.debug(f"Breadcrumb traversal stopped: page_id '{current_id}' not found")
                break

            breadcrumb.insert(0, page)
            current_id = page.parent_id

            # Safety limit to prevent extremely deep hierarchies
            if len(breadcrumb) > 100:
                logger.error(f"Breadcrumb depth exceeded 100 levels for '{page_id}' - stopping")
                break

        # Cache the result
        self._breadcrumb_cache[page_id] = breadcrumb
        return breadcrumb

    def get_breadcrumb(self, page_id: str) -> list[HelpPage]:
        """Get the breadcrumb trail for a page.

        Args:
            page_id: The unique page ID

        Returns:
            List of HelpPage objects from root to current page
        """
        # Return cached breadcrumb (pre-computed during indexing)
        if page_id in self._breadcrumb_cache:
            return self._breadcrumb_cache[page_id]

        # Compute on-demand if not cached (shouldn't happen after parse_xml_structure)
        return self._compute_breadcrumb(page_id)

    def get_breadcrumb_string(self, page_id: str) -> str:
        """Get breadcrumb as a simple string path.

        Args:
            page_id: The unique page ID

        Returns:
            String like 'Root > Section > Subsection > Page'
        """
        breadcrumb = self.get_breadcrumb(page_id)
        return " > ".join(p.text for p in breadcrumb) if breadcrumb else ""

    def get_top_level_categories(self) -> list[dict]:
        """Get all root-level sections (categories) from the help content.

        These are sections with no parent (parent_id is None) and is_section is True.
        Useful for displaying the main navigation structure or filtering searches.

        Returns:
            List of dicts with 'id', 'title', and 'file_path' keys for each root section.
        """
        categories = []
        for _, page in self.pages.items():
            if page.parent_id is None and page.is_section:
                categories.append({"id": page.id, "title": page.text, "file_path": page.file_path})
        # Sort alphabetically by title for consistent ordering
        return sorted(categories, key=lambda x: x["title"].lower())

    def get_section_children(self, section_id: str) -> list[SectionChild]:
        """Get all immediate children of a section.

        Returns both child sections and child pages that have the specified
        section_id as their parent.

        Args:
            section_id: The unique ID of the parent section

        Returns:
            List of dicts with 'id', 'title', 'file_path', and 'is_section' keys.
            Sections are listed first (alphabetically), then pages (alphabetically).
        """
        if section_id not in self.pages:
            logger.warning(f"Section '{section_id}' not found")
            return []

        children: list[SectionChild] = []
        for _, page in self.pages.items():
            if page.parent_id == section_id:
                children.append(
                    {"id": page.id, "title": page.text, "file_path": page.file_path, "is_section": page.is_section}
                )

        # Sort: sections first (alphabetically), then pages (alphabetically)
        sections = sorted([c for c in children if c["is_section"]], key=lambda x: x["title"].lower())
        pages = sorted([c for c in children if not c["is_section"]], key=lambda x: x["title"].lower())
        return sections + pages
