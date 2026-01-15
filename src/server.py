"""B&R Automation Studio Help MCP Server.

This MCP server provides search and retrieval capabilities for B&R Automation Studio help content.
Optimized for thousands of help files with persistent indexing and fast startup.
"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from src.indexer import HelpContentIndexer
from src.search_engine import HelpSearchEngine

# Load environment variables from .env file
load_dotenv()


def get_as_version_config() -> tuple[str, str]:
    """Get Automation Studio version from environment variable.

    Returns:
        Tuple of (version, online_help_base_url)
        Version is '4' or '6' (defaults to '4')
    """
    version = os.getenv("AS_HELP_VERSION", "4").strip()

    if version == "6":
        return "6", "https://help.br-automation.com/#/en/6/"

    # Default to AS4 (most common)
    return "4", "https://help.br-automation.com/#/en/4/"


# Configure logging - use stderr so it doesn't interfere with stdio transport
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Pydantic models for structured output
class SearchResult(BaseModel):
    """A single search result - metadata only. Call get_page_by_id for actual content."""

    page_id: str = Field(description="REQUIRED: Use this with get_page_by_id to get the actual page content")
    title: str = Field(description="Title of the help page")
    file_path: str = Field(description="Relative path to the HTML file")
    online_help_url: str | None = Field(default=None, description="Direct link to B&R online help for this page")
    help_id: str | None = Field(default=None, description="HelpID if available")
    is_section: bool = Field(description="Whether this is a section (True) or page (False)")
    score: float | None = Field(default=None, description="Search relevance score (lower is better)")
    breadcrumb_path: str | None = Field(default=None, description="Navigation path like 'Section > Subsection > Page'")
    category: str | None = Field(default=None, description="Top-level category (e.g., 'Motion', 'Hardware', 'Safety')")
    content_preview: str | None = Field(
        default=None, description="First ~100 chars only. NOT enough to answer questions - call get_page_by_id!"
    )


class SearchResults(BaseModel):
    """Collection of search results."""

    query: str = Field(description="The search query")
    results: list[SearchResult] = Field(description="List of matching pages")
    total: int = Field(description="Total number of results returned")


class PageContent(BaseModel):
    """Full content of a help page."""

    page_id: str = Field(description="Unique identifier for the page")
    title: str = Field(description="Title of the help page")
    file_path: str = Field(description="Relative path to the HTML file")
    online_help_url: str | None = Field(default=None, description="Direct link to B&R online help for this page")
    help_id: str | None = Field(default=None, description="HelpID if available")
    html_content: str | None = Field(default=None, description="Full HTML content")
    plain_text: str | None = Field(default=None, description="Plain text content")
    breadcrumb: list[str] = Field(default_factory=list, description="Breadcrumb trail (titles)")


class BreadcrumbItem(BaseModel):
    """A single item in a breadcrumb trail."""

    page_id: str = Field(description="Unique identifier")
    title: str = Field(description="Page/section title")
    file_path: str = Field(description="Relative path to HTML file")
    is_section: bool = Field(description="Whether this is a section")


class CategoryInfo(BaseModel):
    """A top-level category in the help documentation."""

    id: str = Field(description="Unique identifier for the category (use with browse_section)")
    title: str = Field(description="Display name of the category")
    file_path: str = Field(description="Relative path to the category's HTML file")


class CategoriesResult(BaseModel):
    """List of all top-level categories in the help documentation."""

    categories: list[CategoryInfo] = Field(description="List of top-level categories")
    total: int = Field(description="Total number of categories")


class SectionChild(BaseModel):
    """A child item within a section."""

    id: str = Field(description="Unique identifier (use with browse_section for sections, get_page_by_id for pages)")
    title: str = Field(description="Display name")
    file_path: str = Field(description="Relative path to HTML file")
    is_section: bool = Field(description="True if this is a section (can be browsed), False if it's a page")


class SectionChildren(BaseModel):
    """Children of a section in the help tree."""

    section_id: str = Field(description="ID of the parent section")
    section_title: str = Field(description="Title of the parent section")
    children: list[SectionChild] = Field(description="List of child sections and pages (sections listed first)")
    total: int = Field(description="Total number of children")


@dataclass
class AppContext:
    """Shared application context."""

    indexer: HelpContentIndexer
    search_engine: HelpSearchEngine
    as_version: str  # '4', '6', or 'unknown'
    online_help_base_url: str  # Base URL for online help links


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle - load and index help content on startup."""

    # Get help root from environment (defaults to /data/help for Docker)
    help_root = os.getenv("AS_HELP_ROOT", "/data/help")
    help_root_path = Path(help_root).resolve()

    # Get database path for SQLite FTS5 index
    # Default to /data/db for Docker volumes (not help root for read-only compatibility)
    if help_root.startswith("/data/"):
        default_db_path = "/data/db/.ashelp_search.db"
    else:
        default_db_path = str(help_root_path / ".ashelp_search.db")

    db_path = Path(os.getenv("AS_HELP_DB_PATH", default_db_path)).resolve()

    # Check if force rebuild is requested
    force_rebuild = os.getenv("AS_HELP_FORCE_REBUILD", "false").lower() == "true"

    # Get metadata directory (separate from help root for read-only mounts)
    # Default to /data/db for Docker volumes (not help root for read-only compatibility)
    if help_root.startswith("/data/"):
        default_metadata_dir = "/data/db/.ashelp_metadata"
    else:
        default_metadata_dir = str(help_root_path / ".ashelp_metadata")

    metadata_dir = os.getenv("AS_HELP_METADATA_DIR", default_metadata_dir)
    metadata_path = Path(metadata_dir).resolve()

    # Get AS version from environment variable
    as_version, online_help_base_url = get_as_version_config()

    logger.info("=== B&R Help Server Startup ===")
    logger.info(f"Help root: {help_root_path}")
    logger.info(f"AS Version: {as_version}")
    logger.info(f"Online help base: {online_help_base_url}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Metadata dir: {metadata_path}")
    logger.info(f"Force rebuild: {force_rebuild}")

    # Initialize indexer (parses XML structure)
    logger.info("Initializing help indexer...")
    indexer = HelpContentIndexer(help_root_path, metadata_dir=metadata_path)
    indexer.parse_xml_structure()

    # Log available top-level categories
    categories = indexer.get_top_level_categories()
    logger.info(f"Available top-level categories ({len(categories)}):")
    for cat in categories:
        logger.info(f"  - {cat['title']}")

    # Initialize search engine (builds or loads SQLite FTS5 index)
    logger.info("Initializing search engine...")
    search_engine = HelpSearchEngine(db_path=db_path, indexer=indexer, force_rebuild=force_rebuild)

    logger.info("=== Server ready ===")

    # Yield context to the application
    context = AppContext(
        indexer=indexer, search_engine=search_engine, as_version=as_version, online_help_base_url=online_help_base_url
    )
    yield context

    logger.info("Shutting down help server")


# Create FastMCP server with lifespan management
mcp = FastMCP(  # pragma: no cover
    "B&R Automation Studio Help Server",  # pragma: no cover
    instructions=(  # pragma: no cover
        "B&R Automation Studio Help Server - 100k+ pages of technical documentation.\n\n"  # pragma: no cover
        "*** DISCOVERY & BROWSING ***\n\n"  # pragma: no cover
        "- get_categories - List all top-level categories (e.g., Hardware, Motion, Safety)\n"  # pragma: no cover
        "- browse_section - Navigate into a category/section to see its children\n\n"  # pragma: no cover
        "*** THOROUGH RESEARCH WORKFLOW ***\n\n"  # pragma: no cover
        "For comprehensive answers, use MULTIPLE searches and retrieve MULTIPLE pages:\n\n"  # pragma: no cover
        "1. search_help - Find pages by keyword. Returns titles/page_ids only, NO content.\n"  # pragma: no cover
        "2. get_page_by_id - Get FULL content. Call for EACH relevant page.\n"  # pragma: no cover
        "3. REPEAT - Search with different keywords if needed. Get more pages.\n\n"  # pragma: no cover
        "BEST PRACTICES:\n"  # pragma: no cover
        "- Use get_categories() to discover valid category names for filtering search_help\n"  # pragma: no cover
        "- Use browse_section() to explore a category's structure before searching\n"  # pragma: no cover
        "- Complex questions need 2-5 page retrievals from different angles\n"  # pragma: no cover
        "- Search for the main topic, then related concepts (e.g., 'MC_BR_MoveAbsolute' then 'axis error handling')\n"  # pragma: no cover
        "- If first search doesn't have what you need, try synonyms or related terms\n"  # pragma: no cover
        "- Retrieve pages that look relevant - reading 3 pages is better than guessing\n\n"  # pragma: no cover
        "WARNING: content_preview is ~100 chars - NEVER answer from previews alone.\n"  # pragma: no cover
        "You MUST call get_page_by_id to read actual documentation."  # pragma: no cover
    ),  # pragma: no cover
    lifespan=app_lifespan,  # pragma: no cover
)  # pragma: no cover


@mcp.tool()
def search_help(
    ctx: Context,
    query: str = Field(description="Search query (keywords or phrases). Try different keywords for better coverage."),
    limit: int = Field(
        default=5,
        description="Results per search. Use smaller limits and do multiple searches with different keywords.",
    ),
    content_search: bool = Field(default=True, description="Search in content (True) or titles only (False)"),
    category: str | None = Field(
        default=None,
        description="Filter by category (top-level section name). Common categories: Hardware, Motion, Safety, Vision, Communication, Programming, Visualization, Services. Call get_categories() for complete list.",
    ),
) -> SearchResults:
    """Find help pages matching your query. Returns page_ids ONLY - no actual content.

    WORKFLOW:
    1. Search for main topic -> get_page_by_id for top results
    2. Search for related terms -> get_page_by_id for those too
    3. Combine information from multiple pages for complete answer

    Tips:
    - Call get_categories() first to discover valid category filter values
    - Use specific function/feature names when known (e.g., 'MC_BR_MoveAbsolute')
    - Try variations: 'axis move' vs 'motion control' vs 'positioning'
    - Check breadcrumb_path to understand page context before retrieving

    You MUST call get_page_by_id to read actual content - previews are NOT enough.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Handle FieldInfo objects when function called directly (not through FastMCP framework)
    # This supports both MCP tool invocation and direct test calls
    from pydantic.fields import FieldInfo

    if isinstance(limit, FieldInfo):
        limit = limit.default or 5
    if isinstance(content_search, FieldInfo):
        content_search = content_search.default if content_search.default is not None else True
    if isinstance(category, FieldInfo):
        category = category.default

    # Perform search (returns breadcrumb_path directly from FTS5 index)
    results = app_ctx.search_engine.search(
        query=query, limit=limit, search_in_content=content_search, category=category
    )

    # Convert to SearchResult models
    search_results = []
    for r in results:
        # Get a tiny preview - intentionally truncated to force get_page_by_id usage
        content_preview = None
        if not r["is_section"]:
            full_text = app_ctx.indexer.extract_plain_text(r["page_id"])
            if full_text:
                # Only show first 100 chars - useless for answering, just proves content exists
                content_preview = full_text[:100].strip() + "... [TRUNCATED - call get_page_by_id for full content]"

        # Build online help URL from file path
        online_help_url = None
        if r["file_path"]:
            # Normalize path separators and build URL
            normalized_path = r["file_path"].replace("\\", "/")
            online_help_url = f"{app_ctx.online_help_base_url}{normalized_path}"

        result = SearchResult(
            page_id=r["page_id"],
            title=r["title"],
            file_path=r["file_path"],
            online_help_url=online_help_url,
            help_id=r.get("help_id"),
            is_section=r["is_section"],
            score=r["score"],
            breadcrumb_path=r.get("breadcrumb_path"),  # Now from search index, no extra call!
            category=r.get("category"),  # Top-level category for filtering
            content_preview=content_preview,
        )

        search_results.append(result)

    return SearchResults(query=query, results=search_results, total=len(search_results))


@mcp.tool()
def get_categories(ctx: Context) -> CategoriesResult:
    """Get all top-level categories from the help documentation.

    Returns the main navigation categories (e.g., Hardware, Motion, Safety, etc.).
    Use this to:
    - Discover valid category names for filtering search_help
    - Get an overview of the documentation structure
    - Find the category ID to browse with browse_section

    Each category can be explored further using browse_section(category_id).
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    categories = app_ctx.indexer.get_top_level_categories()

    category_results = [
        CategoryInfo(id=cat["id"], title=cat["title"], file_path=cat["file_path"]) for cat in categories
    ]

    return CategoriesResult(categories=category_results, total=len(category_results))


@mcp.tool()
def browse_section(
    ctx: Context,
    section_id: str = Field(
        description="ID of the section to browse (from get_categories or previous browse_section call)"
    ),
) -> SectionChildren | None:
    """Browse the children of a section in the help tree.

    Use this to navigate the documentation hierarchy:
    1. Call get_categories() to get top-level categories
    2. Call browse_section(category_id) to see what's inside
    3. For child sections (is_section=True), call browse_section again to go deeper
    4. For pages (is_section=False), call get_page_by_id to read the content

    This is useful for:
    - Exploring documentation structure before searching
    - Finding related pages within the same section
    - Understanding the organization of a topic area
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Get the parent section info
    parent = app_ctx.indexer.get_page_by_id(section_id)
    if not parent:
        return None

    children = app_ctx.indexer.get_section_children(section_id)

    child_results = [
        SectionChild(id=child["id"], title=child["title"], file_path=child["file_path"], is_section=child["is_section"])
        for child in children
    ]

    return SectionChildren(
        section_id=section_id, section_title=parent.text, children=child_results, total=len(child_results)
    )


@mcp.tool()
def get_page_by_id(
    ctx: Context,
    page_id: str = Field(description="Page ID from search results. Call multiple times for different pages."),
    include_html: bool = Field(default=False, description="Include full HTML content (rarely needed)"),
    include_text: bool = Field(default=True, description="Include full plain text content"),
    include_breadcrumb: bool = Field(default=True, description="Include navigation breadcrumb path"),
) -> PageContent | None:
    """Get the COMPLETE content of a help page. Call this for EACH page you need.

    For thorough answers:
    - Retrieve the main topic page
    - Also retrieve related pages (examples, error handling, best practices)
    - Cross-reference information from multiple pages

    Don't hesitate to call this multiple times - reading 3-5 pages gives much better answers.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    page = app_ctx.indexer.get_page_by_id(page_id)
    if not page:
        return None

    html_content = None
    plain_text = None
    breadcrumb = []

    if include_html:
        html_content = app_ctx.indexer.extract_html_content(page_id)

    if include_text:
        plain_text = app_ctx.indexer.extract_plain_text(page_id)

    if include_breadcrumb:
        breadcrumb_pages = app_ctx.indexer.get_breadcrumb(page_id)
        breadcrumb = [p.text for p in breadcrumb_pages]

    # Build online help URL
    online_help_url = None
    if page.file_path:
        normalized_path = page.file_path.replace("\\", "/")
        online_help_url = f"{app_ctx.online_help_base_url}{normalized_path}"

    return PageContent(
        page_id=page.id,
        title=page.text,
        file_path=page.file_path,
        online_help_url=online_help_url,
        help_id=page.help_id,
        html_content=html_content,
        plain_text=plain_text,
        breadcrumb=breadcrumb,
    )


@mcp.tool()
def get_page_by_help_id(
    ctx: Context,
    help_id: str = Field(description="HelpID value (e.g., '3002099')"),
    include_html: bool = Field(default=False, description="Include full HTML content"),
    include_text: bool = Field(default=True, description="Include plain text content"),
    include_breadcrumb: bool = Field(
        default=False, description="Include breadcrumb trail (use get_breadcrumb tool for full details)"
    ),
) -> PageContent | None:
    """Retrieve a help page by its HelpID.

    HelpIDs are numeric identifiers used in the B&R help system. This tool
    looks up the page associated with a HelpID and returns its content.
    Breadcrumb is optional - use the dedicated get_breadcrumb tool for navigation.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    page = app_ctx.indexer.get_page_by_help_id(help_id)
    if not page:
        return None

    html_content = None
    plain_text = None
    breadcrumb = []

    if include_html:
        html_content = app_ctx.indexer.extract_html_content(page.id)

    if include_text:
        plain_text = app_ctx.indexer.extract_plain_text(page.id)

    if include_breadcrumb:
        breadcrumb_pages = app_ctx.indexer.get_breadcrumb(page.id)
        breadcrumb = [p.text for p in breadcrumb_pages]

    # Build online help URL
    online_help_url = None
    if page.file_path:
        normalized_path = page.file_path.replace("\\", "/")
        online_help_url = f"{app_ctx.online_help_base_url}{normalized_path}"

    return PageContent(
        page_id=page.id,
        title=page.text,
        file_path=page.file_path,
        online_help_url=online_help_url,
        help_id=page.help_id,
        html_content=html_content,
        plain_text=plain_text,
        breadcrumb=breadcrumb,
    )


@mcp.tool()
async def get_breadcrumb(
    ctx: Context, page_id: str = Field(description="Unique page ID from search results")
) -> list[BreadcrumbItem]:
    """Get detailed navigation breadcrumb for a help page.

    Returns the full hierarchical path from root to the specified page.

    WARNING: This tool is RARELY needed! Search results already include breadcrumb_path
    as a string (e.g., 'Motion > mapp Motion > Programming > Libraries').

    Only call this if you specifically need:
    - The page_id of each parent section (for navigation)
    - To traverse the hierarchy programmatically

    For most questions, the breadcrumb_path string in search results is sufficient.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    breadcrumb_pages = app_ctx.indexer.get_breadcrumb(page_id)

    result = [
        BreadcrumbItem(page_id=p.id, title=p.text, file_path=p.file_path, is_section=p.is_section)
        for p in breadcrumb_pages
    ]

    if not result:
        await ctx.warning(f"No breadcrumb found for page_id: {page_id}")  # pragma: no cover

    return result


@mcp.tool()
async def get_help_statistics(ctx: Context) -> dict[str, int]:
    """Get statistics about the indexed help content.

    Returns counts of total pages, sections, and HelpID mappings.
    Also includes parent-child relationship statistics.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    total_pages = len(app_ctx.indexer.pages)
    total_sections = sum(1 for p in app_ctx.indexer.pages.values() if p.is_section)
    total_help_ids = len(app_ctx.indexer.help_id_map)

    # Check parent-child relationships
    pages_with_parents = sum(1 for p in app_ctx.indexer.pages.values() if p.parent_id is not None)
    root_pages = sum(1 for p in app_ctx.indexer.pages.values() if p.parent_id is None)

    await ctx.info(f"Statistics: {total_pages} total, {total_sections} sections, {total_help_ids} HelpIDs")
    await ctx.info(f"Hierarchy: {pages_with_parents} with parents, {root_pages} root items")

    return {
        "total_pages": total_pages,
        "total_sections": total_sections,
        "regular_pages": total_pages - total_sections,
        "help_id_mappings": total_help_ids,
        "pages_with_parents": pages_with_parents,
        "root_items": root_pages,
    }


# Resource for direct HTML file access
@mcp.resource("help://page/{page_id}")
def get_help_page_resource(page_id: str, ctx: Context) -> str:
    """Read the content of a specific help page.

    Returns the plain text content of the page.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Try to get plain text first (better for LLM)
    text = app_ctx.indexer.extract_plain_text(page_id)
    if text:
        return text

    # Fallback to HTML if text extraction fails but file exists
    html = app_ctx.indexer.extract_html_content(page_id)
    if html:
        return html

    raise ValueError(f"Page {page_id} not found or has no content")


@mcp.resource("help://html/{page_id}")
def get_page_html(page_id: str, ctx: Context) -> str:
    """Get HTML content for a help page by its ID.

    Returns the raw HTML content from the help file.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    html_content = app_ctx.indexer.extract_html_content(page_id)
    if html_content:
        return html_content
    return f"Page not found: {page_id}"


# Prompts for common workflows
@mcp.prompt()
def help_search(topic: str) -> str:
    """Search for a topic in B&R help and return comprehensive results with locations and HelpIDs.

    Use this prompt to search the B&R help documentation and get structured results
    including page IDs, navigation paths, online help URLs, and HelpIDs for context-sensitive help.
    """
    return f"""Search the B&R Automation Studio help documentation comprehensively for: "{topic}"

## Instructions

1. Use the `search_help` tool to find all relevant pages (use limit=10 for comprehensive results)
2. If the first search doesn't cover all aspects, search with alternative keywords or related terms
3. Use the `get_page_by_id` tool to retreive full content for the content summary.
4. For each relevant result, extract and present the following information:

## Required Output Format

For each matching page, provide a structured entry:

### [Page Title]
- **Page ID**: `<page_id>` (use this with get_page_by_id for full content)
- **Help Path**: <Full breadcrumb path showing navigation hierarchy>
- **Online Help**: <URL to B&R online help - can be shared with others>
- **HelpID**: <numeric HelpID if available, or "None">
- **File Path**: `<relative path to HTML file>`
- **Type**: Page or Section
- **Content Summary**:
    <Brief summary of the page content>

## Guidelines

- List results in order of relevance (best matches first)
- Include ALL relevant matches, not just the top one
- If a topic spans multiple pages (e.g., overview + details + examples), include all of them
- Group related pages together when they belong to the same section
- The Online Help URL is useful for sharing with colleagues or opening in a browser
- If no results are found, suggest alternative search terms

## Example Output

### MC_BR_MoveAbsolute
- **Page ID**: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
- **Help Path**: Motion > mapp Motion > Programming > Libraries > McAxis > Function blocks > MC_BR_MoveAbsolute
- **Online Help**: https://help.br-automation.com/#/en/4/motion/mapp_motion/programming/mcaxis/mc_br_moveabsolute.html
- **HelpID**: 5100234
- **File Path**: `motion/mapp_motion/programming/mcaxis/mc_br_moveabsolute.html`
- **Type**: Page
- **Content Summary**:
    The MC_BR_MoveAbsolute function block is used to move an axis to an absolute position.
    It supports various parameters for speed, acceleration, and deceleration.
    Typical usage involves initializing the block, setting the target position, and executing the move command.


Now search for "{topic}" and provide comprehensive results."""


@mcp.prompt()
def help_details(topic: str) -> str:
    """Deep research a topic in B&R help - retrieves and synthesizes content from multiple pages.

    Use this prompt for thorough answers that require reading actual documentation content,
    cross-referencing related pages, and providing a comprehensive explanation.
    """
    return f"""Perform DEEP RESEARCH on the B&R Automation Studio help documentation for: "{topic}"

## Research Workflow

1. **Initial Search**: Use `search_help` with limit=10 to find all relevant pages
2. **Retrieve Content**: Call `get_page_by_id` for the TOP 3-5 most relevant results
3. **Expand Search**: Search for related terms (e.g., if topic is a function block, also search for its error codes, examples, related FBs)
4. **Retrieve More**: Call `get_page_by_id` for additional relevant pages found
5. **Synthesize**: Combine information from ALL retrieved pages into a comprehensive answer

## Required Research Depth

- Read at LEAST 3 pages, preferably 5 or more
- Look for: main documentation, examples, error handling, best practices, related topics
- Cross-reference information between pages
- Note any discrepancies or version-specific information

## Output Format

### Overview
<High-level summary of what {topic} is and its purpose>

### Key Details
<Core technical information synthesized from multiple pages>

### Parameters / Configuration
<If applicable - list parameters, settings, or configuration options>

### Usage Examples
<Code examples or usage patterns from the documentation>

### Error Handling
<Common errors, troubleshooting tips, or error codes if documented>

### Related Topics
<Links to related pages with their Online Help URLs for further reading>

### Sources
List all pages consulted:
- [Page Title](Online Help URL) - brief note on what was found

## Guidelines

- DO NOT answer from search previews - they are truncated and insufficient
- ALWAYS call get_page_by_id multiple times to read actual content
- If the first search doesn't find what you need, try alternative keywords
- Prioritize official function block documentation, then examples, then general guides
- Include Online Help URLs so the user can read more themselves

Now perform deep research on "{topic}" and provide a comprehensive, well-sourced answer."""


@mcp.prompt()
def search_hardware(topic: str) -> str:
    """Search specifically in the Hardware section of the B&R help.

    Use this prompt when looking for information about X20 modules, PLCs,
    drives, motors, or other physical components.
    """
    return f"""Search for "{topic}" specifically in the Hardware category.

## Instructions

1. Use the `search_help` tool with `category='Hardware'` and `limit=10`.
2. Retrieve full content for the most relevant pages using `get_page_by_id`.
3. Provide a summary of the hardware component, including:
   - Technical data / specifications
   - Wiring / pinout information (if available)
   - Status/Error LED descriptions
   - Configuration parameters

Now search for "{topic}" in the Hardware section."""


@mcp.prompt()
def search_motion(topic: str) -> str:
    """Search specifically in the Motion section of the B&R help.

    Use this prompt for questions about ACOPOS, mapp Motion, axis control,
    CNC, robotics, or motion function blocks (MC_*).
    """
    return f"""Search for "{topic}" specifically in the Motion category.

## Instructions

1. Use the `search_help` tool with `category='Motion'` and `limit=10`.
2. Retrieve full content for the most relevant pages using `get_page_by_id`.
3. Provide a detailed explanation including:
   - Function block interface (if applicable)
   - Configuration / parameters
   - Error codes and troubleshooting
   - Usage examples

Now search for "{topic}" in the Motion section."""


@mcp.prompt()
def search_visualization(topic: str) -> str:
    """Search specifically in the Visualization section of the B&R help.

    Use this prompt for questions about mapp View, Visual Components,
    widgets, events, actions, or HMI configuration.
    """
    return f"""Search for "{topic}" specifically in the Visualization category.

## Instructions

1. Use the `search_help` tool with `category='Visualization'` and `limit=10`.
   (Note: If 'Visualization' yields few results, try searching without category as some older content might be under 'VisualComponents' or 'mapp View')
2. Retrieve full content for the most relevant pages using `get_page_by_id`.
3. Provide a summary including:
   - Widget/Component properties and events
   - Configuration details
   - Usage examples

Now search for "{topic}" in the Visualization section."""


@mcp.prompt()
def search_safety(topic: str) -> str:
    """Search specifically in the Safety section of the B&R help.

    Use this prompt for questions about Integrated Safety, SafeLOGIC,
    safety functions, or safety hardware configuration.
    """
    return f"""Search for "{topic}" specifically in the Safety category.

## Instructions

1. Use the `search_help` tool with `category='Safety'` and `limit=10`.
2. Retrieve full content for the most relevant pages using `get_page_by_id`.
3. Provide a detailed explanation including:
   - Safety function details
   - Configuration / parameters
   - Wiring / hardware constraints (if applicable)
   - Usage examples

Now search for "{topic}" in the Safety section."""


@mcp.prompt()
def search_vision(topic: str) -> str:
    """Search specifically in the Vision section of the B&R help.

    Use this prompt for questions about mapp Vision, Smart Camera,
    vision sensors, or image processing functions.
    """
    return f"""Search for "{topic}" specifically in the Vision category.

## Instructions

1. Use the `search_help` tool with `category='Vision'` and `limit=10`.
2. Retrieve full content for the most relevant pages using `get_page_by_id`.
3. Provide a summary including:
   - Component/Function details
   - Configuration parameters
   - Usage examples and best practices

Now search for "{topic}" in the Vision section."""


@mcp.prompt()
def search_communication(topic: str) -> str:
    """Search specifically in the Communication section of the B&R help.

    Use this prompt for questions about POWERLINK, OPC UA, Modbus,
    PROFINET, EtherNet/IP, or other fieldbus protocols.
    """
    return f"""Search for "{topic}" specifically in the Communication category.

## Instructions

1. Use the `search_help` tool with `category='Communication'` and `limit=10`.
2. Retrieve full content for the most relevant pages using `get_page_by_id`.
3. Provide a detailed explanation including:
   - Protocol specifications / limits
   - Configuration steps
   - Function blocks for data exchange
   - Troubleshooting

Now search for "{topic}" in the Communication section."""


@mcp.prompt()
def search_programming(topic: str) -> str:
    """Search specifically in the Programming section of the B&R help.

    Use this prompt for questions about standard libraries, IEC 61131-3
    programming, C/C++ usage, or general system functions.
    """
    return f"""Search for "{topic}" specifically in the Programming category.

## Instructions

1. Use the `search_help` tool with `category='Programming'` and `limit=10`.
2. Retrieve full content for the most relevant pages using `get_page_by_id`.
3. Provide a summary including:
   - Function/Library details
   - Input/Output parameters
   - Return values and error codes
   - Code examples

Now search for "{topic}" in the Programming section."""


@mcp.prompt()
def search_mapp_services(topic: str) -> str:
    """Search specifically for mapp Services.

    Use this prompt for questions about mapp Services like AlarmX, Recipe,
    UserX, Data, Report, Audit, etc. (Excludes mapp Motion/View/Vision).
    """
    return f"""Search for "{topic}" specifically in the Services category.

## Instructions

1. Use the `search_help` tool with `category='Services'` and `limit=10`.
2. Retrieve full content for the most relevant pages using `get_page_by_id`.
3. Provide a detailed explanation including:
   - Component configuration (MpLink)
   - Function block interface
   - UI connection (if applicable)
   - Usage examples

Now search for "{topic}" in the Services section."""


def main():
    """Entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="B&R Automation Studio Help MCP Server")
    parser.add_argument(
        "--help-root",
        help="Path to AS Help Data folder (AS_HELP_ROOT). Example: 'C:\\BRAutomation\\AS412\\Help-en\\Data'",
    )
    parser.add_argument(
        "--db-path",
        help="Path to database file (AS_HELP_DB_PATH). Example: './db/ashelp.db'",
    )
    parser.add_argument(
        "--metadata-dir",
        help="Path to metadata directory (AS_HELP_METADATA_DIR). Example: './metadata'",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Force index rebuild (AS_HELP_FORCE_REBUILD)")
    parser.add_argument(
        "--as-version",
        choices=["4", "6"],
        help="AS version for online help (AS_HELP_VERSION). Default: 4",
    )
    parser.add_argument("--usage", action="store_true", help="Print usage examples and exit")

    # Parse known args to allow them to be passed before or after FastMCP args
    args, _ = parser.parse_known_args()

    if args.help_root:
        os.environ["AS_HELP_ROOT"] = str(Path(args.help_root).resolve())
    if args.db_path:
        os.environ["AS_HELP_DB_PATH"] = str(Path(args.db_path).resolve())
    if args.metadata_dir:
        os.environ["AS_HELP_METADATA_DIR"] = str(Path(args.metadata_dir).resolve())
    if args.force_rebuild:
        os.environ["AS_HELP_FORCE_REBUILD"] = "true"
    if args.as_version:
        os.environ["AS_HELP_VERSION"] = args.as_version

    # Run with stdio transport by default (for local MCP clients like Claude Desktop)
    # To run as HTTP server, use: mcp.run(transport="streamable-http")
    mcp.run()


if __name__ == "__main__":
    main()
