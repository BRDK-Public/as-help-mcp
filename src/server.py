"""B&R Automation Studio Help MCP Server.

This MCP server provides search and retrieval capabilities for B&R Automation Studio help content.
Optimized for thousands of help files with persistent indexing and fast startup.
"""

import asyncio
import logging
import os
from argparse import ArgumentTypeError
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
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


def _build_online_help_url(base_url: str, file_path: str | None) -> str | None:
    """Build normalized online help URL from a relative file path.

    Percent-encodes special characters (parentheses, spaces, etc.) in each
    path segment while preserving the '/' separators.
    """
    if not file_path:
        return None
    from urllib.parse import quote

    normalized_path = file_path.replace("\\", "/")
    # Encode each path segment individually to preserve '/' separators
    encoded_path = "/".join(quote(segment, safe="") for segment in normalized_path.split("/"))
    return f"{base_url}{encoded_path}"


def _parse_bool_arg(value: str) -> bool:
    """Parse strict boolean CLI values for argparse."""
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ArgumentTypeError(f"Invalid boolean value: '{value}'. Use true/false.")


# Configure logging - use stderr so it doesn't interfere with stdio transport
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# Pydantic models for structured output
class SearchResult(BaseModel):
    """A single search result - metadata only. Call get_page_by_id for actual content."""

    page_id: str = Field(description="REQUIRED: Use this with get_page_by_id to get the actual page content")
    title: str = Field(description="Title of the help page")
    file_path: str = Field(description="Relative path to the HTML file")
    online_help_url: str | None = Field(default=None, description="Direct link to B&R online help for this page")
    help_id: str | None = Field(default=None, description="HelpID if available")
    is_section: bool = Field(description="Whether this is a section (True) or page (False)")
    score: float | None = Field(default=None, description="Search relevance score (higher is better, RRF fusion)")
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
    search_mode: str | None = Field(
        default=None,
        description="Search mode: 'hybrid' (semantic + keyword, requires embeddings) or 'keyword' (FTS only, default)",
    )
    status_message: str | None = Field(
        default=None,
        description="Status message when index is not ready (e.g., building). Call get_help_statistics for details.",
    )


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

    # Get database path for LanceDB index directory
    # Default to /data/db for Docker volumes (not help root for read-only compatibility)
    if help_root.startswith("/data/"):
        default_db_path = "/data/db/.ashelp_lance"
    else:
        default_db_path = str(help_root_path / ".ashelp_lance")

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

    # Check for old SQLite index and log migration info
    old_sqlite_db = db_path.parent / ".ashelp_search.db" if db_path.name == ".ashelp_lance" else None
    if old_sqlite_db and old_sqlite_db.exists():
        logger.info(f"Found old SQLite FTS5 index at {old_sqlite_db} - it can be safely deleted")
        logger.info("Migrated from SQLite FTS5 to LanceDB search")

    # Optionally create embedding service for hybrid search
    create_embeddings = os.getenv("CREATE_EMBEDDINGS", "false").lower() == "true"
    embedding_service = None
    if create_embeddings:
        from src.embeddings import EmbeddingService

        logger.info("CREATE_EMBEDDINGS=true — initializing embedding API client...")
        try:
            embedding_service = EmbeddingService()
        except ValueError as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            logger.error("Set EMBEDDING_API_ENDPOINT, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS")
            raise
    else:
        logger.info("CREATE_EMBEDDINGS is not set — using FTS keyword search only")

    # Initialize search engine (constructor is fast — just connects to LanceDB)
    logger.info("Initializing search engine...")
    try:
        search_engine = HelpSearchEngine(
            db_path=db_path, indexer=indexer, force_rebuild=force_rebuild,
            embedding_service=embedding_service,
        )
    except Exception:
        if embedding_service is not None:
            embedding_service.close()
        raise

    # Build/load the index in a background thread so the MCP server can respond
    # to initialize immediately. The search tool will wait for readiness.
    # Store the future so we can await it during shutdown for a clean exit.
    build_type = search_engine.build_status["build_type"]
    init_future = asyncio.get_running_loop().run_in_executor(None, search_engine.initialize)

    if build_type == "none":
        logger.info("=== Server ready (loading existing index) ===")
    elif build_type == "resume":
        logger.info("=== Server ready (resuming interrupted index build in background) ===")
    elif build_type == "incremental":
        logger.info("=== Server ready (incremental index update running in background) ===")
    else:
        logger.info(
            f"=== Server ready ({build_type} index build running in background - this may take several minutes) ==="
        )

    # Yield context to the application
    context = AppContext(
        indexer=indexer, search_engine=search_engine, as_version=as_version, online_help_base_url=online_help_base_url
    )
    yield context

    # Wait for the background index thread to finish before exiting.
    # Exceptions are already captured in search_engine._build_error, so suppress
    # them here to avoid duplicate error reporting and asyncio warnings.
    # Log at debug level so unexpected errors remain discoverable.
    logger.info("Waiting for background index thread to complete...")
    try:
        await init_future
    except Exception as exc:
        logger.debug("Background index thread raised an exception during teardown: %s", exc)

    search_engine.close()
    logger.info("Shutting down help server")


# Create FastMCP server with lifespan management
mcp = FastMCP(  # pragma: no cover
    "B&R Automation Studio Help Server",  # pragma: no cover
    instructions=(  # pragma: no cover
        "B&R Automation Studio Help Server - 100k+ pages of technical documentation.\n\n"  # pragma: no cover
        "CRITICAL: content_preview is ~100 chars. NEVER answer from previews alone. "  # pragma: no cover
        "You MUST call get_page_by_id to read actual documentation.\n\n"  # pragma: no cover
        "*** RESEARCH WORKFLOW ***\n\n"  # pragma: no cover
        "1. search_help — Find pages by keyword or meaning. Returns titles/page_ids only, NO content.\n"  # pragma: no cover
        "2. get_page_by_id — Get FULL content. Use breadcrumb_path from results to pick relevant pages "  # pragma: no cover
        "and skip obvious mismatches (e.g., wrong library variant).\n"  # pragma: no cover
        "3. REPEAT — Search with different keywords or synonyms. Complex questions need 2-5 page retrievals.\n"  # pragma: no cover
        "4. get_page_by_help_id — Use when you have a numeric HelpID (e.g., from error codes or context help).\n\n"  # pragma: no cover
        "*** DISCOVERY & BROWSING ***\n\n"  # pragma: no cover
        "- get_categories — List top-level categories. Call BEFORE using the category filter in search_help.\n"  # pragma: no cover
        "- browse_section — Navigate into a category/section to see its children. "  # pragma: no cover
        "Prefer search_help for direct lookups; use browse_section only to explore structure or find siblings.\n\n"  # pragma: no cover
        "TIPS:\n"  # pragma: no cover
        "- Use specific identifiers when known (e.g., 'MC_BR_MoveAbsolute', 'X20DI9371')\n"  # pragma: no cover
        "- Try keyword variations: 'axis move' vs 'motion control' vs 'positioning'\n"  # pragma: no cover
        "- get_help_statistics — Check index build progress if search returns empty results"  # pragma: no cover
    ),  # pragma: no cover
    lifespan=app_lifespan,  # pragma: no cover
)  # pragma: no cover


@mcp.tool()
def search_help(
    ctx: Context,
    query: str = Field(description="Search query — use specific identifiers (e.g., 'MC_BR_MoveAbsolute') or natural language (e.g., 'how to move an axis'). Try different keywords for better coverage."),
    limit: int = Field(
        default=5,
        description="Max results to return. Use smaller limits with multiple searches rather than one large search.",
    ),
    content_search: bool = Field(default=True, description="True = search titles + content (default). False = titles only (faster, use for known identifiers)."),
    category: str | None = Field(
        default=None,
        description="Filter by top-level category. MUST call get_categories() first to get valid values — do not guess category names.",
    ),
) -> SearchResults:
    """Find help pages by keyword or meaning. Returns page_ids and metadata ONLY — no actual content.

    IMPORTANT: You MUST call get_page_by_id to read content. Previews (~100 chars) are NOT enough to answer questions.

    Use breadcrumb_path in results to assess relevance before retrieving — skip results whose breadcrumb
    clearly shows they are about a different library, module, or topic.

    If search returns empty results, call get_help_statistics to check if the index is still building.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Return immediately with status when the index is not queryable.
    # fts_ready allows keyword-only search while vectors are still building.
    status = app_ctx.search_engine.build_status
    state = status.get("state")
    if state not in ("ready", "fts_ready"):
        phase = status.get("phase", "")
        build_type = status.get("build_type", "unknown")
        processed = status.get("pages_processed", 0)
        total = status.get("pages_total", 0)
        elapsed = status.get("elapsed_seconds")

        progress = f"{processed}/{total} pages" if total else phase
        elapsed_str = f", {elapsed:.0f}s elapsed" if elapsed else ""

        if status.get("error"):
            msg = f"Index build failed: {status['error']}"
        else:
            msg = (
                f"Search index is building ({build_type}): {phase} - {progress}{elapsed_str}. "
                f"Call get_help_statistics to check progress, then retry search."
            )

        logger.info(f"Search called while building: {msg}")
        return SearchResults(query=query, results=[], total=0, status_message=msg)

    # Handle FieldInfo objects when function called directly (not through FastMCP framework)
    # This supports both MCP tool invocation and direct test calls
    from pydantic.fields import FieldInfo

    if isinstance(limit, FieldInfo):
        limit = limit.default or 5
    if isinstance(content_search, FieldInfo):
        content_search = content_search.default if content_search.default is not None else True
    if isinstance(category, FieldInfo):
        category = category.default

    # Perform search (returns breadcrumb_path directly from LanceDB index)
    results = app_ctx.search_engine.search(
        query=query, limit=limit, search_in_content=content_search, category=category
    )

    # Convert to SearchResult models
    search_results = []
    for r in results:
        # Use snippet precomputed by search engine to avoid extra disk I/O per result.
        # Keep previews intentionally short and incomplete to drive get_page_by_id usage.
        content_preview = None
        snippet = r.get("snippet")
        if snippet:
            content_preview = snippet.strip() + "... [TRUNCATED - call get_page_by_id for full content]"

        # Build online help URL from file path
        online_help_url = _build_online_help_url(app_ctx.online_help_base_url, r.get("file_path"))

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

    # Determine search mode from engine state, not from result data.
    # This ensures correct reporting even when the result set is empty.
    engine = app_ctx.search_engine
    if engine._embeddings_enabled and engine.ready:
        search_mode = "hybrid"
    else:
        search_mode = "keyword"

    # Add status message only when embeddings are enabled but not yet ready
    # (i.e., hybrid mode is expected but temporarily degraded to keyword-only)
    status_msg = None
    if search_mode == "keyword" and app_ctx.search_engine._embeddings_enabled:
        status_msg = (
            "Semantic search is still loading — results are keyword-only. Retry later for better relevance ranking."
        )

    return SearchResults(
        query=query,
        results=search_results,
        total=len(search_results),
        search_mode=search_mode,
        status_message=status_msg,
    )


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
        description="Section ID from get_categories or a previous browse_section call."
    ),
) -> SectionChildren | None:
    """Browse the children of a section in the help tree.

    Use browse_section to explore structure or find sibling/related pages within a section.
    Prefer search_help for direct topic lookups — it is faster and more precise.

    Workflow:
    1. get_categories() → get top-level category IDs
    2. browse_section(category_id) → see children
    3. Sections (is_section=True) → browse deeper. Pages (is_section=False) → get_page_by_id.
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
    page_id: str = Field(description="Page ID from search results."),
    include_html: bool = Field(default=False, description="Include raw HTML (only needed for rendering or link extraction)."),
    include_text: bool = Field(default=True, description="Include full plain text content."),
    include_breadcrumb: bool = Field(default=True, description="Include navigation breadcrumb path."),
) -> PageContent | None:
    """Get the COMPLETE content of a help page.

    Before calling, check the breadcrumb_path from search results to confirm the page is relevant.
    Skip pages whose breadcrumb shows they belong to a different library or unrelated topic.

    For thorough answers, retrieve 2-5 pages: the main topic plus related pages
    (examples, error handling, configuration). Cross-reference across pages.
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
    online_help_url = _build_online_help_url(app_ctx.online_help_base_url, page.file_path)

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
    help_id: str = Field(description="Numeric HelpID value (e.g., '3002099'). Found in error messages, context help, and AS project references."),
    include_html: bool = Field(default=False, description="Include raw HTML."),
    include_text: bool = Field(default=True, description="Include plain text content."),
    include_breadcrumb: bool = Field(
        default=True, description="Include breadcrumb trail."
    ),
) -> PageContent | None:
    """Retrieve a help page by its numeric HelpID.

    Use this when you have a HelpID from error codes, context-sensitive help links,
    or AS project references. Returns the same content as get_page_by_id.
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
    online_help_url = _build_online_help_url(app_ctx.online_help_base_url, page.file_path)

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
async def get_help_statistics(ctx: Context) -> dict:
    """Check index build progress and get content statistics.

    Primary use: check if the search index is ready (state='ready' or 'fts_ready').
    Call this when search_help returns empty results to see if the index is still building.
    Also returns counts of total pages, sections, and HelpID mappings.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    total_pages = len(app_ctx.indexer.pages)
    total_sections = sum(1 for p in app_ctx.indexer.pages.values() if p.is_section)
    total_help_ids = len(app_ctx.indexer.help_id_map)

    # Check parent-child relationships
    pages_with_parents = sum(1 for p in app_ctx.indexer.pages.values() if p.parent_id is not None)
    root_pages = sum(1 for p in app_ctx.indexer.pages.values() if p.parent_id is None)

    # Get index build status
    build_status = app_ctx.search_engine.build_status

    await ctx.info(f"Statistics: {total_pages} total, {total_sections} sections, {total_help_ids} HelpIDs")
    await ctx.info(f"Hierarchy: {pages_with_parents} with parents, {root_pages} root items")
    await ctx.info(
        f"Index: state={build_status['state']}, type={build_status['build_type']}, phase={build_status['phase']}"
    )

    result: dict = {
        "total_pages": total_pages,
        "total_sections": total_sections,
        "regular_pages": total_pages - total_sections,
        "help_id_mappings": total_help_ids,
        "pages_with_parents": pages_with_parents,
        "root_items": root_pages,
        "index_status": {
            "state": build_status["state"],
            "build_type": build_status["build_type"],
            "phase": build_status["phase"],
            "pages_total": build_status["pages_total"],
            "pages_processed": build_status["pages_processed"],
            "elapsed_seconds": build_status["elapsed_seconds"],
        },
    }

    # Include incremental stats when available
    if build_status.get("incremental_stats"):
        result["index_status"]["incremental_stats"] = build_status["incremental_stats"]

    if build_status.get("error"):
        result["index_status"]["error"] = build_status["error"]

    return result


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
    parser.add_argument(
        "--force-rebuild",
        type=_parse_bool_arg,
        nargs="?",
        const=True,
        default=None,
        metavar="BOOL",
        help="Force index rebuild: true/false (AS_HELP_FORCE_REBUILD). Omit value for true.",
    )
    parser.add_argument(
        "--as-version",
        choices=["4", "6"],
        help="AS version for online help (AS_HELP_VERSION). Default: 4",
    )
    parser.add_argument(
        "--create-embeddings",
        type=_parse_bool_arg,
        nargs="?",
        const=True,
        default=None,
        metavar="BOOL",
        help="Enable embedding via API: true/false (CREATE_EMBEDDINGS). Omit value for true.",
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
    if args.force_rebuild is not None:
        os.environ["AS_HELP_FORCE_REBUILD"] = "true" if args.force_rebuild else "false"
    if args.as_version:
        os.environ["AS_HELP_VERSION"] = args.as_version
    if args.create_embeddings is not None:
        os.environ["CREATE_EMBEDDINGS"] = "true" if args.create_embeddings else "false"

    # Run with stdio transport by default (for local MCP clients like Claude Desktop)
    # To expose over HTTP, set MCP_TRANSPORT=streamable-http and configure host/port with MCP_HOST/MCP_PORT
    # Default host is 127.0.0.1 (localhost only); set MCP_HOST=0.0.0.0 to bind all interfaces
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
        # Only disable DNS rebinding protection when explicitly opted in
        if os.environ.get("MCP_DISABLE_DNS_REBINDING_PROTECTION", "false").lower() == "true":
            logger.warning("DNS rebinding protection is DISABLED — only safe behind a reverse proxy or firewall")
            mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
