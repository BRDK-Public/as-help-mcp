# B&R Help MCP Server - Copilot Instructions

## Project Overview

This is a **Model Context Protocol (MCP) server** that provides hybrid semantic + keyword search and retrieval for B&R Automation Studio help documentation. Built with Python 3.12+, FastMCP SDK, LanceDB for vector + FTS storage, and sentence-transformers for local embeddings.

**Key Architecture Decision:** LanceDB was chosen for hybrid search (vector + full-text) with Reciprocal Rank Fusion (RRF). Sentence-transformers (`all-MiniLM-L6-v2`) provides local embeddings — no API keys needed.

## Core Architecture

### Four-Layer Design

1. **`indexer.py`** - XML Parser & HTML Extractor
   - Parses `brhelpcontent.xml` (abbreviated tags: `S`=Section, `P`=Page, `t`=Text, `p`=File, `I`=Identifiers)
   - Builds in-memory page tree with parent-child relationships
   - Extracts breadcrumbs with **cycle detection** and **depth limit (100)**
   - Uses **lxml** for fast HTML text extraction (2-3x faster than BeautifulSoup)
   - Uses MD5 hash for change detection (stored in `_index_metadata.json` sidecar)

2. **`embeddings.py`** - Embedding Service
   - Lazy-loads sentence-transformer model on first use
   - Default model: `all-MiniLM-L6-v2` (384-dim, ~22MB download)
   - Configurable via `AS_HELP_EMBEDDING_MODEL` env var
   - `embed_text()` for single texts, `embed_batch()` for bulk embedding
   - 2048 char truncation for long texts

3. **`search_engine.py`** - LanceDB Hybrid Search with RRF
   - **Three search legs** fused via Reciprocal Rank Fusion (RRF, k=60):
     - Title vector similarity (weight 2x)
     - Content vector similarity (weight 1x)
     - Full-text keyword search (weight 1.5x)
   - LanceDB directory-based storage (`.ashelp_lance/`)
   - PyArrow schema: page metadata + 2 vector columns + combined `search_text` for FTS
   - **Query sanitization** for FTS special characters
   - Parallel text extraction using ThreadPoolExecutor
   - Metadata sidecar (`_index_metadata.json`) tracks XML hash + embedding model
   - Accepts optional `embedding_service` parameter for testability

3. **`server.py`** - FastMCP Server
   - Exposes 5 tools: `search_help`, `get_page_by_id`, `get_page_by_help_id`, `get_breadcrumb`, `get_help_statistics`
   - **Intentionally truncated previews** (~100 chars) to force LLM to call `get_page_by_id`
   - Server instructions guide LLM to make **multiple searches and page retrievals**
   - Uses Pydantic models for structured responses
   - Resource endpoint: `help://page/{page_id}` for direct HTML access

### Data Flow

```
brhelpcontent.xml → Indexer → Page Tree (in-memory)
                        ↓
                  HTML Files → lxml → Plain Text
                        ↓
             sentence-transformers → Embeddings (title + content vectors)
                        ↓
                  LanceDB → Table + FTS Index → RRF Hybrid Search → MCP Tools
```

### Search Ranking (RRF)

Results are ranked using Reciprocal Rank Fusion with three search legs:
- **Title vector** (weight 2x): Semantic similarity between query and title embeddings
- **Content vector** (weight 1x): Semantic similarity between query and content embeddings
- **FTS keyword** (weight 1.5x): Tantivy-powered full-text search on combined title+content
- **RRF formula**: `score = Σ weight / (k + rank + 1)` where `k=60`
- Higher score = better match

## Development Workflows

### First-Time Setup

**Docker (Production/Distribution):**
```bash
docker build -t as-help:local .
# See README.md for complete Docker guide
```

**Local Development:**
```bash
# Auto-setup with script (recommended)
./setup.sh

# Manual setup
uv sync  # Creates .venv and installs deps
# Create .env with: AS_HELP_ROOT=/mnt/c/BRAutomation/AS412/Help-en/Data
```

### Testing in VS Code (Recommended)

Use **Run and Debug (Ctrl+Shift+D)** - NOT MCP Inspector (stdio issues on Windows):

1. **"Rebuild BR Help Index"** - First run only 
2. **"Run BR Help MCP Server"** - Fast startup with existing index (<1s)
3. **"Test BR Help Indexer"** - Quick XML parse test (~1s)

**Why not terminal?** Server uses stdio transport for MCP clients. Debugger provides interactive testing without needing a full MCP client.

### Quick Validation

```bash
# Test XML parsing only (1 second)
uv run python test_parsing.py  # Outputs: "Total pages: 107332"

# Test search functionality
uv run python test_search.py   # Runs sample queries
```

## Critical Conventions

### Environment Variables (Required)

- `AS_HELP_ROOT` - Path to `Data` folder containing `brhelpcontent.xml`
  - WSL: `/mnt/c/BRAutomation/AS412/Help-en/Data`
  - Windows: `C:\BRAutomation\AS412\Help-en\Data`
- `AS_HELP_FORCE_REBUILD` - Set `true` for first run, then `false` (auto-rebuild on XML changes)
- `AS_HELP_DB_PATH` - Optional custom DB location (defaults to `{AS_HELP_ROOT}/.ashelp_lance`)

### Abbreviated XML Tags (Critical!)

The B&R XML uses shortened tags - **both formats must be handled**:
- `Section` or `S` (with `Text`/`t`, `File`/`p`)
- `Page` or `P` (with `Text`/`t`, `File`/`p`)
- `Identifiers` or `I` → `HelpID` or `H` (with `Value`/`v`)

See `_process_section()` and `_process_page()` in `indexer.py` for implementation.

### Index Rebuild Logic

**Three-tier strategy** — avoids unnecessary full rebuilds:

1. **No change** (most starts): XML hash matches + model unchanged → load index (<3s)
2. **Incremental update** (AS service packs): XML hash changed, page fingerprints exist in metadata
   - Diffs per-page fingerprints (`hash(title|file_path|parent_id|help_id|is_section)`)
   - Deletes removed/changed rows from LanceDB, embeds & adds new/changed rows
   - Falls back to full rebuild if >50% of pages changed
   - Rebuilds FTS index after mutations
3. **Full rebuild** (first run, model change, legacy metadata): Re-extracts, re-embeds, overwrites LanceDB table

Per-page fingerprints are stored in `_index_metadata.json` alongside the XML hash and embedding model.

See `_detect_build_strategy()` and `_incremental_update()` in `search_engine.py`.

### Content Extraction Strategy

- **Sections**: Index title only (no HTML content)
- **Pages**: Index title + full plain text from HTML
- **Lazy loading**: HTML/text extracted on-demand and cached in `HelpPage` objects
- **lxml**: Uses `text_content()` for fast extraction, strips `<script>` and `<style>` via XPath

## Integration Points

### MCP Client Configuration

Add to client config (e.g., `claude_desktop_config.json` or VS Code settings):

```json
{
  "mcpServers": {
    "as-help": {
      "command": "uv",
      "args": ["run", "as-help-server"],
      "cwd": "/home/username/projects/as-help",
      "env": {
        "AS_HELP_ROOT": "/mnt/c/BRAutomation/AS412/Help-en/Data",
        "AS_HELP_FORCE_REBUILD": "false"
      }
    }
  }
}
```

### Entry Points

- `uv run as-help-server` - Script entry (defined in `pyproject.toml`)
- `python src/server.py` - Direct execution (set PYTHONPATH or run from src/)

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| XML parse | ~2s | pages in-memory |
| First index build | 10-11 min | Parallel HTML extraction + embedding + FTS indexing |
| Subsequent startup | <3s | Load existing DB |
| Search query | 10-50ms | RRF hybrid search |
| Memory usage | 10-30MB | Runtime after index load |

**Log progress every 5000 docs** during indexing to show it's working.

## Common Pitfalls

1. **Don't test with MCP Inspector on Windows** - stdio transport has compatibility issues. Use VS Code debugger.
2. **Always handle both XML tag formats** - Some XMLs use `Section`, others use `S`.
3. **Check metadata hash before rebuilding** - Rebuilding pages is unnecessary waste of time.
4. **Path handling** - B&R uses Windows paths (backslashes), ensure cross-platform Path usage.
5. **LanceDB storage** - Directory-based, no explicit close needed (see `close()` in `search_engine.py`).

## File Structure Patterns

```
src/
  __init__.py           # Package marker
  __main__.py           # Module entry point
  server.py             # FastMCP server + tools (main logic)
  indexer.py            # XML parsing + HTML extraction
  search_engine.py      # LanceDB hybrid search with RRF
  embeddings.py         # Sentence-transformer embedding service
  
Root level:
  test_*.py             # Standalone test scripts (no MCP server)
  setup.ps1             # Auto-setup for team distribution
  mcp.json              # Example MCP client config
  .env.example          # Template for environment vars
```

## Testing Strategy

- **Unit tests**: Use `test_parsing.py` (indexer only) and `test_search.py` (search only)
- **Integration**: Use VS Code debugger with breakpoints in `server.py` tools
- **Production**: Configure in MCP client and test via Copilot chat
- **Never**: Run MCP Inspector on Windows (stdio issues)

## Distribution to Team

Use `setup.sh` (WSL/Linux) or `setup.ps1` (Windows) for local installation, or use Docker for containerized deployment:

```bash
docker pull ghcr.io/YOUR_USERNAME/as-help:latest
```

See README.md for complete setup instructions.

## Key Dependencies

- `lancedb` - Vector + FTS database (LanceDB)
- `sentence-transformers` - Local embedding models
- `pyarrow` - Columnar data for LanceDB tables
- `lxml` - Fast HTML parsing (2-3x faster than BeautifulSoup)
- `python-dotenv` - .env file loading (optional, env vars work directly)

## When Modifying Code

- **Adding tools**: Add `@mcp.tool()` decorated function in `server.py` with Pydantic models
- **Changing XML parsing**: Update both tag formats in `indexer.py` (`Section`/`S`, etc.)
- **Search improvements**: Adjust RRF weights or add search legs in `search_engine.py`
- **Embedding model**: Change `AS_HELP_EMBEDDING_MODEL` env var or update default in `embeddings.py`
- **New metadata**: Update `_save_metadata()` and `_load_metadata()` in `indexer.py`
