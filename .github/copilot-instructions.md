# B&R Help MCP Server - Copilot Instructions

## Project Overview

This is a **Model Context Protocol (MCP) server** that provides keyword and optional semantic search + retrieval for B&R Automation Studio help documentation. Built with Python 3.12+, FastMCP SDK, LanceDB for FTS + optional vector storage, and httpx for optional API-based embeddings.

**Key Architecture Decision:** LanceDB provides full-text search (FTS) by default with no external dependencies. When `CREATE_EMBEDDINGS=true`, the server calls an OpenAI-compatible embedding API to create vectors and enables hybrid search (RRF = Reciprocal Rank Fusion). No local ML models — embeddings are always API-based and optional.

## Core Architecture

### Four-Layer Design

1. **`indexer.py`** - XML Parser & HTML Extractor
   - Parses `brhelpcontent.xml` (abbreviated tags: `S`=Section, `P`=Page, `t`=Text, `p`=File, `I`=Identifiers)
   - Builds in-memory page tree with parent-child relationships
   - Extracts breadcrumbs with **cycle detection** and **depth limit (100)**
   - Uses **lxml** for fast HTML text extraction (2-3x faster than BeautifulSoup)
   - Uses MD5 hash for change detection (stored in `_index_metadata.json` sidecar)

2. **`embeddings.py`** - Optional API-Based Embedding Service
   - Only used when `CREATE_EMBEDDINGS=true`
   - Calls any **OpenAI-compatible** embedding API (OpenAI, Azure OpenAI, GitHub Models, Ollama, LiteLLM)
   - Configured via env vars: `EMBEDDING_API_ENDPOINT`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`
   - `embed_text()` for single texts, `embed_batch()` for bulk (with configurable batch size)
   - Automatic retry on 429/5xx with exponential backoff
   - Uses **httpx** (async-capable HTTP client) — no torch, no sentence-transformers

3. **`search_engine.py`** - LanceDB Dual-Mode Search Engine
   - **FTS-only mode** (default, `CREATE_EMBEDDINGS=false`):
     - Lance native full-text keyword search (BM25 ranking)
     - Tokenizer: stemming, stop-word removal, ASCII folding enabled
     - PyArrow schema: 9 columns (page metadata + `search_text` for FTS)
     - No vector columns — minimal storage overhead
   - **Hybrid mode** (`CREATE_EMBEDDINGS=true`):
     - Three search legs fused via RRF (k=60):
       - Title vector similarity (weight 2x)
       - Content vector similarity (weight 1x)
       - FTS keyword search (weight 1.5x)
     - PyArrow schema: 11 columns (adds `title_vector` + `content_vector`)
   - LanceDB directory-based storage (`.ashelp_lance/`)
   - **Query sanitization** for FTS special characters (shared between Lance native FTS and legacy Tantivy syntax)
   - Parallel text extraction using ThreadPoolExecutor
   - Metadata sidecar (`_index_metadata.json`) tracks XML hash, `embeddings_enabled`, and optional model info

4. **`server.py`** - FastMCP Server
   - Exposes tools: `search_help`, `get_page_by_id`, `get_page_by_help_id`, `get_breadcrumb`, `get_categories`, `browse_section`, `get_help_statistics`
   - **Intentionally truncated previews** (~100 chars) to force LLM to call `get_page_by_id`
   - Server instructions guide LLM to make **multiple searches and page retrievals**
   - Uses Pydantic models for structured responses
   - Reads `CREATE_EMBEDDINGS` env var to conditionally create `EmbeddingService`

### Data Flow

```
FTS-only mode (default):
  brhelpcontent.xml → Indexer → Page Tree (in-memory)
                          ↓
                    HTML Files → lxml → Plain Text
                          ↓
                    LanceDB → Table + FTS Index → Keyword Search → MCP Tools

Hybrid mode (CREATE_EMBEDDINGS=true):
  brhelpcontent.xml → Indexer → Page Tree (in-memory)
                          ↓
                    HTML Files → lxml → Plain Text
                          ↓
               Embedding API → Vectors (title + content)
                          ↓
                    LanceDB → Table + FTS Index + Vectors → RRF Hybrid Search → MCP Tools
```

### Search Ranking

**FTS-only mode:** Lance native BM25 keyword ranking on combined title+content text, with stemming/stop-word removal, over-fetching (limit×3) and title-match + breadcrumb-match reranking.

**Hybrid mode (RRF):**
- **Title vector** (weight 2x NL / 0.5x identifier): Semantic similarity between query and title+breadcrumb embeddings
- **Content vector** (weight 1x NL / 0.5x identifier): Semantic similarity between query and breadcrumb+content embeddings
- **FTS keyword** (weight 1.5x NL / 3x identifier): Lance native full-text search on title+breadcrumb+content
- **Title match** (weight 3x NL / 4x identifier): Exact/substring match of query in page titles
- **Breadcrumb match** (weight 2x NL / 3x identifier): Query terms found in breadcrumb path (helps pages with generic titles under relevant sections)
- **Query-type detection**: Identifier queries (e.g., `MC_MoveAbsolute`, `X20DI9371`) shift weights toward FTS+title+breadcrumb match; natural language queries favor vector similarity
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

### Embedding Variables (Optional — only when `CREATE_EMBEDDINGS=true`)

- `CREATE_EMBEDDINGS` - Master switch: `true` enables API-based embeddings + hybrid search
- `EMBEDDING_API_ENDPOINT` - Base URL of OpenAI-compatible embedding API
- `EMBEDDING_API_KEY` - API key (required)
- `EMBEDDING_MODEL` - Model name (e.g., `text-embedding-3-small`, `text-embedding-ada-002`)
- `EMBEDDING_DIMENSIONS` - Vector dimensions (e.g., `1536`, `384`)
- `EMBEDDING_BATCH_SIZE` - Texts per API call (default: 100)
- `EMBEDDING_MAX_CHARS` - Text truncation limit (default: 8000)
- `EMBEDDING_MAX_WORKERS` - Concurrent API calls (default: 4, set 1 for sequential)

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

**Mode switching:** Changing between FTS-only and hybrid mode (or changing the embedding model) triggers a full rebuild because the PyArrow schema differs (9 vs 11 columns).

See `_detect_build_strategy()` and `_incremental_update()` in `search_engine.py`.

### Content Extraction Strategy

- **Sections**: Index title + full plain text from HTML (when available)
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
| First index build (FTS-only) | ~2-3 min | Parallel HTML extraction + FTS indexing |
| First index build (hybrid) | 10-11 min | Parallel HTML extraction + embedding + FTS indexing |
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
  embeddings.py         # Optional API-based embedding service
  
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
- `httpx` - HTTP client for optional embedding API calls
- `pyarrow` - Columnar data for LanceDB tables
- `lxml` - Fast HTML parsing (2-3x faster than BeautifulSoup)
- `python-dotenv` - .env file loading (optional, env vars work directly)

## When Modifying Code

- **Adding tools**: Add `@mcp.tool()` decorated function in `server.py` with Pydantic models
- **Changing XML parsing**: Update both tag formats in `indexer.py` (`Section`/`S`, etc.)
- **Search improvements**: Adjust RRF weights or add search legs in `search_engine.py`
- **Embedding model**: Change `EMBEDDING_MODEL` env var — any OpenAI-compatible model works
- **New metadata**: Update `_save_metadata()` and `_load_metadata()` in `indexer.py`
