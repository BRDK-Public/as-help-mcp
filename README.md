# AS Help MCP Server

MCP server for B&R Automation Studio help documentation search. Provides keyword search by default using LanceDB with Tantivy FTS, and optional hybrid semantic + keyword search using Reciprocal Rank Fusion (RRF) when an embedding API is configured.

## Features

- **Keyword search** (default): Fast full-text search using LanceDB + Tantivy — no external dependencies
- **Hybrid search** (optional): RRF fusion of vector similarity and keyword matching when embeddings are enabled
- **API-based embeddings**: Works with any OpenAI-compatible endpoint (Ollama, OpenAI, Azure OpenAI, GitHub Models, LiteLLM) — no local ML models required
- **Smart ranking**: Query-type detection shifts weights between FTS and vectors (identifiers like `MC_MoveAbsolute` favor exact match; natural language favors semantic similarity)
- Category filtering and hierarchical browsing
- Auto-generated links to B&R online help (AS4/AS6)
- HelpID lookup for context-sensitive help integration
- Incremental reindexing — only changed pages are re-processed
- Two-phase build: keyword search available within minutes while embeddings build in the background

## Prerequisites

- B&R Automation Studio installed (with help documentation)
- VS Code with GitHub Copilot extension
- **For standalone binary:** Download `as-help-server.exe` from [Releases](../../releases) — no Python or Docker required
- **For UV:** Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- **For Docker:** Docker Desktop
- **Optional** (for hybrid search): An OpenAI-compatible embedding API (e.g., [Ollama](https://ollama.com/) with `nomic-embed-text`)

## Demo
https://github.com/user-attachments/assets/b4df6bc7-ed7c-471f-93b8-db84b0110ac3

## Quick Start (VS Code)

Add to `.vscode/mcp.json` in your workspace:

### Option 1: Standalone Binary (Recommended)

No Python, no Docker — just download the `.exe` from [Releases](../../releases) and place it in `%APPDATA%\as-help-mcp\`.

```json
{
  "servers": {
    "as-help": {
      "command": "${env:APPDATA}\\as-help-mcp\\as-help-server.exe",
      "args": [
        "--help-root",
        "C:\\Program Files (x86)\\BRAutomation\\AS6\\Help-en\\Data",
        "--db-path",
        "${env:APPDATA}\\as-help-mcp\\data\\as6\\.ashelp_lance",
        "--metadata-dir",
        "${env:APPDATA}\\as-help-mcp\\data\\as6\\.ashelp_metadata",
        "--as-version",
        "6"
      ]
    }
  }
}
```

Update `--help-root` to match your AS installation:
- **AS 4.x:** `C:\\BRAutomation\\AS412\\Help-en\\Data`
- **AS 6.x:** `C:\\Program Files (x86)\\BRAutomation\\AS6\\Help-en\\Data`

### Option 2: Docker

```json
{
  "servers": {
    "as-help": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "C:\\Program Files (x86)\\BRAutomation\\AS6\\Help-en\\Data:/data/help:ro",
        "-v", "ashelp-data:/data/db",
        "-e", "AS_HELP_VERSION=6",
        "-e", "AS_HELP_FORCE_REBUILD=false",
        "ghcr.io/brdk-public/as-help-mcp:latest"
      ]
    }
  }
}
```

Update the volume path to match your AS installation:
- **AS 4.x:** `C:\\BRAutomation\\AS412\\Help-en\\Data:/data/help:ro`
- **AS 6.x:** `C:\\Program Files (x86)\\BRAutomation\\AS6\\Help-en\\Data:/data/help:ro`
- **AS 6.x in WSL:** `/mnt/c/Program Files (x86)/BRAutomation/AS6/Help-en/Data:/data/help:ro`

### Option 3: UV (Local Development)

```json
{
  "servers": {
    "as-help": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/as-help-mcp",
        "as-help-server",
        "--help-root",
        "C:\\Program Files (x86)\\BRAutomation\\AS6\\Help-en\\Data",
        "--db-path",
        "..\\data\\as6\\.ashelp_lance",
        "--metadata-dir",
        "..\\data\\as6\\.ashelp_metadata",
        "--as-version",
        "6"
      ]
    }
  }
}
```

Update `--directory` to point to your cloned repository and adjust paths as needed.

---

Restart VS Code, then test in Copilot Chat: *"Search AS help for mapp Motion"*

**First run takes 2-3 minutes** to build the keyword search index. With embeddings enabled, a full hybrid build takes 15-20 minutes (keyword search is available immediately while embeddings build in the background). Subsequent starts are instant (~3s).

---

## Enabling Hybrid Search (Optional)

By default, the server uses keyword-only search (FTS). To enable hybrid semantic + keyword search, configure an OpenAI-compatible embedding API.

### Example: Ollama (Local, Free)

1. Install [Ollama](https://ollama.com/) and pull an embedding model:

```bash
ollama pull nomic-embed-text
```

2. Add `--create-embeddings true` and embedding environment variables to your MCP config:

```json
{
  "servers": {
    "as-help": {
      "command": "${env:APPDATA}\\as-help-mcp\\as-help-server.exe",
      "args": [
        "--help-root", "C:\\Program Files (x86)\\BRAutomation\\AS6\\Help-en\\Data",
        "--db-path", "${env:APPDATA}\\as-help-mcp\\data\\as6\\.ashelp_lance",
        "--metadata-dir", "${env:APPDATA}\\as-help-mcp\\data\\as6\\.ashelp_metadata",
        "--as-version", "6",
        "--create-embeddings", "true"
      ],
      "env": {
        "EMBEDDING_API_ENDPOINT": "http://localhost:11434",
        "EMBEDDING_API_KEY": "ollama",
        "EMBEDDING_MODEL": "nomic-embed-text",
        "EMBEDDING_DIMENSIONS": "768",
        "EMBEDDING_BATCH_SIZE": "100",
        "EMBEDDING_MAX_CHARS": "4000"
      }
    }
  }
}
```

Any OpenAI-compatible endpoint works — OpenAI, Azure OpenAI, GitHub Models, LiteLLM, etc. Just update the endpoint, key, model, and dimensions accordingly.

### How Hybrid Search Works

When embeddings are enabled, the server uses **Reciprocal Rank Fusion (RRF)** to combine four search signals:

| Signal | NL Weight | ID Weight | Description |
|--------|-----------|-----------|-------------|
| Title vector | 2.0 | 0.5 | Semantic similarity between query and title+breadcrumb embeddings |
| Content vector | 1.0 | 0.5 | Semantic similarity between query and breadcrumb+content embeddings |
| FTS keyword | 1.5 | 3.0 | Tantivy full-text search on title+breadcrumb+content |
| Title match | 3.0 | 4.0 | Exact/substring match of query in page titles |

**Query-type detection** automatically selects weights: identifier queries (e.g., `MC_MoveAbsolute`, `X20DI9371`) shift toward FTS + title match; natural language queries favor vector similarity.

For a deep dive into the RAG architecture — chunking strategy, two-phase build, RRF fusion, embedding model choice, and alternatives considered — see **[RAG.md](RAG.md)**.

---

## Local Development Setup

### Option 1: UV (Recommended)

```bash
# Clone and install
git clone <repository-url>
cd as-help-mcp
uv sync --extra test --extra dev

# Run server with command line arguments (precedence over .env)
uv run as-help-server --help-root "C:\BRAutomation\AS412\Help-en\Data" --db-path "data\.ashelp_lance" --metadata-dir "data\.ashelp_metadata"

# Or use relative paths (automatically resolved)
uv run as-help-server --db-path ./data/lance_index --metadata-dir ./data
```

### Option 2: Environment Variables (.env)

You can also create a `.env` file in the root directory. Command-line arguments will override these values if provided.

```bash
AS_HELP_ROOT=C:\Program Files (x86)\BRAutomation\AS6\Help-en\Data
AS_HELP_VERSION=6
```

### CLI Arguments

Run `uv run as-help-server --help` for full details.

| Argument | Env Var Equivalent | Description |
|----------|--------------------|-------------|
| `--help-root` | `AS_HELP_ROOT` | Path to AS Help Data folder |
| `--db-path` | `AS_HELP_DB_PATH` | Path to the LanceDB directory |
| `--metadata-dir` | `AS_HELP_METADATA_DIR` | Path to the indexing metadata directory |
| `--as-version` | `AS_HELP_VERSION` | AS version for online help (`4` or `6`) |
| `--force-rebuild` | `AS_HELP_FORCE_REBUILD` | Force a full index rebuild |
| `--create-embeddings` | `CREATE_EMBEDDINGS` | Enable API-based embeddings for hybrid search |

### Embedding Configuration (Environment Variables)

These are only needed when `--create-embeddings true` is set:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_API_ENDPOINT` | *(required)* | Base URL of OpenAI-compatible API |
| `EMBEDDING_API_KEY` | *(required)* | API key / bearer token |
| `EMBEDDING_MODEL` | *(required)* | Model name (e.g., `nomic-embed-text`, `text-embedding-3-small`) |
| `EMBEDDING_DIMENSIONS` | *(required)* | Vector dimensions (e.g., `768`, `1536`) |
| `EMBEDDING_BATCH_SIZE` | `100` | Texts per API call |
| `EMBEDDING_MAX_CHARS` | `8000` | Truncate input texts to this length |

### Option 3: Docker Compose

```bash
# Local build
docker compose build

# Run with your help files mounted
docker compose run --rm \
  -v "C:\Program Files (x86)\BRAutomation\AS6\Help-en\Data:/data/help:ro" \
  as-help-local
```

### Testing with MCP Inspector

The MCP Inspector provides a web UI for testing tools and prompts:

```bash
# With UV
uv run mcp dev src/server.py

# Opens browser at http://localhost:5173
```

Note: On Windows, use VS Code's Run and Debug panel instead (stdio transport issues with Inspector).

### VS Code Debugging

Use the launch configurations in `.vscode/launch.json`:

1. **Rebuild BR Help Index** - First run to build index
2. **Run BR Help MCP Server** - Normal server startup
3. **Test BR Help Indexer** - Quick XML parse test

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| XML parse | ~2s | 58K+ pages in-memory |
| First index build (FTS-only) | ~2-3 min | Parallel HTML extraction + FTS indexing |
| First index build (hybrid) | 15-20 min | + embedding via API (keyword search available immediately) |
| Subsequent startup | ~3s | Load existing index |
| Search query | 10-50ms | RRF hybrid or FTS keyword |
| Memory usage | 10-30MB | Runtime after index load |

---

## Tools

| Tool | Description |
|------|-------------|
| `search_help` | Hybrid semantic + keyword search with RRF ranking and optional category filter |
| `get_categories` | List top-level categories for filtering |
| `browse_section` | Navigate help tree hierarchically |
| `get_page_by_id` | Get full page content |
| `get_page_by_help_id` | Retrieve page by numeric HelpID |
| `get_breadcrumb` | Get navigation path |
| `get_help_statistics` | Get content and index build statistics |

## Prompts

| Prompt | Description |
|--------|-------------|
| `help_search` | Structured search with page IDs, breadcrumbs, and HelpIDs |
| `help_details` | Deep research with content synthesis from multiple pages |

---

## Multiple AS Versions

```json
{
  "servers": {
    "as-help-4": {
      "command": "${env:APPDATA}\\as-help-mcp\\as-help-server.exe",
      "args": [
        "--help-root", "C:\\BRAutomation\\AS412\\Help-en\\Data",
        "--db-path", "${env:APPDATA}\\as-help-mcp\\data\\as4\\.ashelp_lance",
        "--metadata-dir", "${env:APPDATA}\\as-help-mcp\\data\\as4\\.ashelp_metadata",
        "--as-version", "4"
      ]
    },
    "as-help-6": {
      "command": "${env:APPDATA}\\as-help-mcp\\as-help-server.exe",
      "args": [
        "--help-root", "C:\\Program Files (x86)\\BRAutomation\\AS6\\Help-en\\Data",
        "--db-path", "${env:APPDATA}\\as-help-mcp\\data\\as6\\.ashelp_lance",
        "--metadata-dir", "${env:APPDATA}\\as-help-mcp\\data\\as6\\.ashelp_metadata",
        "--as-version", "6"
      ]
    }
  }
}
```

