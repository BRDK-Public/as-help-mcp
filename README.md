# AS Help MCP Server

MCP server for B&R Automation Studio help documentation search. Provides hybrid semantic + keyword search using LanceDB with Reciprocal Rank Fusion (RRF).

## Features

- Hybrid search combining vector similarity and keyword matching via RRF
- Local sentence-transformer embeddings (no API keys needed)
- Category filtering and hierarchical browsing
- Auto-generated links to B&R online help (AS4/AS6)
- HelpID lookup for context-sensitive help integration
- Auto-reindex on changes in the help directory via MD5 hash
- Parallel indexing with multiple threads

## Prerequisites

- B&R Automation Studio installed (with help documentation)
- Docker Desktop installed and running
- VS Code with GitHub Copilot extension

## Demo
https://github.com/user-attachments/assets/b4df6bc7-ed7c-471f-93b8-db84b0110ac3

## Quick Start (VS Code)

Add to `.vscode/mcp.json` in your workspace:

### Option 1: Docker (Recommended)

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

### Option 2: UV (Local Development)

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

**First run takes 10-15 minutes** to build the search index (includes downloading the embedding model ~22MB and generating embeddings). Subsequent starts are instant.

---

## Local Development Setup

### Option 1: UV (Recommended)

```bash
# Clone and install
git clone <repository-url>
cd as-help
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
| `--embedding-device` | `AS_HELP_EMBEDDING_DEVICE` | Optional embedding device override (`cpu`, `cuda`, `mps`) |
| `--embed-batch-size` | `AS_HELP_EMBED_BATCH_SIZE` | Optional embedding batch size override (integer > 0) |

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

## Environment Variables & CLI Arguments

The server supports configuration via both environment variables and command-line arguments. **CLI arguments take precedence.** Relative paths are automatically resolved to absolute paths.

| CLI Argument | Environment Variable | Default | Description |
|--------------|----------------------|---------|-------------|
| `--help-root` | `AS_HELP_ROOT` | `/data/help` | Path to AS Help Data folder |
| `--as-version` | `AS_HELP_VERSION` | `4` | AS version for online help (`4` or `6`) |
| `--force-rebuild` | `AS_HELP_FORCE_REBUILD` | `false` | Force index rebuild |
| `--db-path` | `AS_HELP_DB_PATH` | `{root}/.ashelp_lance` | LanceDB directory |
| `--metadata-dir` | `AS_HELP_METADATA_DIR` | `{root}/.ashelp_metadata` | Metadata directory |
| `--embedding-device` | `AS_HELP_EMBEDDING_DEVICE` | `auto` | Embedding device: auto-detect CUDA/MPS/CPU (override optional) |
| `--embed-batch-size` | `AS_HELP_EMBED_BATCH_SIZE` | `auto` | Embedding batch size (auto if unset) |

## Performance And GPU Acceleration

By default, the server installs **CPU-only PyTorch** (~200 MB) and runs embeddings on CPU. This works out of the box with no extra configuration.

### Optional: Enable CUDA GPU Acceleration

If you have an NVIDIA GPU and want faster embedding (especially for first-time index builds), install with the `cuda` extra:

```bash
uv sync --extra cuda
```

This downloads the CUDA-enabled PyTorch (~2.4 GiB) and enables GPU-accelerated embeddings. The server auto-detects the best device at startup:

1. `cuda` if available (requires `--extra cuda` install)
2. `mps` on Apple Silicon
3. `cpu` fallback

Optional overrides are available for troubleshooting or benchmarking:

```bash
uv run as-help-server --embedding-device cuda --embed-batch-size 512
```

Recommended batch sizes:

- GPU (`cuda`): start with `--embed-batch-size 512` (increase if VRAM allows)
- CPU: start with `--embed-batch-size 128`

## Troubleshooting Slow Model Import

If startup repeatedly shows:

`Still loading embedding model (importing sentence_transformers)...`

for a long time, use this checklist:

1. Ensure only one `as-help-server` process is running.
2. Wait for first-run import/download to complete once (cold environment can be slow).
3. Set `HF_TOKEN` to improve Hugging Face download rate limits.
4. Temporarily force CPU to verify GPU stack health:

```bash
uv run as-help-server --embedding-device cpu
```

5. If CPU works and CUDA hangs, update NVIDIA driver / CUDA runtime / PyTorch CUDA build alignment.

The server now logs periodic model-load heartbeat messages and uses a cross-process build lock to avoid duplicate heavy rebuilds.

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
| `get_help_statistics` | Get content statistics |

## Prompts

| Prompt | Description |
|--------|-------------|
| `help_search` | Structured search with page IDs, breadcrumbs, and HelpIDs |
| `help_details` | Deep research with content synthesis from multiple pages |
| `search_hardware` | Filter: X20 modules, PLCs, drives, motors |
| `search_motion` | Filter: ACOPOS, mapp Motion, MC_* blocks |
| `search_visualization` | Filter: mapp View, widgets, HMI |
| `search_safety` | Filter: SafeLOGIC, safety functions |
| `search_vision` | Filter: mapp Vision, Smart Camera |
| `search_communication` | Filter: POWERLINK, OPC UA, Modbus |
| `search_programming` | Filter: IEC 61131-3, C/C++, libraries |
| `search_mapp_services` | Filter: AlarmX, Recipe, UserX |

## Resources

| URI | Description |
|-----|-------------|
| `help://page/{page_id}` | Plain text content |
| `help://html/{page_id}` | Raw HTML content |

---

## Multiple AS Versions

```json
{
  "servers": {
    "as-help-4": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "C:\\BRAutomation\\AS412\\Help-en\\Data:/data/help:ro",
        "-v", "ashelp-data-4:/data/db",
        "-e", "AS_HELP_VERSION=4",
        "-e", "AS_HELP_FORCE_REBUILD=false",
        "ghcr.io/brdk-public/as-help-mcp:latest"
      ]
    },
    "as-help-6": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "C:\\Program Files (x86)\\BRAutomation\\AS6\\Help-en\\Data:/data/help:ro",
        "-v", "ashelp-data-6:/data/db",
        "-e", "AS_HELP_VERSION=6",
        "-e", "AS_HELP_FORCE_REBUILD=false",
        "ghcr.io/brdk-public/as-help-mcp:latest"
      ]
    }
  }
}
```

