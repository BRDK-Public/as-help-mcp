# AS Help MCP Server

MCP server for B&R Automation Studio help documentation search. Provides full-text search across all help pages using SQLite FTS5 with BM25 ranking.

## Features

- Full-text search with BM25 ranking 
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
        "..\\data\\as6\\.ashelp\\search.db",
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

**First run takes 5-10 minutes** to build the search index. Subsequent starts are instant.

---

## Local Development Setup

### Option 1: UV (Recommended)

```bash
# Clone and install
git clone <repository-url>
cd as-help
uv sync --extra test --extra dev

# Run server with command line arguments (precedence over .env)
uv run as-help-server --help-root "C:\BRAutomation\AS412\Help-en\Data" --db-path "data\.ashelp\search.db" --metadata-dir "data\.ashelp_metadata"

# Or use relative paths (automatically resolved)
uv run as-help-server --db-path ./data/ashelp.db --metadata-dir ./data
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
| `--db-path` | `AS_HELP_DB_PATH` | Path to the SQLite database file |
| `--metadata-dir` | `AS_HELP_METADATA_DIR` | Path to the indexing metadata directory |
| `--as-version` | `AS_HELP_VERSION` | AS version for online help (`4` or `6`) |
| `--force-rebuild` | `AS_HELP_FORCE_REBUILD` | Force a full index rebuild |

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
| `--db-path` | `AS_HELP_DB_PATH` | `{root}/.ashelp_search.db` | Database location |
| `--metadata-dir` | `AS_HELP_METADATA_DIR` | `{root}/.ashelp_metadata` | Metadata directory |

---

## Tools

| Tool | Description |
|------|-------------|
| `search_help` | Full-text search with BM25 ranking and optional category filter |
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

