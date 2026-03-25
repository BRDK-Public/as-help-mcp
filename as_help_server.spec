# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for B&R Help MCP Server.

Build with:
    pyinstaller as_help_server.spec

Or use the convenience script:
    uv run python build.py
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

block_cipher = None

# Only use collect_submodules for smaller packages that PyInstaller doesn't have
# built-in hooks for. Let PyInstaller's own hooks handle torch, numpy, scipy, sklearn.
collected_submodules = []
for pkg in [
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "lancedb",
    "tokenizers",
    "tqdm",
]:
    collected_submodules += collect_submodules(pkg)

# Collect data files needed at runtime (e.g. model configs, version files)
extra_datas = []
for pkg in ["transformers", "huggingface_hub", "sentence_transformers", "lancedb"]:
    extra_datas += collect_data_files(pkg)

# Collect native shared libraries (.dll/.so/.dylib)
extra_binaries = []
for pkg in ["tokenizers", "lancedb"]:
    extra_binaries += collect_dynamic_libs(pkg)

hidden_imports = collected_submodules + [
    # MCP SDK internals
    "mcp",
    "mcp.server",
    "mcp.server.fastmcp",
    "mcp.server.stdio",
    "mcp.server.sse",
    "mcp.server.streamable_http",
    "mcp.types",
    "mcp.shared",
    "mcp.shared.exceptions",
    # ASGI / HTTP stack (used by MCP even in stdio mode)
    "starlette",
    "starlette.applications",
    "starlette.routing",
    "starlette.requests",
    "starlette.responses",
    "starlette.middleware",
    "anyio",
    "anyio._backends",
    "anyio._backends._asyncio",
    "httpx",
    "httpx._transports",
    "httpx._transports.default",
    "httpcore",
    "sniffio",
    "h11",
    "certifi",
    "idna",
    # Pydantic (dynamic model generation)
    "pydantic",
    "pydantic.fields",
    "pydantic._internal",
    "pydantic._internal._core_utils",
    "pydantic._internal._generate_schema",
    "pydantic._internal._validators",
    "pydantic_core",
    "annotated_types",
    "typing_extensions",
    # XML parsing
    "lxml",
    "lxml.etree",
    "lxml.html",
    "defusedxml",
    "defusedxml.ElementTree",
    # Standard library modules that PyInstaller may miss
    "sqlite3",
    "json",
    "hashlib",
    "logging",
    "xml.etree.ElementTree",
    # dotenv
    "dotenv",
    # Our source package
    "src",
    "src.server",
    "src.indexer",
    "src.search_engine",
    # Async support
    "asyncio",
    "concurrent.futures",
]

a = Analysis(
    ["src/server.py"],
    pathex=["."],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude UI/visualization modules not needed by a CLI server
        "tkinter",
        "matplotlib",
        "pandas",
        "PIL",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="as-help-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Must be True for stdio MCP transport
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
