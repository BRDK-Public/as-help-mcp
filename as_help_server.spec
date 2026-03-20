# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for B&R Help MCP Server.

Build with:
    pyinstaller as_help_server.spec

Or use the convenience script:
    uv run python build.py
"""

import sys
from pathlib import Path

block_cipher = None

# Collect all hidden imports needed by MCP SDK and dependencies
hidden_imports = [
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
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce binary size
        "tkinter",
        "matplotlib",
        "numpy",
        "pandas",
        "PIL",
        "scipy",
        "setuptools",
        "pip",
        "wheel",
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
