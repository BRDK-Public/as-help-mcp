"""Build script for creating a standalone binary with PyInstaller.

Usage:
    uv run python build.py                  # Build using spec file
    uv run python build.py --clean          # Clean build artifacts first
    uv run python build.py --clean --deploy # Build and copy to %APPDATA%\as-help-mcp
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_build_artifacts():
    """Remove previous build artifacts."""
    dirs_to_clean = ["build", "dist"]
    for d in dirs_to_clean:
        p = Path(d)
        if p.exists():
            print(f"Removing {p}/")
            shutil.rmtree(p)


def build():
    """Run PyInstaller build."""
    spec_file = Path("as_help_server.spec")
    if not spec_file.exists():
        print(f"Error: {spec_file} not found. Run from project root.")
        sys.exit(1)

    print("Building as-help-server binary with PyInstaller...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec_file), "--noconfirm"],
        check=False,
    )

    if result.returncode != 0:
        print("Build failed!")
        sys.exit(result.returncode)

    output = Path("dist/as-help-server.exe") if sys.platform == "win32" else Path("dist/as-help-server")
    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"\nBuild successful: {output} ({size_mb:.1f} MB)")
    else:
        print(f"\nBuild completed but output not found at expected path: {output}")

    return output if output.exists() else None


def deploy(exe_path: Path):
    """Copy the built binary to %APPDATA%\\as-help-mcp."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        print("Error: APPDATA environment variable not set.")
        sys.exit(1)

    dest_dir = Path(appdata) / "as-help-mcp"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / exe_path.name
    print(f"\nDeploying {exe_path.name} -> {dest}")
    shutil.copy2(exe_path, dest)
    print(f"Deployed ({dest.stat().st_size / (1024 * 1024):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build as-help-server binary")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts before building")
    parser.add_argument("--deploy", action="store_true", help="Copy binary to %%APPDATA%%\\as-help-mcp after build")
    args = parser.parse_args()

    if args.clean:
        clean_build_artifacts()

    exe_path = build()

    if args.deploy:
        if exe_path is None:
            print("Error: No binary found to deploy.")
            sys.exit(1)
        deploy(exe_path)


if __name__ == "__main__":
    main()
