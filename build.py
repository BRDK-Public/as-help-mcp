"""Build script for creating a standalone binary with PyInstaller.

Usage:
    uv run python build.py          # Build using spec file
    uv run python build.py --clean  # Clean build artifacts first
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Build as-help-server binary")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts before building")
    args = parser.parse_args()

    if args.clean:
        clean_build_artifacts()

    build()


if __name__ == "__main__":
    main()
