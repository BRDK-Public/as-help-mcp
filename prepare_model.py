#!/usr/bin/env python3
"""Export the sentence-transformer model to a local directory for Docker builds.

The Docker builder cannot download from HuggingFace Hub in corporate networks
(corporate proxy TLS inspection causes SSL verification failures). This script
exports the model on the host machine (where the corporate CA is trusted) to a
local directory that the Dockerfile COPYs into the image.

Usage:
    uv run python prepare_model.py

The exported model is saved to .model_cache/all-MiniLM-L6-v2/ (~22MB).
This directory is git-ignored but included in the Docker build context.
"""

import os
import sys
from pathlib import Path

MODEL_NAME = os.getenv("AS_HELP_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CACHE_DIR = Path(".model_cache") / MODEL_NAME


def main():
    if CACHE_DIR.exists() and any(CACHE_DIR.iterdir()):
        print(f"Model already exported at {CACHE_DIR}")
        print("Delete the directory to re-export: rm -rf .model_cache/")
        return

    print(f"Downloading and exporting model '{MODEL_NAME}'...")
    print("(This uses the host's network and certificate store)")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(CACHE_DIR))

    # Calculate size
    total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
    print(f"Model exported to {CACHE_DIR} ({total / 1024 / 1024:.1f} MB)")
    print("You can now run: docker compose build as-help-local")


if __name__ == "__main__":
    main()
