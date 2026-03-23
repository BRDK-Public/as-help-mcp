"""Test incremental reindex by tampering with stored metadata.

Run with: uv run python test_incremental.py

This script:
1. Loads the existing _index_metadata.json
2. Fakes the xml_hash to trigger change detection
3. Tampers with a few fingerprints to simulate added/removed/changed pages
4. Runs the search engine init to observe incremental behavior
5. Verifies the index still works with a search query
"""

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HELP_ROOT = Path(r"C:\Program Files (x86)\BRAutomation\AS6\Help-en\Data")
DB_PATH = Path(r"..\data\as6\.ashelp_lance").resolve()
METADATA_DIR = Path(r"..\data\as6\.ashelp_metadata").resolve()
META_FILE = DB_PATH / "_index_metadata.json"


def main():
    # --- Step 1: Verify metadata exists ---
    if not META_FILE.exists():
        print("ERROR: _index_metadata.json not found. Run a full build first.")
        sys.exit(1)

    with open(META_FILE) as f:
        metadata = json.load(f)

    original_hash = metadata["xml_hash"]
    fps = metadata.get("page_fingerprints", {})
    print(f"Metadata loaded: {metadata['page_count']} pages, {len(fps)} fingerprints")
    print(f"Original XML hash: {original_hash}")

    # --- Step 2: Tamper metadata to simulate changes ---
    # 2a. Fake the XML hash (triggers "XML changed" detection)
    metadata["xml_hash"] = "fake_hash_to_trigger_incremental"

    # 2b. Pick 3 real page IDs to simulate changes
    page_ids = list(fps.keys())
    changed_id = page_ids[0]    # simulate a changed page
    removed_id = page_ids[1]    # simulate a removed page
    fake_added_id = "fake-0000-0000-0000-added-page"  # simulate an added page

    # Store originals for restoration
    original_changed_fp = fps[changed_id]
    original_removed_fp = fps[removed_id]

    # Modify fingerprint for "changed" page
    fps[changed_id] = "aaaa_fake_changed_fingerprint"
    # Remove fingerprint for "removed" page
    del fps[removed_id]
    # Note: we can't add a fake page id to fingerprints because
    # the indexer won't have it in memory — the incremental code
    # will detect it as "in new XML but not in old metadata" = added

    # Write tampered metadata
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nTampered metadata written:")
    print(f"  - Faked xml_hash")
    print(f"  - Changed fingerprint for page: {changed_id}")
    print(f"  - Removed fingerprint for page: {removed_id}")

    # --- Step 3: Initialize search engine and observe incremental logic ---
    print("\n=== Starting search engine (should detect incremental) ===\n")
    from src.indexer import HelpContentIndexer
    from src.search_engine import HelpSearchEngine

    indexer = HelpContentIndexer(HELP_ROOT, metadata_dir=METADATA_DIR)
    indexer.parse_xml_structure()

    engine = HelpSearchEngine(db_path=DB_PATH, indexer=indexer, force_rebuild=False)
    print(f"Build strategy: {engine._build_strategy}")
    assert engine._build_strategy == "incremental", f"Expected 'incremental', got '{engine._build_strategy}'"

    start = time.time()
    engine.initialize()
    elapsed = time.time() - start
    print(f"\nIncremental update completed in {elapsed:.1f}s")

    # --- Step 4: Verify search still works ---
    print("\n=== Testing search after incremental update ===")
    results = engine.search("mapp View changelog", limit=3)
    print(f"Search returned {len(results)} results:")
    for r in results:
        print(f"  - {r['title']} (score: {r['score']:.4f})")

    assert len(results) > 0, "Search returned no results after incremental update!"

    # --- Step 5: Verify new metadata was saved ---
    with open(META_FILE) as f:
        new_metadata = json.load(f)

    real_hash = indexer._get_xml_hash()
    assert new_metadata["xml_hash"] == real_hash, "Metadata xml_hash not updated after incremental!"
    print(f"\nMetadata updated correctly: xml_hash = {real_hash}")

    new_fps = new_metadata.get("page_fingerprints", {})
    assert removed_id in new_fps, f"Removed page {removed_id} should be back in fingerprints"
    assert new_fps[changed_id] != "aaaa_fake_changed_fingerprint", "Changed page fingerprint not restored"
    print(f"Fingerprints restored: {len(new_fps)} entries")

    engine.close()
    print("\n=== ALL INCREMENTAL TESTS PASSED ===")


if __name__ == "__main__":
    main()
