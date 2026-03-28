"""LanceDB search engine with FTS (default) and optional hybrid RRF search.

By default, uses LanceDB for full-text keyword search only (no embeddings).
When an embedding API is configured (CREATE_EMBEDDINGS=true), adds vector
columns and enables hybrid search with Reciprocal Rank Fusion (RRF).
"""

import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import pyarrow as pa

if TYPE_CHECKING:
    from src.embeddings import EmbeddingService

from src.indexer import HelpContentIndexer

logger = logging.getLogger(__name__)

# Standard RRF constant (from the original RRF paper)
RRF_K = 60

# RRF weights for each search signal (natural-language defaults)
WEIGHT_TITLE_VECTOR = 2.0
WEIGHT_CONTENT_VECTOR = 1.0
WEIGHT_FTS_KEYWORD = 1.5
WEIGHT_TITLE_MATCH = 3.0  # exact / substring title match bonus
WEIGHT_BREADCRUMB_MATCH = 2.0  # query terms found in breadcrumb path

# Alternate weights when query looks like a technical identifier
WEIGHT_TITLE_VECTOR_ID = 0.5
WEIGHT_CONTENT_VECTOR_ID = 0.5
WEIGHT_FTS_KEYWORD_ID = 3.0
WEIGHT_TITLE_MATCH_ID = 4.0
WEIGHT_BREADCRUMB_MATCH_ID = 3.0

# Number of pages per chunk during index build (saves progress after each chunk)
BUILD_CHUNK_SIZE = 5000

# Pattern for technical identifiers: PascalCase, snake_case, UPPER_CASE, dotted names
# e.g. MC_MoveAbsolute, AsGuard, SYS_Lib, X20DI9371, mapp.Motion
_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*$"  # single token with optional dots/underscores
)


def _is_identifier_query(query: str) -> bool:
    """Detect if a query looks like a technical identifier rather than natural language.

    Heuristic: the query is 1-2 words and each word matches the identifier pattern
    (PascalCase, snake_case, dotted name, or product code like X20DI9371).
    """
    words = query.strip().split()
    if not words or len(words) > 2:
        return False
    return all(_IDENTIFIER_RE.match(w) for w in words)


class HelpSearchEngine:
    """Search engine using LanceDB with FTS and optional vector hybrid search.

    When `embedding_service` is None (default), only FTS keyword search is
    available.  When an `EmbeddingService` is provided, the engine creates
    vector columns and uses RRF fusion of title vectors, content vectors, and
    FTS keywords.
    """

    TABLE_NAME = "help_pages"

    # Class-level tracking of active db_paths in this process
    _active_db_paths: set[str] = set()
    _active_db_paths_lock = threading.Lock()

    def __init__(
        self,
        db_path: Path,
        indexer: HelpContentIndexer,
        force_rebuild: bool = False,
        embedding_service: "EmbeddingService | None" = None,
    ):
        self.db_path = Path(db_path)
        self.indexer = indexer
        self.embedder = embedding_service  # None = FTS-only mode
        self._embeddings_enabled = embedding_service is not None
        self._ready = threading.Event()
        self._fts_ready = threading.Event()
        self._build_error: Exception | None = None

        # Build status tracking
        self._build_status: dict = {
            "state": "initializing",
            "build_type": None,
            "phase": "",
            "pages_total": 0,
            "pages_processed": 0,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "incremental_stats": None,
            "embeddings_enabled": self._embeddings_enabled,
        }

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._metadata_path = self.db_path / "_index_metadata.json"
        self._build_progress_path = self.db_path / "_build_progress.json"
        self._build_lock_path = self.db_path / "_build.lock"
        self._build_lock_owned = False
        self._instance_lock_path = self.db_path / "_instance.lock"
        self._instance_lock_owned = False
        self._acquire_instance_lock()

        if force_rebuild:
            self._build_strategy = "full"
        elif self._has_resumable_build():
            self._build_strategy = "resume"
        elif not self._index_exists():
            self._build_strategy = "full"
        else:
            self._build_strategy = self._detect_build_strategy()

        self._build_status["build_type"] = self._build_strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self):
        """Build or load the search index.

        When embeddings are disabled, builds FTS-only index (fast).
        When embeddings are enabled, builds in two phases:
        - Phase 1: Extract text -> write to LanceDB -> build FTS (keyword search available)
        - Phase 2: Embed via API -> overwrite with vectors -> rebuild FTS (hybrid search)

        Returns self for convenience chaining.
        """
        try:
            self._build_status["started_at"] = time.time()
            self._build_status["state"] = "building"
            self._build_status["pages_total"] = len(self.indexer.pages)

            if self._build_strategy in ("full", "incremental", "resume"):
                self._acquire_build_lock()

            if self._build_strategy == "full":
                logger.info("Building new search index (full)...")
                sys.stderr.flush()
                if self._embeddings_enabled:
                    self._build_index_two_phase()
                else:
                    self._build_fts_index()
            elif self._build_strategy == "resume":
                resume_ids = self._get_indexed_page_ids()
                remaining = len(self.indexer.pages) - len(resume_ids)
                logger.info(
                    f"Resuming interrupted build: {len(resume_ids)} pages already indexed, "
                    f"{remaining} remaining..."
                )
                sys.stderr.flush()
                if self._embeddings_enabled:
                    self._build_index_two_phase(resume_ids=resume_ids)
                else:
                    self._build_fts_index(resume_ids=resume_ids)
            elif self._build_strategy == "incremental":
                self._fts_ready.set()
                self._build_status["state"] = "fts_ready"
                logger.info("Performing incremental index update (keyword search available)...")
                sys.stderr.flush()
                self._incremental_update()
            else:
                self._build_status["phase"] = "loading existing index"
                logger.info("Loading existing search index...")
                sys.stderr.flush()
                self._load_index()
                self._fts_ready.set()

            self._build_status["state"] = "ready"
            self._build_status["phase"] = "complete"
            self._build_status["completed_at"] = time.time()
        except Exception as e:
            self._build_error = e
            self._build_status["state"] = "error"
            self._build_status["error"] = str(e)
            logger.error(f"Search index initialization failed: {e}")
            raise
        finally:
            self._release_build_lock()
            self._ready.set()
        return self

    @property
    def ready(self) -> bool:
        """Whether the search index is fully ready."""
        return self._ready.is_set() and self._build_status.get("state") == "ready"

    @property
    def fts_ready(self) -> bool:
        """Whether keyword (FTS) search is available."""
        return self._fts_ready.is_set() and self._build_status.get("state") in ("fts_ready", "ready")

    @property
    def build_status(self) -> dict:
        """Current build status snapshot."""
        status = dict(self._build_status)
        if status["started_at"] and not status["completed_at"]:
            status["elapsed_seconds"] = round(time.time() - status["started_at"], 1)
        elif status["started_at"] and status["completed_at"]:
            status["elapsed_seconds"] = round(status["completed_at"] - status["started_at"], 1)
        else:
            status["elapsed_seconds"] = None
        return status

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the index is ready. Returns True if ready, False on timeout."""
        return self._ready.wait(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Index existence and change detection
    # ------------------------------------------------------------------

    def _index_exists(self) -> bool:
        try:
            if self.TABLE_NAME not in self.db.list_tables().tables:
                return False
            table = self.db.open_table(self.TABLE_NAME)
            if table.count_rows() == 0:
                logger.info("LanceDB table exists but is empty - will rebuild")
                return False
            return True
        except Exception:
            return False

    def _detect_build_strategy(self) -> str:
        """Determine build strategy: full, incremental, or none."""
        if not self._metadata_path.exists():
            return "full"
        try:
            with open(self._metadata_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            return "full"

        # Embedding mode changed -> full rebuild
        stored_embeddings = metadata.get("embeddings_enabled", False)
        if stored_embeddings != self._embeddings_enabled:
            logger.info(
                "Embedding mode changed (%s -> %s) - full rebuild required",
                stored_embeddings, self._embeddings_enabled,
            )
            return "full"

        # If embeddings enabled, check model name
        if self._embeddings_enabled and self.embedder is not None:
            if metadata.get("embedding_model") != self.embedder.model_name:
                logger.info("Embedding model changed - full rebuild required")
                return "full"

        # XML unchanged -> nothing to do
        if metadata.get("xml_hash") == self.indexer._get_xml_hash():
            return "none"

        # XML changed - can we do incremental?
        if metadata.get("page_fingerprints"):
            logger.info("XML changed - incremental update possible")
            return "incremental"

        logger.info("XML changed but no page fingerprints stored - full rebuild required")
        return "full"

    def _save_metadata(self):
        """Save index metadata including per-page fingerprints."""
        metadata: dict = {
            "xml_hash": self.indexer._get_xml_hash(),
            "indexed_at": time.time(),
            "page_count": len(self.indexer.pages),
            "help_id_count": len(self.indexer.help_id_map),
            "embeddings_enabled": self._embeddings_enabled,
            "page_fingerprints": self.indexer.get_page_fingerprints(),
        }
        if self._embeddings_enabled and self.embedder is not None:
            metadata["embedding_model"] = self.embedder.model_name
            metadata["embedding_dimension"] = self.embedder.dimension
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Text extraction helper
    # ------------------------------------------------------------------

    def _extract_text_for_page(self, page_id: str, page) -> tuple:
        """Extract text for a single page (used for parallel processing).

        Both sections and pages get their HTML content extracted when available.
        Many B&R sections contain substantive documentation (e.g., LED tables,
        wiring diagrams) that is valuable for search.
        """
        plain_text = self.indexer._extract_plain_text_no_cache(page) or ""

        breadcrumb_path = self.indexer.get_breadcrumb_string(page_id)

        category = ""
        breadcrumb = self.indexer.get_breadcrumb(page_id)
        if breadcrumb and len(breadcrumb) > 0:
            category = breadcrumb[0].text

        return (
            page_id,
            page.text,       # title
            plain_text,       # content
            page.file_path,
            page.help_id or "",
            1 if page.is_section else 0,
            breadcrumb_path,
            category,
        )

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _get_fts_schema(self) -> pa.Schema:
        """PyArrow schema for FTS-only mode (no vector columns)."""
        return pa.schema([
            pa.field("page_id", pa.utf8()),
            pa.field("title", pa.utf8()),
            pa.field("content", pa.utf8()),
            pa.field("search_text", pa.utf8()),
            pa.field("file_path", pa.utf8()),
            pa.field("help_id", pa.utf8()),
            pa.field("is_section", pa.int32()),
            pa.field("breadcrumb_path", pa.utf8()),
            pa.field("category", pa.utf8()),
        ])

    def _get_hybrid_schema(self) -> pa.Schema:
        """PyArrow schema with vector columns for hybrid search."""
        dim = self.embedder.dimension  # type: ignore[union-attr]
        return pa.schema([
            pa.field("page_id", pa.utf8()),
            pa.field("title", pa.utf8()),
            pa.field("content", pa.utf8()),
            pa.field("search_text", pa.utf8()),
            pa.field("file_path", pa.utf8()),
            pa.field("help_id", pa.utf8()),
            pa.field("is_section", pa.int32()),
            pa.field("breadcrumb_path", pa.utf8()),
            pa.field("category", pa.utf8()),
            pa.field("title_vector", pa.list_(pa.float32(), dim)),
            pa.field("content_vector", pa.list_(pa.float32(), dim)),
        ])

    def _get_table_schema(self) -> pa.Schema:
        """Return the appropriate schema based on embedding mode."""
        if self._embeddings_enabled:
            return self._get_hybrid_schema()
        return self._get_fts_schema()

    def _records_to_fts_arrow(self, records) -> pa.Table:
        """Convert records to a FTS-only PyArrow table (no vectors)."""
        search_texts = [f"{r[1]} {r[6]} {r[2]}" for r in records]
        data = {
            "page_id": [r[0] for r in records],
            "title": [r[1] for r in records],
            "content": [r[2] for r in records],
            "search_text": search_texts,
            "file_path": [r[3] for r in records],
            "help_id": [r[4] for r in records],
            "is_section": [r[5] for r in records],
            "breadcrumb_path": [r[6] for r in records],
            "category": [r[7] for r in records],
        }
        return pa.table(data, schema=self._get_fts_schema())

    def _records_to_hybrid_arrow(self, records, title_vectors, content_vectors) -> pa.Table:
        """Convert records + embeddings to a hybrid PyArrow table."""
        search_texts = [f"{r[1]} {r[6]} {r[2]}" for r in records]
        data = {
            "page_id": [r[0] for r in records],
            "title": [r[1] for r in records],
            "content": [r[2] for r in records],
            "search_text": search_texts,
            "file_path": [r[3] for r in records],
            "help_id": [r[4] for r in records],
            "is_section": [r[5] for r in records],
            "breadcrumb_path": [r[6] for r in records],
            "category": [r[7] for r in records],
            "title_vector": title_vectors,
            "content_vector": content_vectors,
        }
        return pa.table(data, schema=self._get_hybrid_schema())

    def _build_content_vectors(self, records, title_vectors) -> list[list[float]]:
        """Build content vectors; reuse title vectors for pages without content.

        Prepends the breadcrumb path to the content text before embedding to
        give the vector context about which product/section the page belongs to.
        """
        content_indices: list[int] = []
        content_texts: list[str] = []

        for idx, record in enumerate(records):
            content = record[2]
            if content:
                content_indices.append(idx)
                # Prepend breadcrumb for product/section context
                breadcrumb = record[6]
                embed_text = f"{breadcrumb} | {content}" if breadcrumb else content
                content_texts.append(embed_text)

        content_vectors = list(title_vectors)

        if content_texts:
            embedded_contents = self.embedder.embed_batch(content_texts, show_progress=False)  # type: ignore[union-attr]
            for idx, vec in zip(content_indices, embedded_contents, strict=True):
                content_vectors[idx] = vec

        return content_vectors

    # ------------------------------------------------------------------
    # FTS-only build (default, no embeddings)
    # ------------------------------------------------------------------

    def _build_fts_index(self, resume_ids: set[str] | None = None):
        """Build FTS-only index: extract text -> write to LanceDB -> create FTS index."""
        start_time = time.time()
        all_pages = list(self.indexer.pages.items())
        total_pages = len(all_pages)

        if resume_ids:
            pages_to_process = [(pid, page) for pid, page in all_pages if pid not in resume_ids]
            already_done = total_pages - len(pages_to_process)
            table_created = True
        else:
            pages_to_process = all_pages
            already_done = 0
            table_created = False
            try:
                if self.TABLE_NAME in self.db.list_tables().tables:
                    self.db.drop_table(self.TABLE_NAME)
            except Exception:
                pass

        self._build_status["pages_total"] = total_pages
        self._build_status["pages_processed"] = already_done

        if not pages_to_process:
            logger.info("All pages already indexed, finalizing...")
            self._finalize_fts_build(start_time, total_pages)
            self._fts_ready.set()
            return

        self._save_build_progress()

        max_workers = min(int(os.cpu_count() or 4), 10)
        remaining = len(pages_to_process)
        total_chunks = (remaining + BUILD_CHUNK_SIZE - 1) // BUILD_CHUNK_SIZE

        if already_done > 0:
            logger.info(
                f"Skipped {already_done} already-indexed pages. "
                f"Extracting text for {remaining} remaining pages ({total_chunks} chunks, {max_workers} workers)..."
            )
        else:
            logger.info(f"Extracting text for {remaining} pages ({total_chunks} chunks, {max_workers} workers)...")
        sys.stderr.flush()

        for chunk_start in range(0, remaining, BUILD_CHUNK_SIZE):
            chunk = pages_to_process[chunk_start : chunk_start + BUILD_CHUNK_SIZE]
            chunk_num = chunk_start // BUILD_CHUNK_SIZE + 1

            self._build_status["phase"] = f"extracting text (chunk {chunk_num}/{total_chunks})"
            records = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                extraction_results = executor.map(
                    lambda item: self._extract_text_for_page(item[0], item[1]),
                    chunk,
                    chunksize=min(100, max(1, max_workers * 2)),
                )
                for result in extraction_results:
                    records.append(result)

            chunk_data = self._records_to_fts_arrow(records)

            self._build_status["phase"] = f"saving (chunk {chunk_num}/{total_chunks})"
            if not table_created:
                self.db.create_table(self.TABLE_NAME, chunk_data)
                table_created = True
            else:
                table = self.db.open_table(self.TABLE_NAME)
                table.add(chunk_data)

            processed = already_done + chunk_start + len(chunk)
            self._build_status["pages_processed"] = processed
            pct = processed * 100 // total_pages
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            logger.info(f"Phase 1: {pct}% ({processed}/{total_pages} pages, {rate:.0f} pages/s)")
            sys.stderr.flush()

        self._finalize_fts_build(start_time, total_pages)
        self._fts_ready.set()

    def _finalize_fts_build(self, start_time: float, total_pages: int):
        """Create FTS index and save metadata after text extraction."""
        table = self.db.open_table(self.TABLE_NAME)
        self._build_status["phase"] = "creating FTS index"
        logger.info("Creating FTS index...")
        table.create_fts_index("search_text", replace=True)

        self._save_metadata()
        self._clear_build_progress()

        self._build_status["pages_processed"] = total_pages
        elapsed = time.time() - start_time
        logger.info(f"FTS search index built in {elapsed:.1f}s ({total_pages} documents)")

    # ------------------------------------------------------------------
    # Two-phase build (embeddings enabled)
    # ------------------------------------------------------------------

    STAGING_TABLE = "help_pages_staging"

    def _build_index_two_phase(self, resume_ids: set[str] | None = None):
        """Build index with embeddings: Phase 1 = FTS, Phase 2 = vectors.

        Memory-efficient: each phase writes in chunks so only ~5000 pages
        are in memory at a time.  Phase 2 writes to a staging table while
        keyword search stays available on the original table, then does a
        brief swap at the end.
        """
        start_time = time.time()
        all_pages = list(self.indexer.pages.items())
        total_pages = len(all_pages)

        if resume_ids:
            pages_to_process = [(pid, page) for pid, page in all_pages if pid not in resume_ids]
            already_done = total_pages - len(pages_to_process)
            table_created = True
        else:
            pages_to_process = all_pages
            already_done = 0
            table_created = False
            try:
                if self.TABLE_NAME in self.db.list_tables().tables:
                    self.db.drop_table(self.TABLE_NAME)
            except Exception:
                pass

        self._build_status["pages_total"] = total_pages
        self._build_status["pages_processed"] = already_done

        if not pages_to_process:
            if not resume_ids:
                # No pages at all (edge case) — finalize and return
                logger.info("No pages to index, finalizing...")
                self._finalize_fts_build(start_time, total_pages)
                self._fts_ready.set()
                self._build_status["state"] = "fts_ready"
                return
            # Resume with Phase 1 complete: build FTS, then fall through to Phase 2
            logger.info(
                f"Phase 1 already complete ({already_done} pages resumed). "
                "Building FTS index, then proceeding to Phase 2 for embeddings..."
            )

        self._save_build_progress()

        max_workers = min(int(os.cpu_count() or 4), 10)
        remaining = len(pages_to_process)
        total_chunks = (remaining + BUILD_CHUNK_SIZE - 1) // BUILD_CHUNK_SIZE
        dim = self.embedder.dimension  # type: ignore[union-attr]

        # -- Phase 1: Extract text, write with zero vectors, build FTS --
        if already_done > 0:
            logger.info(
                f"Phase 1: Skipped {already_done} already-indexed pages. "
                f"Extracting text for {remaining} remaining pages ({total_chunks} chunks)..."
            )
        else:
            logger.info(f"Phase 1: Extracting text for {remaining} pages ({total_chunks} chunks)...")
        sys.stderr.flush()

        for chunk_start in range(0, remaining, BUILD_CHUNK_SIZE):
            chunk = pages_to_process[chunk_start : chunk_start + BUILD_CHUNK_SIZE]
            chunk_num = chunk_start // BUILD_CHUNK_SIZE + 1

            self._build_status["phase"] = f"extracting text (chunk {chunk_num}/{total_chunks})"
            records = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                extraction_results = executor.map(
                    lambda item: self._extract_text_for_page(item[0], item[1]),
                    chunk,
                    chunksize=min(100, max(1, max_workers * 2)),
                )
                for result in extraction_results:
                    records.append(result)

            zero_title_vectors = [[0.0] * dim for _ in records]
            zero_content_vectors = [[0.0] * dim for _ in records]
            chunk_data = self._records_to_hybrid_arrow(records, zero_title_vectors, zero_content_vectors)

            self._build_status["phase"] = f"saving text (chunk {chunk_num}/{total_chunks})"
            if not table_created:
                self.db.create_table(self.TABLE_NAME, chunk_data)
                table_created = True
            else:
                table = self.db.open_table(self.TABLE_NAME)
                table.add(chunk_data)

            processed = already_done + chunk_start + len(chunk)
            self._build_status["pages_processed"] = processed
            pct = processed * 100 // total_pages
            elapsed_p1 = time.time() - start_time
            rate = processed / elapsed_p1 if elapsed_p1 > 0 else 0
            logger.info(f"Phase 1: {pct}% ({processed}/{total_pages} pages, {rate:.0f} pages/s)")
            sys.stderr.flush()

        # Build FTS index -> keyword search available
        self._build_status["phase"] = "creating FTS index (keyword search)"
        logger.info("Creating FTS index...")
        table = self.db.open_table(self.TABLE_NAME)
        table.create_fts_index("search_text", replace=True)

        self._fts_ready.set()
        self._build_status["state"] = "fts_ready"
        phase1_elapsed = time.time() - start_time
        logger.info(f"Phase 1 complete in {phase1_elapsed:.1f}s - keyword search is now available")
        sys.stderr.flush()

        # -- Phase 2: Chunked embed + write to staging table --
        # Keyword search stays available on the original table while we embed.
        # All pages are re-extracted from HTML (fast) so we don't need to keep
        # Phase 1 records in memory.
        self._cleanup_staging_table()

        embed_total = total_pages
        embed_chunks = (embed_total + BUILD_CHUNK_SIZE - 1) // BUILD_CHUNK_SIZE
        self._build_status["phase"] = "embedding via API"
        logger.info(f"Phase 2: Embedding {embed_total} pages via API (chunked, low memory)...")
        sys.stderr.flush()

        staging_created = False
        phase2_start = time.time()

        for chunk_start in range(0, embed_total, BUILD_CHUNK_SIZE):
            chunk = all_pages[chunk_start : chunk_start + BUILD_CHUNK_SIZE]
            chunk_num = chunk_start // BUILD_CHUNK_SIZE + 1

            # Re-extract text from HTML (parallel, fast)
            self._build_status["phase"] = f"extracting + embedding (chunk {chunk_num}/{embed_chunks})"
            records = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                extraction_results = executor.map(
                    lambda item: self._extract_text_for_page(item[0], item[1]),
                    chunk,
                    chunksize=min(100, max(1, max_workers * 2)),
                )
                for result in extraction_results:
                    records.append(result)

            # Embed titles with breadcrumb context
            titles = [f"{r[1]} | {r[6]}" if r[6] else r[1] for r in records]
            title_vectors = self.embedder.embed_batch(titles, show_progress=False)  # type: ignore[union-attr]
            content_vectors = self._build_content_vectors(records, title_vectors)

            # Write chunk to staging table
            chunk_data = self._records_to_hybrid_arrow(records, title_vectors, content_vectors)
            if not staging_created:
                self.db.create_table(self.STAGING_TABLE, chunk_data)
                staging_created = True
            else:
                self.db.open_table(self.STAGING_TABLE).add(chunk_data)

            # Free memory before next chunk
            del records, title_vectors, content_vectors, chunk_data

            embedded_so_far = min(chunk_start + len(chunk), embed_total)
            self._build_status["pages_processed"] = embedded_so_far
            pct = embedded_so_far * 100 // embed_total
            elapsed_p2 = time.time() - phase2_start
            rate = embedded_so_far / elapsed_p2 if elapsed_p2 > 0 else 0
            eta = (embed_total - embedded_so_far) / rate if rate > 0 else 0
            logger.info(f"Phase 2: {pct}% ({embedded_so_far}/{embed_total} pages, {rate:.0f} pages/s, ETA {eta:.0f}s)")
            sys.stderr.flush()

        # Swap staging → final (brief FTS suspension)
        self._swap_staging_table()

        self._save_metadata()
        self._clear_build_progress()

        self._build_status["pages_processed"] = total_pages
        elapsed = time.time() - start_time
        logger.info(f"Phase 2 complete - full hybrid search ready in {elapsed:.1f}s ({total_pages} documents)")

    def _cleanup_staging_table(self):
        """Remove any leftover staging table from a previous interrupted build."""
        try:
            tables = self.db.list_tables().tables if hasattr(self.db.list_tables(), 'tables') else self.db.list_tables()
            if self.STAGING_TABLE in tables:
                self.db.drop_table(self.STAGING_TABLE)
        except Exception:
            pass

    def _swap_staging_table(self):
        """Replace the main table with the staging table.

        Uses filesystem rename for an atomic swap, keeping FTS downtime
        to a minimum.  Falls back to an in-memory copy if rename fails.
        """
        self._fts_ready.clear()
        self._build_status["state"] = "building"
        self._build_status["phase"] = "swapping staging table"
        logger.info("Swapping staging table → final (keyword search briefly unavailable)...")

        # Drop the original (Phase 1) table
        try:
            self.db.drop_table(self.TABLE_NAME)
        except Exception:
            pass

        # Try filesystem rename (zero-copy, no memory spike)
        swapped = False
        for suffix in [".lance", ""]:
            staging_dir = self.db_path / f"{self.STAGING_TABLE}{suffix}"
            final_dir = self.db_path / f"{self.TABLE_NAME}{suffix}"
            if staging_dir.is_dir():
                try:
                    staging_dir.rename(final_dir)
                    # Reconnect so LanceDB sees the renamed directory
                    self.db = lancedb.connect(str(self.db_path))
                    swapped = True
                    logger.info("Staging table swapped via filesystem rename")
                    break
                except OSError as e:
                    logger.warning(f"Filesystem rename failed ({e}), falling back to copy")

        # Fallback: read staging data through Arrow and recreate
        if not swapped:
            staging = self.db.open_table(self.STAGING_TABLE)
            arrow_data = staging.to_arrow()
            self.db.create_table(self.TABLE_NAME, arrow_data)
            del arrow_data
            self.db.drop_table(self.STAGING_TABLE)
            logger.info("Staging table swapped via Arrow copy")

        # Rebuild FTS on the final table
        self._build_status["phase"] = "rebuilding FTS index"
        logger.info("Rebuilding FTS index...")
        table = self.db.open_table(self.TABLE_NAME)
        table.create_fts_index("search_text", replace=True)
        self._fts_ready.set()
        self._build_status["state"] = "fts_ready"

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def _incremental_update(self):
        """Incrementally update the index by diffing page fingerprints."""
        start_time = time.time()

        with open(self._metadata_path) as f:
            metadata = json.load(f)
        old_fps: dict[str, str] = metadata.get("page_fingerprints", {})
        new_fps = self.indexer.get_page_fingerprints()

        old_ids = set(old_fps.keys())
        new_ids = set(new_fps.keys())

        added = new_ids - old_ids
        removed = old_ids - new_ids
        changed = {pid for pid in (old_ids & new_ids) if old_fps[pid] != new_fps[pid]}

        to_upsert = added | changed
        to_delete = removed | changed

        self._build_status["incremental_stats"] = {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "unchanged": len(new_ids) - len(to_upsert),
        }
        self._build_status["pages_total"] = len(to_upsert)
        self._build_status["phase"] = "computing diff"

        logger.info(
            f"Incremental diff: {len(added)} added, {len(removed)} removed, "
            f"{len(changed)} changed, {len(new_ids) - len(to_upsert)} unchanged"
        )

        if not to_upsert and not to_delete:
            logger.info("No page-level changes detected - skipping update")
            self._save_metadata()
            return

        # Fall back to full rebuild if >50% changed
        if len(to_upsert) > len(new_ids) * 0.5:
            logger.info(f"Too many changes ({len(to_upsert)}/{len(new_ids)}) - falling back to full rebuild")
            # Suspend search availability during full rebuild
            self._fts_ready.clear()
            self._build_status["state"] = "building"
            if self._embeddings_enabled:
                self._build_index_two_phase()
            else:
                self._build_fts_index()
            return

        table = self.db.open_table(self.TABLE_NAME)

        # Delete removed and changed rows
        if to_delete:
            self._build_status["phase"] = "deleting old rows"
            delete_list = list(to_delete)
            batch_size = 500
            for i in range(0, len(delete_list), batch_size):
                batch = delete_list[i : i + batch_size]
                id_literals = ", ".join(f"'{pid.replace(chr(39), chr(39)+chr(39))}'" for pid in batch)
                table.delete(f"page_id IN ({id_literals})")
            logger.info(f"Deleted {len(to_delete)} rows from index")

        # Extract, embed (if enabled), and insert new/changed pages
        if to_upsert:
            self._build_status["phase"] = "extracting text"
            pages_to_index = [(pid, self.indexer.pages[pid]) for pid in to_upsert]
            max_workers = int(os.cpu_count() or 4)

            records = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                extraction_results = executor.map(
                    lambda item: self._extract_text_for_page(item[0], item[1]),
                    pages_to_index,
                    chunksize=min(100, max(1, max_workers * 2)),
                )
                for result in extraction_results:
                    records.append(result)

            logger.info(f"Extracted text for {len(records)} pages")
            self._build_status["pages_processed"] = len(records)

            if self._embeddings_enabled:
                titles = [r[1] for r in records]
                self._build_status["phase"] = "embedding titles"
                title_vectors = self.embedder.embed_batch(titles, show_progress=False)  # type: ignore[union-attr]
                self._build_status["phase"] = "embedding content"
                content_vectors = self._build_content_vectors(records, title_vectors)
                new_data = self._records_to_hybrid_arrow(records, title_vectors, content_vectors)
            else:
                new_data = self._records_to_fts_arrow(records)

            table.add(new_data)
            logger.info(f"Added {len(records)} rows to index")

        # Rebuild FTS index
        self._build_status["phase"] = "rebuilding FTS index"
        logger.info("Rebuilding FTS index...")
        table.create_fts_index("search_text", replace=True)

        self._save_metadata()

        elapsed = time.time() - start_time
        logger.info(
            f"Incremental update complete in {elapsed:.1f}s (+{len(added)} -{len(removed)} ~{len(changed)} pages)"
        )

    def _load_index(self):
        """Load existing search index and log stats."""
        table = self.db.open_table(self.TABLE_NAME)
        doc_count = table.count_rows()
        mode = "hybrid" if self._embeddings_enabled else "FTS-only"
        logger.info(f"Loaded search index with {doc_count} documents ({mode})")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _build_category_filter(category: str | None) -> str | None:
        """Build a safe SQL where clause for category filtering."""
        if not category:
            return None
        safe = re.sub(r"[^\w\s.-]", "", category)
        return f"lower(category) = '{safe.lower()}'"

    def _vector_search(
        self, table, query_vector: list[float], column_name: str, limit: int, where_clause: str | None
    ) -> list[dict]:
        """Run vector similarity search on a specific column."""
        try:
            builder = table.search(query_vector, vector_column_name=column_name)
            if where_clause:
                builder = builder.where(where_clause)
            return builder.limit(limit).to_list()  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(f"Vector search on {column_name} failed: {e}")
            return []

    def _fts_search(self, table, query: str, limit: int, where_clause: str | None) -> list[dict]:
        """Run full-text keyword search with query sanitization."""
        sanitized = query
        for char in "\"'*:(){}^+[]-":
            sanitized = sanitized.replace(char, " ")

        fts_keywords = {"and", "or", "not", "near"}
        terms = [t.strip() for t in sanitized.split() if len(t.strip()) >= 2 and t.strip().lower() not in fts_keywords]

        if not terms:
            return []

        fts_query = " ".join(terms)
        try:
            builder = table.search(fts_query, query_type="fts")
            if where_clause:
                builder = builder.where(where_clause)
            return builder.limit(limit).to_list()  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(f"FTS search failed for '{fts_query}': {e}")
            return []

    @staticmethod
    def _generate_snippet(content: str, query: str) -> str | None:
        """Generate a text snippet around the first matching term."""
        if not content:
            return None

        sanitized = query
        for char in "\"'*:(){}^+[]-":
            sanitized = sanitized.replace(char, " ")
        terms = [t for t in sanitized.split() if len(t) >= 2]

        if terms:
            lower_content = content.lower()
            best_pos = len(content)
            for term in terms:
                pos = lower_content.find(term.lower())
                if 0 <= pos < best_pos:
                    best_pos = pos
            if best_pos < len(content):
                start = max(0, best_pos - 40)
                end = min(len(content), best_pos + 120)
                return ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")

        return content[:160] + ("..." if len(content) > 160 else "")

    @staticmethod
    def _apply_breadcrumb_bonus(
        query: str, page_data: dict[str, dict], rrf_scores: dict[str, float], weight: float
    ) -> None:
        """Add RRF bonus for pages whose breadcrumb path contains query terms.

        Pages are ranked by how many query terms appear in their breadcrumb,
        with more matches ranked higher.
        """
        sanitized = query
        for char in "\"'*:(){}^+[]-/":
            sanitized = sanitized.replace(char, " ")
        terms = [t.lower() for t in sanitized.split() if len(t) >= 2]
        if not terms:
            return

        # Score each page by number of query terms found in breadcrumb
        breadcrumb_hits: list[tuple[str, int]] = []
        for pid, data in page_data.items():
            bc = (data.get("breadcrumb_path") or "").lower()
            if not bc:
                continue
            hits = sum(1 for t in terms if t in bc)
            if hits > 0:
                breadcrumb_hits.append((pid, hits))

        # Sort by hit count descending (more matching terms = better rank)
        breadcrumb_hits.sort(key=lambda x: x[1], reverse=True)
        for rank, (pid, _hits) in enumerate(breadcrumb_hits):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + weight / (RRF_K + rank + 1)

    def _breadcrumb_retrieval(
        self, table, query: str, limit: int, where_clause: str | None
    ) -> list[dict]:
        """Retrieve pages whose breadcrumb contains ALL distinctive query terms.

        Uses a SQL scan with AND filter to find pages where the breadcrumb path
        contains all query terms (≥3 chars). Results are sorted in Python by
        the number of matching terms, so the most relevant pages come first.

        This is an independent retrieval leg that can surface pages missed by
        the main FTS search (which penalizes long documents via BM25 normalization).

        Requires at least 2 query terms to avoid overly broad single-term matches
        that would add noise to the RRF fusion.
        """
        sanitized = query
        for char in "\"'*:(){}^+[]-/":
            sanitized = sanitized.replace(char, " ")

        fts_keywords = {"and", "or", "not", "near"}
        terms = [t.strip().lower() for t in sanitized.split()
                 if len(t.strip()) >= 3 and t.strip().lower() not in fts_keywords]

        # Require at least 2 terms — single-term breadcrumb matches are too broad
        # (e.g. "ACP10" alone matches 200+ pages, just adding noise)
        if len(terms) < 2:
            return []

        # Escape SQL LIKE wildcards in query terms to prevent unexpected matches
        def _escape_like(term: str) -> str:
            return term.replace("%", "\\%").replace("_", "\\_")

        # Build WHERE clause: breadcrumb must contain ALL query terms (AND)
        bc_conditions = [f"lower(breadcrumb_path) LIKE '%{_escape_like(t)}%'" for t in terms]
        bc_filter = " AND ".join(bc_conditions)
        if where_clause:
            combined_filter = f"({bc_filter}) AND ({where_clause})"
        else:
            combined_filter = bc_filter

        try:
            # Use a generous scan limit since AND filter is narrow
            scan_limit = max(limit * 5, 200)
            raw_results = (
                table.search()
                .where(combined_filter)
                .limit(scan_limit)
                .to_list()
            )

            if not raw_results:
                return []

            # Sort by breadcrumb match quality: count matching terms, prefer shorter breadcrumbs
            def _bc_sort_key(row):
                bc = (row.get("breadcrumb_path") or "").lower()
                hits = sum(1 for t in terms if t in bc)
                return (-hits, len(bc))  # more hits first, shorter breadcrumbs first

            raw_results.sort(key=_bc_sort_key)
            return raw_results[:limit]
        except Exception as e:
            logger.warning(f"Breadcrumb retrieval failed: {e}")
            return []

    def search(
        self, query: str, limit: int = 20, search_in_content: bool = True, category: str | None = None
    ) -> list[dict]:
        """Search for help pages.

        When embeddings are enabled and ready, uses hybrid RRF search.
        Otherwise, uses FTS keyword search only.
        """
        if not query.strip():
            return []

        table = self.db.open_table(self.TABLE_NAME)
        where_clause = self._build_category_filter(category)

        # Use vectors only when embeddings are enabled AND index is fully ready
        use_vectors = self._embeddings_enabled and self.ready

        # Choose RRF weights based on query type
        is_identifier = _is_identifier_query(query)
        if is_identifier:
            w_title_vec = WEIGHT_TITLE_VECTOR_ID
            w_content_vec = WEIGHT_CONTENT_VECTOR_ID
            w_fts = WEIGHT_FTS_KEYWORD_ID
            w_title_match = WEIGHT_TITLE_MATCH_ID
            w_breadcrumb = WEIGHT_BREADCRUMB_MATCH_ID
        else:
            w_title_vec = WEIGHT_TITLE_VECTOR
            w_content_vec = WEIGHT_CONTENT_VECTOR
            w_fts = WEIGHT_FTS_KEYWORD
            w_title_match = WEIGHT_TITLE_MATCH
            w_breadcrumb = WEIGHT_BREADCRUMB_MATCH

        if use_vectors:
            try:
                query_vector = self.embedder.embed_text(query)  # type: ignore[union-attr]
            except Exception as e:
                logger.warning(f"Embedding API error, falling back to keyword search: {e}")
                use_vectors = False

        if use_vectors:
            fetch_limit = min(limit * 3, 100)

            title_results = self._vector_search(table, query_vector, "title_vector", fetch_limit, where_clause)
            content_results = []
            if search_in_content:
                content_results = self._vector_search(table, query_vector, "content_vector", fetch_limit, where_clause)
            fts_results = self._fts_search(table, query, fetch_limit, where_clause)

            # RRF Fusion
            rrf_scores: dict[str, float] = {}
            page_data: dict[str, dict] = {}

            for rank, row in enumerate(title_results):
                pid = row["page_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_title_vec / (RRF_K + rank + 1)
                if pid not in page_data:
                    page_data[pid] = row

            for rank, row in enumerate(content_results):
                pid = row["page_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_content_vec / (RRF_K + rank + 1)
                if pid not in page_data:
                    page_data[pid] = row

            for rank, row in enumerate(fts_results):
                pid = row["page_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_fts / (RRF_K + rank + 1)
                if pid not in page_data:
                    page_data[pid] = row

            # 4th leg: title exact / substring match bonus
            query_lower = query.strip().lower()
            title_match_candidates = sorted(
                [
                    (pid, data)
                    for pid, data in page_data.items()
                    if query_lower in data.get("title", "").lower()
                ],
                key=lambda x: (
                    x[1].get("title", "").lower() != query_lower,  # exact match first
                    len(x[1].get("title", "")),                    # shorter titles first
                ),
            )
            for rank, (pid, _data) in enumerate(title_match_candidates):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_title_match / (RRF_K + rank + 1)

            # 5th leg: breadcrumb retrieval — pulls pages by breadcrumb match
            #   (independent retrieval, can add NEW pages not found by other legs)
            bc_results = self._breadcrumb_retrieval(table, query, fetch_limit, where_clause)
            for rank, row in enumerate(bc_results):
                pid = row["page_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_breadcrumb / (RRF_K + rank + 1)
                if pid not in page_data:
                    page_data[pid] = row

            sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)[:limit]
            search_mode = "hybrid"
        else:
            # Over-fetch to allow reranking (same approach as hybrid mode)
            fetch_limit = min(limit * 3, 100)
            fts_results = self._fts_search(table, query, fetch_limit, where_clause)
            rrf_scores = {}
            page_data = {}

            for rank, row in enumerate(fts_results):
                pid = row["page_id"]
                rrf_scores[pid] = w_fts / (RRF_K + rank + 1)
                page_data[pid] = row

            # Title match bonus in keyword mode too
            query_lower = query.strip().lower()
            title_match_candidates = sorted(
                [
                    (pid, data)
                    for pid, data in page_data.items()
                    if query_lower in data.get("title", "").lower()
                ],
                key=lambda x: (
                    x[1].get("title", "").lower() != query_lower,
                    len(x[1].get("title", "")),
                ),
            )
            for rank, (pid, _data) in enumerate(title_match_candidates):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_title_match / (RRF_K + rank + 1)

            # Breadcrumb retrieval in keyword mode too
            bc_results = self._breadcrumb_retrieval(table, query, fetch_limit, where_clause)
            for rank, row in enumerate(bc_results):
                pid = row["page_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_breadcrumb / (RRF_K + rank + 1)
                if pid not in page_data:
                    page_data[pid] = row

            sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)[:limit]
            search_mode = "keyword"

        results = []
        for pid in sorted_ids:
            row = page_data[pid]
            snippet = self._generate_snippet(row.get("content", ""), query)
            results.append({
                "page_id": pid,
                "title": row.get("title", ""),
                "file_path": row.get("file_path", ""),
                "help_id": row.get("help_id") or None,
                "is_section": bool(row.get("is_section", 0)),
                "breadcrumb_path": row.get("breadcrumb_path") or None,
                "category": row.get("category") or None,
                "score": rrf_scores[pid],
                "snippet": snippet,
                "search_mode": search_mode,
            })

        logger.info(f"Search for '{query}' (cat={category}, mode={search_mode}) returned {len(results)} results")
        return results

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def _has_resumable_build(self) -> bool:
        if not self._build_progress_path.exists():
            return False
        try:
            if self.TABLE_NAME not in self.db.list_tables().tables:
                return False
            table = self.db.open_table(self.TABLE_NAME)
            if table.count_rows() == 0:
                return False
            with open(self._build_progress_path) as f:
                progress = json.load(f)
            # Match XML hash and embedding mode
            if progress.get("xml_hash") != self.indexer._get_xml_hash():
                return False
            if progress.get("embeddings_enabled", False) != self._embeddings_enabled:
                return False
            if self._embeddings_enabled and self.embedder is not None:
                if progress.get("embedding_model") != self.embedder.model_name:
                    return False
            return True
        except (json.JSONDecodeError, OSError, Exception):
            return False

    def _save_build_progress(self):
        progress: dict = {
            "xml_hash": self.indexer._get_xml_hash(),
            "embeddings_enabled": self._embeddings_enabled,
        }
        if self._embeddings_enabled and self.embedder is not None:
            progress["embedding_model"] = self.embedder.model_name
        with open(self._build_progress_path, "w") as f:
            json.dump(progress, f)

    def _clear_build_progress(self):
        if self._build_progress_path.exists():
            self._build_progress_path.unlink()

    def _get_indexed_page_ids(self) -> set[str]:
        try:
            table = self.db.open_table(self.TABLE_NAME)
            arrow_table = table.to_arrow().select(["page_id"])
            return set(arrow_table["page_id"].to_pylist())
        except Exception as e:
            logger.warning(f"Could not read existing page IDs: {e}")
            return set()

    # ------------------------------------------------------------------
    # Instance lock: one server per db_path
    # ------------------------------------------------------------------

    def _acquire_instance_lock(self):
        resolved = str(self.db_path.resolve())

        with self._active_db_paths_lock:
            if resolved in self._active_db_paths:
                raise RuntimeError(
                    f"Another HelpSearchEngine in this process is already using "
                    f"database at {self.db_path}. Close it first, "
                    f"or use a different --db-path for each help root."
                )

            lock_info = self._read_instance_lock()
            if lock_info is not None:
                other_pid = lock_info.get("pid")
                if other_pid and self._is_process_alive(other_pid) and other_pid != os.getpid():
                    raise RuntimeError(
                        f"Another as-help-server (PID {other_pid}) is already using "
                        f"database at {self.db_path}. Stop the other instance first, "
                        f"or use a different --db-path for each help root."
                    )
                if other_pid and other_pid != os.getpid():
                    logger.debug("Overwriting stale instance lock (PID %s no longer running)", other_pid)

            self._write_instance_lock()
            self._active_db_paths.add(resolved)

        self._instance_lock_owned = True
        logger.info("Acquired instance lock for %s (PID %s)", self.db_path, os.getpid())

    def _release_instance_lock(self):
        if not self._instance_lock_owned:
            return
        try:
            self._instance_lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            resolved = str(self.db_path.resolve())
            with self._active_db_paths_lock:
                self._active_db_paths.discard(resolved)
        except Exception:
            pass
        finally:
            self._instance_lock_owned = False

    def _read_instance_lock(self) -> dict | None:
        try:
            with open(self._instance_lock_path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _write_instance_lock(self):
        with open(self._instance_lock_path, "w") as f:
            json.dump({"pid": os.getpid(), "started_at": time.time()}, f)

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == STILL_ACTIVE
                return False
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    # ------------------------------------------------------------------
    # Build lock
    # ------------------------------------------------------------------

    def _acquire_build_lock(self, timeout_seconds: int = 1800):
        if self._build_lock_owned:
            return

        start = time.time()
        warned = False

        while True:
            try:
                fd = os.open(str(self._build_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    json.dump({"pid": os.getpid(), "started_at": time.time()}, f)
                self._build_lock_owned = True
                logger.info("Acquired build lock")
                return
            except FileExistsError:
                try:
                    lock_data = json.loads(self._build_lock_path.read_text())
                    lock_pid = lock_data.get("pid")
                except (OSError, json.JSONDecodeError, ValueError):
                    lock_pid = None

                if self._instance_lock_owned:
                    logger.warning("Removing stale build lock (this process holds instance lock)")
                    self._build_lock_path.unlink(missing_ok=True)
                    continue

                if lock_pid and not self._is_process_alive(lock_pid):
                    logger.warning("Removing stale build lock (PID %s no longer running)", lock_pid)
                    self._build_lock_path.unlink(missing_ok=True)
                    continue

                try:
                    age = time.time() - self._build_lock_path.stat().st_mtime
                    if age > 6 * 3600:
                        logger.warning("Removing stale build lock older than 6h")
                        self._build_lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass

                if not warned:
                    logger.info("Another as-help-server instance is rebuilding. Waiting for build lock...")
                    warned = True

                if time.time() - start > timeout_seconds:
                    raise TimeoutError("Timed out waiting for build lock") from None

                time.sleep(2)

    def _release_build_lock(self):
        if not self._build_lock_owned:
            return
        try:
            self._build_lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            self._build_lock_owned = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close database connection and release instance lock."""
        self._release_instance_lock()
        if self.embedder is not None and hasattr(self.embedder, "close"):
            try:
                self.embedder.close()
            except Exception:
                pass
        self.db = None

    def __del__(self):
        self.close()
