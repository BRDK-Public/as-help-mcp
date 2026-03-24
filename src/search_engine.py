"""LanceDB hybrid search engine with RRF (Reciprocal Rank Fusion).

Combines three search signals for superior relevance:
1. Title vector similarity (semantic title match)
2. Content vector similarity (semantic content match)
3. Full-text keyword search (weight 1.5x)
Results are fused using Reciprocal Rank Fusion (RRF).
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

import lancedb
import pyarrow as pa

from src.embeddings import EmbeddingService
from src.indexer import HelpContentIndexer

logger = logging.getLogger(__name__)

# Standard RRF constant (from the original RRF paper)
RRF_K = 60

# RRF weights for each search signal
WEIGHT_TITLE_VECTOR = 2.0
WEIGHT_CONTENT_VECTOR = 1.0
WEIGHT_FTS_KEYWORD = 1.5

# Number of pages per chunk during index build (saves progress after each chunk)
BUILD_CHUNK_SIZE = 5000


class HelpSearchEngine:
    """Hybrid search engine using LanceDB with RRF fusion.

    Combines title vector search, content vector search, and full-text keyword
    search via Reciprocal Rank Fusion for best-of-both-worlds relevance.
    """

    TABLE_NAME = "help_pages"

    # Class-level tracking of active db_paths in this process (for same-process detection)
    _active_db_paths: set[str] = set()
    _active_db_paths_lock = threading.Lock()

    def __init__(
        self,
        db_path: Path,
        indexer: HelpContentIndexer,
        force_rebuild: bool = False,
        embedding_service: EmbeddingService | None = None,
    ):
        """Initialize search engine.

        Args:
            db_path: Path to LanceDB directory
            indexer: HelpContentIndexer for accessing help data
            force_rebuild: Force rebuild even if valid index exists
            embedding_service: Optional EmbeddingService instance (created internally if not provided)
        """
        self.db_path = Path(db_path)
        self.indexer = indexer
        self.embedder = embedding_service or EmbeddingService()
        self._ready = threading.Event()
        self._fts_ready = threading.Event()
        self._build_error: Exception | None = None

        # Build status tracking (thread-safe via GIL for simple dict updates)
        self._build_status: dict = {
            "state": "initializing",  # initializing | building | fts_ready | ready | error
            "build_type": None,       # full | incremental | none
            "phase": "",              # current phase description
            "pages_total": 0,
            "pages_processed": 0,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "incremental_stats": None, # {added, removed, changed, unchanged} for incremental builds
        }

        # Ensure directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Connect to LanceDB (directory-based, lightweight)
        self.db = lancedb.connect(str(self.db_path))

        # Metadata sidecar file for change detection
        self._metadata_path = self.db_path / "_index_metadata.json"

        # Build progress marker for resume support
        self._build_progress_path = self.db_path / "_build_progress.json"
        # Build lock to avoid concurrent heavy rebuild/model-load in multiple processes
        self._build_lock_path = self.db_path / "_build.lock"
        self._build_lock_owned = False

        # Per-db-path instance lock: prevents multiple servers using the same database.
        # Different db paths (e.g. AS4 vs AS6) can run concurrently.
        self._instance_lock_path = self.db_path / "_instance.lock"
        self._instance_lock_owned = False
        self._acquire_instance_lock()

        # Determine build strategy: full, resume, incremental, or none
        if force_rebuild:
            self._build_strategy = "full"
        elif self._has_resumable_build():
            self._build_strategy = "resume"
        elif not self._index_exists():
            self._build_strategy = "full"
        else:
            self._build_strategy = self._detect_build_strategy()

        self._build_status["build_type"] = self._build_strategy

    def initialize(self):
        """Build or load the search index (may be slow on first run).

        For full/resume builds, uses a two-phase approach:
        - Phase 1: Extract text → write with zero vectors → build FTS → keyword search available
        - Phase 2: Load embedding model → embed → overwrite table → full hybrid search

        Call this after construction. Can be run in a background thread to
        avoid blocking the MCP server startup.

        Returns:
            self, for convenience chaining in tests.
        """
        try:
            self._build_status["started_at"] = time.time()
            self._build_status["state"] = "building"
            self._build_status["pages_total"] = len(self.indexer.pages)

            if self._build_strategy in ("full", "incremental", "resume"):
                self._acquire_build_lock()

            if self._build_strategy == "full":
                logger.info("Building new search index (full, two-phase)...")
                sys.stderr.flush()
                self._build_index_two_phase()
            elif self._build_strategy == "resume":
                resume_ids = self._get_indexed_page_ids()
                logger.info(f"Resuming interrupted build ({len(resume_ids)} pages already indexed)...")
                sys.stderr.flush()
                self._build_index_two_phase(resume_ids=resume_ids)
            elif self._build_strategy == "incremental":
                # Existing index already has FTS — make it available immediately
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
                # Existing index: both FTS and vectors are ready
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
        """Whether the search index is fully ready (hybrid search with vectors)."""
        # _ready is set in finally{} for both success and error to unblock waiters.
        # A usable index requires explicit ready state.
        return self._ready.is_set() and self._build_status.get("state") == "ready"

    @property
    def fts_ready(self) -> bool:
        """Whether keyword (FTS) search is available.

        True when the FTS index exists and can serve keyword queries,
        even if vector embeddings are still being computed.
        """
        return self._fts_ready.is_set() and self._build_status.get("state") in ("fts_ready", "ready")

    @property
    def build_status(self) -> dict:
        """Current build status snapshot (safe to read from any thread)."""
        status = dict(self._build_status)
        # Add elapsed time for in-progress builds
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

    def _index_exists(self) -> bool:
        """Check if LanceDB table exists and has data."""
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

    def _needs_reindex(self) -> bool:
        """Check if XML or embedding model has changed since last index."""
        if not self._metadata_path.exists():
            return True
        try:
            with open(self._metadata_path) as f:
                metadata = json.load(f)
            return (
                metadata.get("xml_hash") != self.indexer._get_xml_hash()
                or metadata.get("embedding_model") != self.embedder.model_name
            )
        except (json.JSONDecodeError, KeyError, OSError):
            return True

    def _detect_build_strategy(self) -> str:
        """Determine whether we need a full rebuild, incremental update, or nothing.

        Returns:
            "full" - No metadata, embedding model changed, or no stored fingerprints
            "incremental" - XML changed but we have page fingerprints to diff against
            "none" - Nothing changed
        """
        if not self._metadata_path.exists():
            return "full"
        try:
            with open(self._metadata_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            return "full"

        # Embedding model change → must re-embed everything
        if metadata.get("embedding_model") != self.embedder.model_name:
            logger.info("Embedding model changed - full rebuild required")
            return "full"

        # XML unchanged → nothing to do
        if metadata.get("xml_hash") == self.indexer._get_xml_hash():
            return "none"

        # XML changed — can we do incremental?
        if metadata.get("page_fingerprints"):
            logger.info("XML changed - incremental update possible")
            return "incremental"

        # XML changed but no stored fingerprints → full rebuild
        logger.info("XML changed but no page fingerprints stored - full rebuild required")
        return "full"

    def _save_metadata(self):
        """Save index metadata including per-page fingerprints to JSON sidecar file."""
        metadata = {
            "xml_hash": self.indexer._get_xml_hash(),
            "indexed_at": time.time(),
            "page_count": len(self.indexer.pages),
            "help_id_count": len(self.indexer.help_id_map),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "page_fingerprints": self.indexer.get_page_fingerprints(),
        }
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _extract_text_for_page(self, page_id: str, page) -> tuple:
        """Extract text for a single page (used for parallel processing).

        Args:
            page_id: The page ID
            page: The HelpPage object

        Returns:
            Tuple of (page_id, title, content, file_path, help_id, is_section, breadcrumb_path, category)
        """
        plain_text = ""
        if not page.is_section:
            # Extract text WITHOUT caching (save memory during indexing)
            plain_text = self.indexer._extract_plain_text_no_cache(page) or ""

        # Get pre-computed breadcrumb path
        breadcrumb_path = self.indexer.get_breadcrumb_string(page_id)

        # Extract category (top-level section) from breadcrumb for efficient filtering
        category = ""
        breadcrumb = self.indexer.get_breadcrumb(page_id)
        if breadcrumb and len(breadcrumb) > 0:
            # First item in breadcrumb is the root category
            category = breadcrumb[0].text

        return (
            page_id,
            page.text,  # title
            plain_text,  # content
            page.file_path,
            page.help_id or "",
            1 if page.is_section else 0,
            breadcrumb_path,
            category,
        )

    def _has_resumable_build(self) -> bool:
        """Check if there's a partial build from an interrupted session that can be resumed."""
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
            return (
                progress.get("xml_hash") == self.indexer._get_xml_hash()
                and progress.get("embedding_model") == self.embedder.model_name
            )
        except (json.JSONDecodeError, OSError, Exception):
            return False

    def _save_build_progress(self):
        """Save marker that a build is in progress (for resume detection on restart)."""
        progress = {
            "xml_hash": self.indexer._get_xml_hash(),
            "embedding_model": self.embedder.model_name,
        }
        with open(self._build_progress_path, "w") as f:
            json.dump(progress, f)

    # ------------------------------------------------------------------
    # Instance lock: one server per db_path
    # ------------------------------------------------------------------

    def _acquire_instance_lock(self):
        """Acquire a per-db-path instance lock. Raises if another live server holds it."""
        resolved = str(self.db_path.resolve())

        # Same-process check (e.g. two engines in the same Python process)
        with self._active_db_paths_lock:
            if resolved in self._active_db_paths:
                raise RuntimeError(
                    f"Another HelpSearchEngine in this process is already using "
                    f"database at {self.db_path}. Close it first, "
                    f"or use a different --db-path for each help root."
                )

        # Cross-process check via lock file
        lock_info = self._read_instance_lock()
        if lock_info is not None:
            other_pid = lock_info.get("pid")
            if other_pid and self._is_process_alive(other_pid) and other_pid != os.getpid():
                raise RuntimeError(
                    f"Another as-help-server (PID {other_pid}) is already using "
                    f"database at {self.db_path}. Stop the other instance first, "
                    f"or use a different --db-path for each help root."
                )
            # Stale lock from a dead process -- safe to overwrite
            if other_pid and other_pid != os.getpid():
                logger.debug("Overwriting stale instance lock (PID %s no longer running)", other_pid)

        self._write_instance_lock()
        with self._active_db_paths_lock:
            self._active_db_paths.add(resolved)
        self._instance_lock_owned = True
        logger.info("Acquired instance lock for %s (PID %s)", self.db_path, os.getpid())

    def _release_instance_lock(self):
        """Release instance lock if owned by this process."""
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
        """Read instance lock file, or None if absent/corrupt."""
        try:
            with open(self._instance_lock_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _write_instance_lock(self):
        """Write current PID to instance lock file."""
        with open(self._instance_lock_path, "w") as f:
            json.dump({"pid": os.getpid(), "started_at": time.time()}, f)

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check whether a process with the given PID is still running."""
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
    # Build lock: serialises heavy rebuild/model-load across processes
    # ------------------------------------------------------------------

    def _acquire_build_lock(self, timeout_seconds: int = 1800):
        """Acquire a file-based lock so only one process runs heavy build work."""
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
                # Reap stale lock files from crashed processes.
                try:
                    lock_data = json.loads(self._build_lock_path.read_text())
                    lock_pid = lock_data.get("pid")
                except (OSError, json.JSONDecodeError, ValueError):
                    lock_pid = None

                # If we hold the instance lock, we know no other live server
                # is using this db path — any build lock is leftover from a crash.
                if self._instance_lock_owned:
                    logger.warning("Removing stale build lock (this process holds instance lock)")
                    self._build_lock_path.unlink(missing_ok=True)
                    continue

                # Check if the lock holder is still alive
                if lock_pid and not self._is_process_alive(lock_pid):
                    logger.warning("Removing stale build lock (PID %s no longer running)", lock_pid)
                    self._build_lock_path.unlink(missing_ok=True)
                    continue

                # Fallback: reap very old locks (e.g. PID was recycled)
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
        """Release build lock if owned by this process."""
        if not self._build_lock_owned:
            return
        try:
            self._build_lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            self._build_lock_owned = False

    def _clear_build_progress(self):
        """Remove build progress marker (build completed successfully)."""
        if self._build_progress_path.exists():
            self._build_progress_path.unlink()

    def _get_indexed_page_ids(self) -> set[str]:
        """Get set of page_ids already in the LanceDB table."""
        try:
            table = self.db.open_table(self.TABLE_NAME)
            # Read only page_id column via lance scanner to avoid loading vectors
            try:
                arrow_table = table.to_lance().to_table(columns=["page_id"])
                return set(arrow_table["page_id"].to_pylist())
            except (AttributeError, Exception):
                # Fallback: read full table (slower but always works)
                df = table.to_pandas()
                return set(df["page_id"].tolist())
        except Exception as e:
            logger.warning(f"Could not read existing page IDs: {e}")
            return set()

    def _get_table_schema(self) -> pa.Schema:
        """Get the PyArrow schema for the LanceDB help_pages table."""
        dim = self.embedder.dimension
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

    def _records_to_arrow(self, records, title_vectors, content_vectors) -> pa.Table:
        """Convert extracted records + embeddings to a PyArrow table."""
        search_texts = [f"{r[1]} {r[2]}" for r in records]
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
        return pa.table(data, schema=self._get_table_schema())

    def _build_content_vectors(self, records, title_vectors) -> list[list[float]]:
        """Build content vectors efficiently.

        For sections (empty content), we reuse title vectors instead of embedding
        the same title text again. For regular pages, we embed only non-empty
        content strings.
        """
        content_indices: list[int] = []
        content_texts: list[str] = []

        for idx, record in enumerate(records):
            content = record[2]
            if content:
                content_indices.append(idx)
                content_texts.append(content)

        # Default: reuse title vectors (especially useful for section rows)
        content_vectors = list(title_vectors)

        if content_texts:
            embedded_contents = self.embedder.embed_batch(content_texts)
            for idx, vec in zip(content_indices, embedded_contents, strict=True):
                content_vectors[idx] = vec

        return content_vectors

    def _build_index_two_phase(self, resume_ids: set[str] | None = None):
        """Build search index in two phases: FTS-first, then vectors.

        Phase 1: Extract text → write to LanceDB with zero vectors → build FTS index
                 → set _fts_ready (keyword search available immediately)
        Phase 2: Load embedding model → compute real vectors → overwrite table
                 → rebuild FTS → set _ready (full hybrid search)

        Args:
            resume_ids: Set of page_ids already indexed (for resume). None for fresh build.
        """
        start_time = time.time()
        all_pages = list(self.indexer.pages.items())
        total_pages = len(all_pages)

        if resume_ids:
            pages_to_process = [(pid, page) for pid, page in all_pages if pid not in resume_ids]
            already_done = total_pages - len(pages_to_process)
            table_created = True
            logger.info(f"Resuming: {already_done} already indexed, {len(pages_to_process)} remaining")
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
            self._finalize_build(start_time, total_pages)
            self._fts_ready.set()
            self._build_status["state"] = "fts_ready"
            return

        self._save_build_progress()

        max_workers = min(int(os.cpu_count() or 4), 10)
        remaining = len(pages_to_process)
        total_chunks = (remaining + BUILD_CHUNK_SIZE - 1) // BUILD_CHUNK_SIZE
        dim = self.embedder.dimension

        # ── Phase 1: Extract text, write with zero vectors, build FTS ──
        logger.info(f"Phase 1: Extracting text for {remaining} pages ({total_chunks} chunks)...")
        sys.stderr.flush()

        all_records: list[tuple] = []

        for chunk_start in range(0, remaining, BUILD_CHUNK_SIZE):
            chunk = pages_to_process[chunk_start:chunk_start + BUILD_CHUNK_SIZE]
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

            all_records.extend(records)

            # Write chunk with zero vectors
            zero_title_vectors = [[0.0] * dim for _ in records]
            zero_content_vectors = [[0.0] * dim for _ in records]
            chunk_data = self._records_to_arrow(records, zero_title_vectors, zero_content_vectors)

            self._build_status["phase"] = f"saving text (chunk {chunk_num}/{total_chunks})"
            if not table_created:
                self.db.create_table(self.TABLE_NAME, chunk_data)
                table_created = True
            else:
                table = self.db.open_table(self.TABLE_NAME)
                table.add(chunk_data)

            processed = already_done + chunk_start + len(chunk)
            self._build_status["pages_processed"] = processed
            logger.info(f"Phase 1 chunk {chunk_num}/{total_chunks}: {processed}/{total_pages} pages")
            sys.stderr.flush()

        # Build FTS index → keyword search available
        self._build_status["phase"] = "creating FTS index (keyword search)"
        logger.info("Creating FTS index...")
        table = self.db.open_table(self.TABLE_NAME)
        table.create_fts_index("search_text", replace=True)

        self._fts_ready.set()
        self._build_status["state"] = "fts_ready"
        phase1_elapsed = time.time() - start_time
        logger.info(f"Phase 1 complete in {phase1_elapsed:.1f}s — keyword search is now available")
        sys.stderr.flush()

        # ── Phase 2: Load embedding model, compute real vectors, overwrite table ──
        self._build_status["phase"] = "loading embedding model"
        logger.info("Phase 2: Loading embedding model...")
        sys.stderr.flush()
        self.embedder._load_model()

        # Embed in chunks to avoid excessive memory usage
        logger.info(f"Phase 2: Embedding {len(all_records)} pages...")
        all_title_vectors: list[list[float]] = []
        all_content_vectors: list[list[float]] = []

        for chunk_start in range(0, len(all_records), BUILD_CHUNK_SIZE):
            chunk_records = all_records[chunk_start:chunk_start + BUILD_CHUNK_SIZE]
            chunk_num = chunk_start // BUILD_CHUNK_SIZE + 1

            titles = [r[1] for r in chunk_records]
            self._build_status["phase"] = f"embedding (chunk {chunk_num}/{total_chunks})"
            title_vectors = self.embedder.embed_batch(titles)
            content_vectors = self._build_content_vectors(chunk_records, title_vectors)

            all_title_vectors.extend(title_vectors)
            all_content_vectors.extend(content_vectors)

            logger.info(f"Phase 2 embedding chunk {chunk_num}/{total_chunks}")
            sys.stderr.flush()

        # Overwrite the table with real vectors
        self._build_status["phase"] = "writing vectors to index"
        logger.info("Writing vectors to index...")
        full_data = self._records_to_arrow(all_records, all_title_vectors, all_content_vectors)
        self.db.drop_table(self.TABLE_NAME)
        self.db.create_table(self.TABLE_NAME, full_data)

        # Rebuild FTS after table overwrite
        self._build_status["phase"] = "rebuilding FTS index"
        logger.info("Rebuilding FTS index...")
        table = self.db.open_table(self.TABLE_NAME)
        table.create_fts_index("search_text", replace=True)

        self._save_metadata()
        self._clear_build_progress()

        self._build_status["pages_processed"] = total_pages
        elapsed = time.time() - start_time
        logger.info(f"Phase 2 complete — full hybrid search ready in {elapsed:.1f}s ({total_pages} documents)")

    def _build_index(self, resume_ids: set[str] | None = None):
        """Build search index in chunks with resume support.

        Processes pages in chunks of BUILD_CHUNK_SIZE, writing each chunk to
        LanceDB immediately.  If the server is killed mid-build, the next
        start detects the partial table and resumes where it left off.

        Args:
            resume_ids: Set of page_ids already indexed (for resume). None for fresh build.
        """
        start_time = time.time()
        all_pages = list(self.indexer.pages.items())
        total_pages = len(all_pages)

        # For resume: skip already-indexed pages
        if resume_ids:
            pages_to_process = [(pid, page) for pid, page in all_pages if pid not in resume_ids]
            already_done = total_pages - len(pages_to_process)
            table_created = True  # Table exists from the interrupted build
            logger.info(f"Resuming: {already_done} already indexed, {len(pages_to_process)} remaining")
        else:
            pages_to_process = all_pages
            already_done = 0
            table_created = False
            # Clean start - drop any leftover partial table
            try:
                if self.TABLE_NAME in self.db.list_tables().tables:
                    self.db.drop_table(self.TABLE_NAME)
            except Exception:
                pass

        self._build_status["pages_total"] = total_pages
        self._build_status["pages_processed"] = already_done

        if not pages_to_process:
            logger.info("All pages already indexed, finalizing...")
            self._finalize_build(start_time, total_pages)
            return

        # Save progress marker so we can resume if interrupted
        self._save_build_progress()

        # Cap workers to avoid starving the MCP event loop / stdio transport
        max_workers = min(int(os.cpu_count() or 4), 10)
        remaining = len(pages_to_process)
        total_chunks = (remaining + BUILD_CHUNK_SIZE - 1) // BUILD_CHUNK_SIZE

        logger.info(f"Processing {remaining} pages in {total_chunks} chunks ({max_workers} workers)...")
        sys.stderr.flush()

        for chunk_start in range(0, remaining, BUILD_CHUNK_SIZE):
            chunk = pages_to_process[chunk_start:chunk_start + BUILD_CHUNK_SIZE]
            chunk_num = chunk_start // BUILD_CHUNK_SIZE + 1
            chunk_time = time.time()

            # 1. Extract text
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

            # 2. Embed titles
            titles = [r[1] for r in records]
            self._build_status["phase"] = f"embedding (chunk {chunk_num}/{total_chunks})"
            title_vectors = self.embedder.embed_batch(titles)

            # 3. Embed content (reuse title vectors for section rows)
            content_vectors = self._build_content_vectors(records, title_vectors)

            # 4. Write chunk to LanceDB
            chunk_data = self._records_to_arrow(records, title_vectors, content_vectors)
            self._build_status["phase"] = f"saving (chunk {chunk_num}/{total_chunks})"

            if not table_created:
                self.db.create_table(self.TABLE_NAME, chunk_data)
                table_created = True
            else:
                table = self.db.open_table(self.TABLE_NAME)
                table.add(chunk_data)

            processed = already_done + chunk_start + len(chunk)
            self._build_status["pages_processed"] = processed
            chunk_elapsed = time.time() - chunk_time
            logger.info(
                f"Chunk {chunk_num}/{total_chunks}: {processed}/{total_pages} pages ({chunk_elapsed:.1f}s)"
            )
            sys.stderr.flush()

        self._finalize_build(start_time, total_pages)

    def _finalize_build(self, start_time: float, total_pages: int):
        """Create FTS index and save metadata after all chunks are written."""
        table = self.db.open_table(self.TABLE_NAME)

        self._build_status["phase"] = "creating FTS index"
        logger.info("Creating FTS index...")
        table.create_fts_index("search_text", replace=True)

        self._save_metadata()
        self._clear_build_progress()

        self._build_status["pages_processed"] = total_pages
        elapsed = time.time() - start_time
        logger.info(f"Search index built successfully in {elapsed:.1f}s ({total_pages} documents)")

    def _incremental_update(self):
        """Incrementally update the index by diffing page fingerprints.

        Compares stored per-page fingerprints against current ones to find
        added, removed, and changed pages.  Only those pages are re-extracted,
        re-embedded, and written to LanceDB.  The FTS index is rebuilt at the
        end (required by LanceDB after row mutations).
        """
        start_time = time.time()

        # Load old fingerprints from metadata
        with open(self._metadata_path) as f:
            metadata = json.load(f)
        old_fps: dict[str, str] = metadata.get("page_fingerprints", {})

        # Compute current fingerprints from the freshly-parsed XML
        new_fps = self.indexer.get_page_fingerprints()

        old_ids = set(old_fps.keys())
        new_ids = set(new_fps.keys())

        added = new_ids - old_ids
        removed = old_ids - new_ids
        changed = {pid for pid in (old_ids & new_ids) if old_fps[pid] != new_fps[pid]}

        to_upsert = added | changed
        to_delete = removed | changed  # delete old rows for changed pages, then re-add

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

        # If nothing changed (e.g. XML whitespace change), skip heavy work
        if not to_upsert and not to_delete:
            logger.info("No page-level changes detected - skipping update")
            self._save_metadata()
            return

        # Fall back to full rebuild if >50% of pages changed (not worth incremental overhead)
        if len(to_upsert) > len(new_ids) * 0.5:
            logger.info(
                f"Too many changes ({len(to_upsert)}/{len(new_ids)}) - falling back to full rebuild"
            )
            self._build_index()
            return

        table = self.db.open_table(self.TABLE_NAME)

        # --- Delete removed and changed rows ---
        if to_delete:
            self._build_status["phase"] = "deleting old rows"
            # Build safe filter: page_id IN ('id1', 'id2', ...)
            # Process in batches to avoid oversized SQL expressions
            delete_list = list(to_delete)
            batch_size = 500
            for i in range(0, len(delete_list), batch_size):
                batch = delete_list[i : i + batch_size]
                id_literals = ", ".join(f"'{pid}'" for pid in batch)
                table.delete(f"page_id IN ({id_literals})")
            logger.info(f"Deleted {len(to_delete)} rows from index")

        # --- Extract text, embed, and insert new/changed pages ---
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

            # Embed
            titles = [r[1] for r in records]
            self._build_status["phase"] = "embedding titles"
            title_vectors = self.embedder.embed_batch(titles)

            self._build_status["phase"] = "embedding content"
            content_vectors = self._build_content_vectors(records, title_vectors)

            new_data = self._records_to_arrow(records, title_vectors, content_vectors)
            table.add(new_data)
            logger.info(f"Added {len(records)} rows to index")

        # Rebuild FTS index (required after row mutations)
        self._build_status["phase"] = "rebuilding FTS index"
        logger.info("Rebuilding FTS index...")
        table.create_fts_index("search_text", replace=True)

        # Save updated metadata with new fingerprints
        self._save_metadata()

        elapsed = time.time() - start_time
        logger.info(
            f"Incremental update complete in {elapsed:.1f}s "
            f"(+{len(added)} -{len(removed)} ~{len(changed)} pages)"
        )

    def _load_index(self):
        """Load existing search index and log stats."""
        table = self.db.open_table(self.TABLE_NAME)
        doc_count = table.count_rows()
        logger.info(f"Loaded search index with {doc_count} documents")

    @staticmethod
    def _build_category_filter(category: str | None) -> str | None:
        """Build a safe SQL where clause for category filtering.

        Sanitizes input to prevent SQL injection in LanceDB filter expressions.
        """
        if not category:
            return None
        # Strip characters that could be used for SQL injection
        safe = re.sub(r"[^\w\s.-]", "", category)
        return f"lower(category) = '{safe.lower()}'"

    def _vector_search(self, table, query_vector: list[float], column_name: str, limit: int, where_clause: str | None) -> list[dict]:
        """Run vector similarity search on a specific column."""
        try:
            builder = table.search(query_vector, vector_column_name=column_name)
            if where_clause:
                builder = builder.where(where_clause)
            return builder.limit(limit).to_list()
        except Exception as e:
            logger.warning(f"Vector search on {column_name} failed: {e}")
            return []

    def _fts_search(self, table, query: str, limit: int, where_clause: str | None) -> list[dict]:
        """Run full-text keyword search with query sanitization."""
        # Sanitize query: remove FTS special characters
        sanitized = query
        for char in '"\'*:(){}^+[]-':
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
            return builder.limit(limit).to_list()
        except Exception as e:
            logger.warning(f"FTS search failed for '{fts_query}': {e}")
            return []

    @staticmethod
    def _generate_snippet(content: str, query: str) -> str | None:
        """Generate a text snippet around the first matching term."""
        if not content:
            return None

        # Parse search terms from query
        sanitized = query
        for char in '"\'*:(){}^+[]-':
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

    def search(
        self, query: str, limit: int = 20, search_in_content: bool = True, category: str | None = None
    ) -> list[dict]:
        """Search for help pages using hybrid search with RRF fusion.

        When the index is in fts_ready state (vectors not yet computed), falls
        back to keyword-only search. Once fully ready, uses full hybrid RRF.

        Args:
            query: Search query (keywords or natural language)
            limit: Maximum number of results
            search_in_content: Search in content (True) or titles only (False)
            category: Optional category filter (case-insensitive)

        Returns:
            List of search results with page_id, title, file_path, help_id,
            is_section, breadcrumb_path, category, score, snippet, search_mode
        """
        if not query.strip():
            return []

        table = self.db.open_table(self.TABLE_NAME)
        where_clause = self._build_category_filter(category)

        # Determine search mode based on readiness
        use_vectors = self.ready  # Full hybrid only when vectors are computed

        if use_vectors:
            # Full hybrid search: vectors + FTS with RRF fusion
            query_vector = self.embedder.embed_text(query)
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
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + WEIGHT_TITLE_VECTOR / (RRF_K + rank + 1)
                if pid not in page_data:
                    page_data[pid] = row

            for rank, row in enumerate(content_results):
                pid = row["page_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + WEIGHT_CONTENT_VECTOR / (RRF_K + rank + 1)
                if pid not in page_data:
                    page_data[pid] = row

            for rank, row in enumerate(fts_results):
                pid = row["page_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + WEIGHT_FTS_KEYWORD / (RRF_K + rank + 1)
                if pid not in page_data:
                    page_data[pid] = row

            sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)[:limit]
            search_mode = "hybrid"
        else:
            # FTS-only mode (vectors not yet available)
            fts_results = self._fts_search(table, query, limit, where_clause)
            rrf_scores = {}
            page_data = {}

            for rank, row in enumerate(fts_results):
                pid = row["page_id"]
                rrf_scores[pid] = WEIGHT_FTS_KEYWORD / (RRF_K + rank + 1)
                page_data[pid] = row

            sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)[:limit]
            search_mode = "keyword"

        # Build result dicts
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

    def close(self):
        """Close database connection and release instance lock."""
        self._release_instance_lock()
        self.db = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
