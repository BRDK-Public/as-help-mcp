"""SQLite FTS5 search engine with persistent indexing."""

import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.indexer import HelpContentIndexer

logger = logging.getLogger(__name__)


class HelpSearchEngine:
    """Full-text search engine using SQLite FTS5."""

    def __init__(self, db_path: Path, indexer: HelpContentIndexer, force_rebuild: bool = False):
        """Initialize search engine.

        Args:
            db_path: Path to SQLite database file
            indexer: HelpContentIndexer for accessing help data
            force_rebuild: Force rebuild even if valid index exists
        """
        self.db_path = Path(db_path)
        self.indexer = indexer

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database with thread safety for async MCP handlers
        # check_same_thread=False is safe because we don't use multiple concurrent writers
        # timeout=30 allows waiting for locks (important for Docker/volume mounts)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Enable WAL mode for better concurrent access on volume mounts
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            logger.info("SQLite WAL mode enabled for better concurrent access")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not enable WAL mode: {e} (continuing with default journal)")  # pragma: no cover

        # Determine if we need to build the index
        needs_build = force_rebuild or not self._index_exists() or self._needs_reindex()

        if needs_build:
            logger.info("Building new search index...")
            self._build_index()
        else:
            logger.info("Loading existing search index...")
            self._load_index()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _index_exists(self) -> bool:
        """Check if FTS5 table exists and has data."""
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='help_fts'
        """)
        if not cursor.fetchone():
            return False

        # Also check if the table actually has data
        cursor = self.conn.execute("SELECT COUNT(*) FROM help_fts")
        count = cursor.fetchone()[0]
        if count == 0:
            logger.info("FTS5 table exists but is empty - will rebuild")  # pragma: no cover
            return False  # pragma: no cover

        return True

    def _needs_reindex(self) -> bool:
        """Check if XML has changed since last index."""
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='index_metadata'
        """)
        if not cursor.fetchone():
            return True  # pragma: no cover

        cursor = self.conn.execute("SELECT xml_hash FROM index_metadata LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return True  # pragma: no cover

        return row[0] != self.indexer._get_xml_hash()

    def _create_tables(self):
        """Create FTS5 table and metadata table."""
        # Create FTS5 virtual table for search
        # Note: category is stored as UNINDEXED for exact-match filtering (faster than LIKE on file_path)
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS help_fts USING fts5(
                page_id UNINDEXED,
                title,
                content,
                file_path UNINDEXED,
                help_id UNINDEXED,
                is_section UNINDEXED,
                breadcrumb_path UNINDEXED,
                category UNINDEXED,
                tokenize='porter unicode61'
            )
        """)

        # Create metadata table for tracking changes
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                xml_hash TEXT PRIMARY KEY,
                indexed_at REAL,
                page_count INTEGER,
                help_id_count INTEGER
            )
        """)

        # Don't commit here - let caller control transaction

    def _load_index(self):
        """Load existing search index."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM help_fts")
        doc_count = cursor.fetchone()[0]
        logger.info(f"Loaded search index with {doc_count} documents")

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

    def _build_index(self):
        """Build search index from help content with parallel text extraction."""
        start_time = time.time()

        # Close any existing connection and remove lock files to ensure clean state
        # This is critical on Docker volume mounts where file locks can persist
        try:
            if self.conn:
                self.conn.close()
                logger.info("Closed existing connection for clean rebuild")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Error closing existing connection: {e}")  # pragma: no cover

        # Remove lock/journal files that might be stale (esp. on volume mounts)
        for lock_file in [f"{self.db_path}-wal", f"{self.db_path}-shm", f"{self.db_path}.lock"]:
            try:
                lock_path = Path(lock_file)
                if lock_path.exists():
                    lock_path.unlink()  # pragma: no cover
                    logger.info(f"Removed stale lock file: {lock_file}")  # pragma: no cover
            except Exception as e:  # pragma: no cover
                logger.warning(f"Could not remove lock file {lock_file}: {e}")  # pragma: no cover

        # Reconnect with fresh connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode if not already enabled
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not enable WAL mode: {e}")  # pragma: no cover

        cursor = self.conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF")  # Disable sync during build
        cursor.execute("PRAGMA journal_mode = MEMORY")  # Keep journal in memory
        cursor.execute("PRAGMA cache_size = -64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in memory

        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS help_fts")
        cursor.execute("DROP TABLE IF EXISTS index_metadata")

        # Create tables
        self._create_tables()

        # Start single transaction for entire index
        cursor.execute("BEGIN TRANSACTION")

        # Disable FTS5 automerge during bulk insert (prevents slowdown at ~25k pages)
        cursor.execute("INSERT INTO help_fts(help_fts, rank) VALUES('automerge', 0)")
        cursor.execute("INSERT INTO help_fts(help_fts, rank) VALUES('crisismerge', 16384)")

        # Prepare batch insert
        indexed_count = 0
        batch_size = 1000  # Larger batches since we're in one transaction
        batch = []

        logger.info(f"Indexing {len(self.indexer.pages)} pages...")

        # Use ThreadPoolExecutor for parallel I/O operations
        max_workers = int(os.cpu_count() or 4)
        logger.info(f"Extracting text content in parallel using {max_workers} CPUs (this may take 1-2 minutes)...")

        # Process in chunks to avoid memory buildup and FTS5 segment merge delays
        all_pages = list(self.indexer.pages.items())

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Use map() with chunksize for memory-efficient streaming of results
                # This avoids holding all 10K futures in memory at once
                extraction_results = executor.map(
                    lambda item: self._extract_text_for_page(item[0], item[1]),
                    all_pages,
                    chunksize=min(100, max(1, max_workers * 2)),
                )

                chunk_start_time = time.time()
                chunk_indexed_start = 0

                # Process results as they complete (streaming)
                for result in extraction_results:
                    try:
                        batch.append(result)
                        indexed_count += 1

                        # Insert batch when full (no commit, still in transaction)
                        if len(batch) >= batch_size:
                            cursor.executemany(
                                """
                                INSERT INTO help_fts (page_id, title, content, file_path, help_id, is_section, breadcrumb_path, category)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                batch,
                            )
                            batch = []

                            if indexed_count % 5000 == 0:
                                elapsed = time.time() - start_time
                                pages_per_sec = indexed_count / elapsed
                                recent_elapsed = time.time() - chunk_start_time
                                recent_rate = (
                                    (indexed_count - chunk_indexed_start) / recent_elapsed if recent_elapsed > 0 else 0
                                )
                                eta = (len(self.indexer.pages) - indexed_count) / pages_per_sec
                                logger.info(
                                    f"Indexed {indexed_count}/{len(self.indexer.pages)} pages... "
                                    f"(avg: {pages_per_sec:.0f} p/s, recent: {recent_rate:.0f} p/s, ETA: {eta:.0f}s)"
                                )

                    except Exception as e:  # pragma: no cover
                        # result[0] is page_id from _extract_text_for_page
                        page_id = (
                            result[0] if isinstance(result, tuple) and len(result) > 0 else "unknown"
                        )  # pragma: no cover
                        logger.warning(f"Failed to extract text for page {page_id}: {e}")  # pragma: no cover
                        # Add empty entry to maintain progress
                        page = self.indexer.pages.get(page_id)  # pragma: no cover
                        if page:  # pragma: no cover
                            breadcrumb_path = self.indexer.get_breadcrumb_string(page_id)  # pragma: no cover
                            breadcrumb = self.indexer.get_breadcrumb(page_id)  # pragma: no cover
                            category = breadcrumb[0].text if breadcrumb else ""  # pragma: no cover
                            batch.append(  # pragma: no cover
                                (  # pragma: no cover
                                    page_id,  # pragma: no cover
                                    page.text,  # pragma: no cover
                                    "",  # empty content on error
                                    page.file_path,  # pragma: no cover
                                    page.help_id or "",  # pragma: no cover
                                    1 if page.is_section else 0,  # pragma: no cover
                                    breadcrumb_path,  # pragma: no cover
                                    category,  # pragma: no cover
                                )  # pragma: no cover
                            )  # pragma: no cover
                            indexed_count += 1  # pragma: no cover

            # Insert remaining batch
            if batch:
                cursor.executemany(
                    """
                    INSERT INTO help_fts (page_id, title, content, file_path, help_id, is_section, breadcrumb_path, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    batch,
                )

            # Save metadata
            cursor.execute(
                """
                INSERT INTO index_metadata (xml_hash, indexed_at, page_count, help_id_count)
                VALUES (?, ?, ?, ?)
            """,
                (self.indexer._get_xml_hash(), time.time(), len(self.indexer.pages), len(self.indexer.help_id_map)),
            )

            # Commit the single transaction
            cursor.execute("COMMIT")

            # Re-enable automerge and optimize the FTS5 index
            logger.info("Optimizing FTS5 index...")
            cursor.execute("INSERT INTO help_fts(help_fts, rank) VALUES('automerge', 8)")
            cursor.execute("INSERT INTO help_fts(help_fts) VALUES('optimize')")
            self.conn.commit()

            # Restore safe settings
            cursor.execute("PRAGMA synchronous = FULL")
            cursor.execute("PRAGMA journal_mode = DELETE")

            elapsed = time.time() - start_time
            logger.info(
                f"Search index built successfully in {elapsed:.1f}s ({indexed_count} documents)"
            )  # pragma: no cover

        except Exception as e:  # pragma: no cover
            try:  # pragma: no cover
                if cursor:  # pragma: no cover
                    cursor.execute("ROLLBACK")  # pragma: no cover
            except Exception as rollback_error:  # pragma: no cover
                logger.warning(f"Rollback failed: {rollback_error}")  # pragma: no cover
            logger.error(f"Index build failed, rolled back: {e}")
            raise

    def search(
        self, query: str, limit: int = 20, search_in_content: bool = True, category: str | None = None
    ) -> list[dict]:
        """Search for help pages.

        Args:
            query: Search query
            limit: Maximum number of results
            search_in_content: Search in content (True) or titles only (False)
            category: Optional category filter (matches start of file_path)

        Returns:
            List of search results with page_id, title, file_path, help_id, is_section, score
        """
        if not query.strip():
            return []

        # Sanitize query: remove FTS5 special characters that could break syntax
        # See: https://www.sqlite.org/fts5.html#full_text_query_syntax
        sanitized = query.replace('"', " ").replace("'", " ").replace("*", " ")
        sanitized = sanitized.replace(":", " ").replace("(", " ").replace(")", " ")
        sanitized = sanitized.replace("{", " ").replace("}", " ").replace("-", " ")
        sanitized = sanitized.replace("^", " ").replace("+", " ")  # Boost and NEAR operators
        sanitized = sanitized.replace("[", " ").replace("]", " ")  # Bracket syntax

        # Split into terms and filter out short terms and FTS5 keywords
        fts5_keywords = {"and", "or", "not", "near"}  # Case-insensitive FTS5 operators
        terms = [t.strip() for t in sanitized.split() if len(t.strip()) >= 2 and t.strip().lower() not in fts5_keywords]
        if not terms:
            return []

        # Build enhanced query with prefix matching for partial words
        # e.g., "motor speed" -> '"motor"* "speed"*' to match "motors", "speeding", etc.
        enhanced_terms = [f'"{term}"*' for term in terms]

        # Build FTS5 query
        if search_in_content:
            fts_query = " ".join(enhanced_terms)
        else:
            fts_query = f"title : {' '.join(enhanced_terms)}"

        try:
            # Build SQL query
            sql = """
                SELECT
                    page_id,
                    title,
                    file_path,
                    help_id,
                    is_section,
                    breadcrumb_path,
                    category,
                    bm25(help_fts, 10.0, 1.0) as score,
                    snippet(help_fts, 2, '>>>', '<<<', '...', 32) as snippet
                FROM help_fts
                WHERE help_fts MATCH ?
            """
            params = [fts_query]

            # Add category filter if provided
            # Uses dedicated category column for efficient exact matching (vs LIKE on file_path)
            if category:
                # Case-insensitive comparison using LOWER()
                sql += " AND LOWER(category) = LOWER(?)"
                params.append(category)

            sql += """
                ORDER BY bm25(help_fts, 10.0, 1.0)
                LIMIT ?
            """
            params.append(str(limit))

            # Execute query
            cursor = self.conn.execute(sql, params)

            results = []
            for row in cursor:
                results.append(
                    {
                        "page_id": row["page_id"],
                        "title": row["title"],
                        "file_path": row["file_path"],
                        "help_id": row["help_id"] if row["help_id"] else None,
                        "is_section": bool(row["is_section"]),
                        "breadcrumb_path": row["breadcrumb_path"] if row["breadcrumb_path"] else None,
                        "category": row["category"] if row["category"] else None,
                        "score": abs(float(row["score"])),  # BM25 returns negative
                        "snippet": row["snippet"] if row["snippet"] else None,
                    }
                )

            logger.info(f"Search for '{query}' (cat={category}) returned {len(results)} results")  # pragma: no cover
            return results  # pragma: no cover

        except sqlite3.OperationalError as e:  # pragma: no cover
            logger.error(f"Search error for query '{query}': {e}")  # pragma: no cover
            # Fallback: try simple OR search if enhanced query fails
            try:  # pragma: no cover
                fallback_terms = " OR ".join([f'"{t}"' for t in terms])  # pragma: no cover
                sql = """# pragma: no cover
                    SELECT page_id, title, file_path, help_id, is_section, breadcrumb_path,
                           category, bm25(help_fts, 10.0, 1.0) as score,
                           snippet(help_fts, 2, '>>>', '<<<', '...', 32) as snippet
                    FROM help_fts WHERE help_fts MATCH ?
                """  # pragma: no cover
                params = [fallback_terms]  # pragma: no cover

                if category:  # pragma: no cover
                    sql += " AND LOWER(category) = LOWER(?)"  # pragma: no cover
                    params.append(category)  # pragma: no cover

                sql += " ORDER BY bm25(help_fts, 10.0, 1.0) LIMIT ?"  # pragma: no cover
                params.append(str(limit))  # pragma: no cover

                cursor = self.conn.execute(sql, params)  # pragma: no cover
                results = [  # pragma: no cover
                    {  # pragma: no cover
                        "page_id": r["page_id"],  # pragma: no cover
                        "title": r["title"],  # pragma: no cover
                        "file_path": r["file_path"],  # pragma: no cover
                        "help_id": r["help_id"] or None,  # pragma: no cover
                        "is_section": bool(r["is_section"]),  # pragma: no cover
                        "breadcrumb_path": r["breadcrumb_path"] or None,  # pragma: no cover
                        "category": r["category"] or None,  # pragma: no cover
                        "score": abs(float(r["score"])),  # pragma: no cover
                        "snippet": r["snippet"] or None,  # pragma: no cover
                    }  # pragma: no cover
                    for r in cursor  # pragma: no cover
                ]  # pragma: no cover
                logger.info(f"Fallback search returned {len(results)} results")  # pragma: no cover
                return results
            except Exception as e2:
                logger.error(f"Fallback search also failed: {e2}")
                return []

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
