from typing import Any, Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from psycopg2.extensions import register_adapter, AsIs

# Define how psycopg2 should handle a numpy.ndarray
def adapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))  # Converts the array to a standard SQL tuple

# Register the adapter
register_adapter(np.ndarray, adapt_numpy_array)

# Try importing pgvector adapter if available
try:
    from pgvector.psycopg2 import register_vector
    PGVECTOR_LIB_AVAILABLE = True
except ImportError:
    PGVECTOR_LIB_AVAILABLE = False


class VectorDatabase:
    """
    PostgreSQL vector database handler tailored for local PostgreSQL instances.
    Automatically detects if native `pgvector` extension is installed on the local server;
    if not available, falls back to standard PostgreSQL array storage (REAL[]) with cosine similarity.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        sslmode: str = "disable",
    ):
        self.conn_params = {
            "host": host or "localhost",
            "port": int(port) if port else 5432,
            "dbname": database or "postgres",
            "user": user or "postgres",
            "password": password or "",
            "sslmode": sslmode or "disable",
        }
        self.has_pgvector: Optional[bool] = None

    def get_connection(self):
        """Create and return a new PostgreSQL connection."""
        conn = psycopg2.connect(**self.conn_params)
        if self.has_pgvector and PGVECTOR_LIB_AVAILABLE:
            try:
                register_vector(conn)
            except Exception:
                pass
        return conn

    def check_extension_support(self, conn) -> bool:
        """Check if pgvector extension can be created or already exists on the local server."""
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                self.has_pgvector = True
                if PGVECTOR_LIB_AVAILABLE:
                    register_vector(conn)
                return True
        except Exception:
            conn.rollback()
            self.has_pgvector = False
            return False

    def test_connection(self) -> Tuple[bool, str]:
        """Test local PostgreSQL database connection and detect vector support."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version_info = cur.fetchone()[0]
                has_vector = self.check_extension_support(conn)
                vector_status = (
                    "Native pgvector extension is AVAILABLE & ACTIVE"
                    if has_vector
                    else "Standard Float Array mode ACTIVE (pgvector extension not found on local DB, using built-in array vector mode)"
                )
            return True, f"Connected to local PostgreSQL successfully!\nVersion: {version_info.split(',')[0]}\nMode: {vector_status}"
        except Exception as e:
            return False, f"Connection to local PostgreSQL failed: {str(e)}"

    def init_db(self, dimension: int = 768) -> None:
        """
        Initialize the local PostgreSQL tables for storing embeddings.
        Supports both pgvector and standard array fallback.
        Stores both URLs and documents in the same table.
        """
        with self.get_connection() as conn:
            has_vector = self.check_extension_support(conn)
            with conn.cursor() as cur:
                if has_vector:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS url_embeddings (
                            id SERIAL PRIMARY KEY,
                            url TEXT NOT NULL,
                            title TEXT,
                            source_type VARCHAR(50) DEFAULT 'url',
                            chunk_index INTEGER NOT NULL,
                            chunk_text TEXT NOT NULL,
                            embedding vector({dimension}) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    #cur.execute(
                    #"""
                     #   CREATE INDEX IF NOT EXISTS url_embeddings_cosine_idx 
                       # ON url_embeddings USING hnsw (embedding vector_cosine_ops);
                       # """
                    #)
                else:
                    # Fallback for standard local PostgreSQL without pgvector compiled
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS url_embeddings (
                            id SERIAL PRIMARY KEY,
                            url TEXT NOT NULL,
                            title TEXT,
                            source_type VARCHAR(50) DEFAULT 'url',
                            chunk_index INTEGER NOT NULL,
                            chunk_text TEXT NOT NULL,
                            embedding REAL[] NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    #cur.execute(
                       # """
                       # CREATE INDEX IF NOT EXISTS url_embeddings_url_idx 
                       # ON url_embeddings (url);
                       # """
                    #)
            conn.commit()

    def store_url_chunks(
        self,
        url: str,
        title: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store chunks and their embeddings for a given URL in the local PostgreSQL DB.
        """
        return self._store_chunks(url, title, chunks, embeddings, source_type='url')

    def store_document_chunks(
        self,
        document_id: str,
        title: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store chunks and their embeddings for a document in the local PostgreSQL DB.
        document_id: unique identifier for the document (e.g., filename or path)
        """
        return self._store_chunks(document_id, title, chunks, embeddings, source_type='document')

    def _store_chunks(
        self,
        source_id: str,
        title: str,
        chunks: List[str],
        embeddings: List[List[float]],
        source_type: str = 'url',
    ) -> int:
        """
        Internal method to store chunks and embeddings for any source type.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")

        with self.get_connection() as conn:
            has_vector = self.has_pgvector if self.has_pgvector is not None else self.check_extension_support(conn)
            with conn.cursor() as cur:
                # Remove existing chunks for this source to prevent duplicate data
                cur.execute("DELETE FROM url_embeddings WHERE url = %s AND source_type = %s;", (source_id, source_type))

                records = []
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    if has_vector and PGVECTOR_LIB_AVAILABLE:
                        vec_data = np.array(emb, dtype=np.float32)
                    else:
                        vec_data = [float(x) for x in emb]
                    records.append((source_id, title, source_type, idx, chunk, vec_data))

                query = """
                    INSERT INTO url_embeddings (url, title, source_type, chunk_index, chunk_text, embedding)
                    VALUES %s
                """
                execute_values(
                    cur,
                    query,
                    records,
                    template="(%s, %s, %s, %s, %s, %s)",
                )
            conn.commit()
        return len(chunks)

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        url_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search against stored embeddings.
        """
        with self.get_connection() as conn:
            has_vector = self.has_pgvector if self.has_pgvector is not None else self.check_extension_support(conn)
            with conn.cursor() as cur:
                if has_vector:
                    query_vec = np.array(query_embedding, dtype=np.float32)
                    #query_vec = arr.list()
                    if url_filter:
                        cur.execute(
                            """
                            SELECT 
                                id, 
                                url, 
                                title, 
                                chunk_index, 
                                chunk_text, 
                                1 - (embedding <=> %s) AS similarity
                            FROM url_embeddings
                            WHERE url = %s
                            ORDER BY embedding <=> %s ASC
                            LIMIT %s;
                            """,
                            (query_vec, url_filter, query_vec, top_k),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT 
                                id, 
                                url, 
                                title, 
                                chunk_index, 
                                chunk_text, 
                                1 - (embedding <=> %s) AS similarity
                            FROM url_embeddings
                            ORDER BY embedding <=> %s ASC
                            LIMIT %s;
                            """,
                            (query_vec, query_vec, top_k),
                        )
                    rows = cur.fetchall()
                    return [
                        {
                            "id": row[0],
                            "url": row[1],
                            "title": row[2],
                            "chunk_index": row[3],
                            "chunk_text": row[4],
                            "similarity": float(row[5]),
                        }
                        for row in rows
                    ]
                else:
                    # Array mode calculation via python/numpy
                    if url_filter:
                        cur.execute(
                            """
                            SELECT id, url, title, chunk_index, chunk_text, embedding
                            FROM url_embeddings
                            WHERE url = %s;
                            """,
                            (url_filter,),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, url, title, chunk_index, chunk_text, embedding
                            FROM url_embeddings;
                            """
                        )
                    rows = cur.fetchall()
                    if not rows:
                        return []

                    q_emb = np.array(query_embedding, dtype=np.float32)
                    q_norm = np.linalg.norm(q_emb)

                    scored_results = []
                    for row in rows:
                        row_emb = np.array(row[5], dtype=np.float32)
                        dot_prod = np.dot(q_emb, row_emb)
                        denom = q_norm * np.linalg.norm(row_emb)
                        sim = float(dot_prod / denom) if denom != 0 else 0.0
                        scored_results.append({
                            "id": row[0],
                            "url": row[1],
                            "title": row[2],
                            "chunk_index": row[3],
                            "chunk_text": row[4],
                            "similarity": sim,
                        })

                    scored_results.sort(key=lambda x: x["similarity"], reverse=True)
                    return scored_results[:top_k]

    def list_indexed_urls(self) -> List[Dict[str, Any]]:
        """List all indexed URLs from local PostgreSQL with chunk counts and date added."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        url, 
                        title, 
                        COUNT(*) as chunk_count, 
                        MAX(created_at) as last_indexed
                    FROM url_embeddings
                    GROUP BY url, title
                    ORDER BY last_indexed DESC;
                    """
                )
                rows = cur.fetchall()
                return [
                    {
                        "url": row[0],
                        "title": row[1] or row[0],
                        "chunk_count": row[2],
                        "last_indexed": row[3].strftime("%Y-%m-%d %H:%M:%S") if row[3] else "N/A",
                    }
                    for row in rows
                ]

    def delete_url(self, url: str) -> int:
        """Delete all chunks for a specific URL from local PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM url_embeddings WHERE url = %s;", (url,))
                count = cur.rowcount
            conn.commit()
            return count

    def clear_all(self) -> None:
        """Clear all stored embeddings in local PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE url_embeddings;")
            conn.commit()
