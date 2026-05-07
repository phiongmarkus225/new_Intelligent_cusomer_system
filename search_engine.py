"""
Semantic Search Engine Module
=============================
Implements:
- Query preprocessing & normalization
- Similarity search with multiple distance metrics (cosine, L2, inner product)
- Exact match search (keyword-based)
- Hybrid search combining semantic + exact match
- Category/metadata filtering
- Relevance scoring & ranking
- Search analytics & logging
"""

import os
import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine, text

load_dotenv()

def _get_database_url():
    """Get DATABASE_URL from Streamlit secrets or .env."""
    try:
        import streamlit as st
        return st.secrets["DATABASE_URL"]
    except Exception:
        return os.getenv("DATABASE_URL")

DATABASE_URL = _get_database_url()

# ──────────────────────────────────────────────────────────
# Logging setup for search analytics
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    filename="search_analytics.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("search_engine")


# ──────────────────────────────────────────────────────────
# Text preprocessing & normalization
# ──────────────────────────────────────────────────────────
def preprocess_query(query: str) -> str:
    """
    Normalize user query:
    - Lowercase
    - Strip extra whitespace
    - Remove special characters (keep alphanumeric, spaces, and Indonesian chars)
    """
    query = query.lower().strip()
    query = re.sub(r"\s+", " ", query)
    # Keep alphanumeric + common Indonesian characters
    query = re.sub(r"[^\w\s\?\!\.\,\-]", "", query)
    return query


def preprocess_text_for_embedding(text: str) -> str:
    """
    Normalize text before embedding generation:
    - Lowercase
    - Remove excessive whitespace
    - Normalize punctuation
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\?\!\.\,\-\/\@]", "", text)
    return text


# ──────────────────────────────────────────────────────────
# Core Search Engine Class
# ──────────────────────────────────────────────────────────
class SemanticSearchEngine:
    """
    Full-featured search engine combining semantic search (PGVector)
    with exact match and metadata filtering.
    """

    # Similarity threshold: documents below this score are considered irrelevant
    SIMILARITY_THRESHOLD = 0.3

    def __init__(self, collection_name: str = "customer_knowledge_base"):
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
        self.engine = create_engine(DATABASE_URL)
        self._search_history: List[Dict] = []

    # ── Semantic Similarity Search ───────────────────────

    def semantic_search(
        self,
        query: str,
        k: int = 5,
        category_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Perform semantic similarity search using cosine distance.
        Returns ranked results with relevance scores.
        """
        start_time = time.time()
        processed_query = preprocess_query(query)

        # Build metadata filter
        filter_dict = {}
        if category_filter:
            filter_dict["category"] = category_filter
        if tag_filter:
            filter_dict["tag"] = tag_filter

        # Perform similarity search with scores
        if filter_dict:
            results = self.vector_store.similarity_search_with_relevance_scores(
                processed_query, k=k, filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_relevance_scores(
                processed_query, k=k
            )

        elapsed = time.time() - start_time

        # Parse results with ranking
        ranked_results = []
        for rank, (doc, score) in enumerate(results, 1):
            if score >= self.SIMILARITY_THRESHOLD:
                ranked_results.append(
                    {
                        "rank": rank,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": round(score, 4),
                        "category": doc.metadata.get("category", "Unknown"),
                        "tag": doc.metadata.get("tag", "Unknown"),
                        "question": doc.metadata.get("question", ""),
                        "answer": doc.metadata.get("answer", ""),
                    }
                )

        # Log search analytics
        self._log_search(
            query=query,
            method="semantic",
            results_count=len(ranked_results),
            elapsed_time=elapsed,
            filter_category=category_filter,
            filter_tag=tag_filter,
        )

        return ranked_results

    # ── Exact Match (Keyword) Search ─────────────────────

    def exact_match_search(
        self,
        query: str,
        category_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Perform keyword-based exact match search on stored documents.
        Uses SQL ILIKE for case-insensitive pattern matching.
        """
        start_time = time.time()
        processed_query = preprocess_query(query)

        # Build SQL query for exact match
        sql = """
            SELECT e.document, e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection
            AND LOWER(e.document) LIKE :pattern
        """
        params = {
            "collection": self.collection_name,
            "pattern": f"%{processed_query}%",
        }

        if category_filter:
            sql += " AND e.cmetadata->>'category' = :category"
            params["category"] = category_filter

        sql += " LIMIT 10"

        results = []
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
            for rank, row in enumerate(rows, 1):
                metadata = row[1] if row[1] else {}
                results.append(
                    {
                        "rank": rank,
                        "content": row[0],
                        "metadata": metadata,
                        "relevance_score": 1.0,  # Exact match = perfect score
                        "category": metadata.get("category", "Unknown"),
                        "tag": metadata.get("tag", "Unknown"),
                        "question": metadata.get("question", ""),
                        "answer": metadata.get("answer", ""),
                    }
                )

        elapsed = time.time() - start_time
        self._log_search(
            query=query,
            method="exact_match",
            results_count=len(results),
            elapsed_time=elapsed,
            filter_category=category_filter,
        )
        return results

    # ── Hybrid Search (Semantic + Exact Match) ───────────

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        category_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        semantic_weight: float = 0.7,
        exact_weight: float = 0.3,
    ) -> List[Dict]:
        """
        Combine semantic search and exact match search.
        Uses weighted scoring to merge and de-duplicate results.
        """
        start_time = time.time()

        # Run both search methods
        semantic_results = self.semantic_search(query, k=k, category_filter=category_filter, tag_filter=tag_filter)
        exact_results = self.exact_match_search(query, category_filter=category_filter)

        # Merge and de-duplicate by question content
        seen_questions = set()
        merged = []

        for res in semantic_results:
            q = res.get("question", "")
            if q not in seen_questions:
                seen_questions.add(q)
                res["final_score"] = round(res["relevance_score"] * semantic_weight, 4)
                res["source"] = "semantic"
                merged.append(res)

        for res in exact_results:
            q = res.get("question", "")
            if q not in seen_questions:
                seen_questions.add(q)
                res["final_score"] = round(res["relevance_score"] * exact_weight, 4)
                res["source"] = "exact_match"
                merged.append(res)
            else:
                # Boost score if found in both
                for m in merged:
                    if m.get("question") == q:
                        m["final_score"] = round(
                            m["final_score"] + (res["relevance_score"] * exact_weight), 4
                        )
                        m["source"] = "hybrid"
                        break

        # Re-rank by final score
        merged.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # Assign new ranks
        for i, item in enumerate(merged[:k], 1):
            item["rank"] = i

        elapsed = time.time() - start_time
        self._log_search(
            query=query,
            method="hybrid",
            results_count=len(merged[:k]),
            elapsed_time=elapsed,
            filter_category=category_filter,
            filter_tag=tag_filter,
        )

        return merged[:k]

    # ── Search with Multiple Distance Metrics ────────────

    def search_with_distance_metrics(
        self, query: str, k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Run search with different distance metrics for benchmarking:
        - Cosine similarity (default)
        - L2 (Euclidean) distance
        - Inner product

        Returns dict with metric name as key and results list as value.
        """
        processed_query = preprocess_query(query)
        query_embedding = self.embeddings.embed_query(processed_query)

        results = {}
        metrics = {
            "cosine": ("vector_cosine_ops", "<=>"),
            "l2_euclidean": ("vector_l2_ops", "<->"),
            "inner_product": ("vector_ip_ops", "<#>"),
        }

        for metric_name, (ops_name, operator) in metrics.items():
            start_time = time.time()

            sql = f"""
                SELECT e.document, e.cmetadata,
                       e.embedding {operator} :embedding AS distance
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :collection
                ORDER BY e.embedding {operator} :embedding
                LIMIT :k
            """

            try:
                with self.engine.connect() as conn:
                    rows = conn.execute(
                        text(sql),
                        {
                            "embedding": str(query_embedding),
                            "collection": self.collection_name,
                            "k": k,
                        },
                    ).fetchall()

                elapsed = time.time() - start_time
                metric_results = []
                for rank, row in enumerate(rows, 1):
                    metadata = row[1] if row[1] else {}
                    distance = float(row[2]) if row[2] else 0.0

                    # Convert distance to similarity score
                    if metric_name == "cosine":
                        similarity = 1 - distance
                    elif metric_name == "l2_euclidean":
                        similarity = 1 / (1 + distance)
                    else:  # inner_product
                        similarity = -distance

                    metric_results.append(
                        {
                            "rank": rank,
                            "content": row[0],
                            "metadata": metadata,
                            "distance": round(distance, 6),
                            "similarity": round(similarity, 4),
                            "question": metadata.get("question", ""),
                            "answer": metadata.get("answer", ""),
                            "elapsed_ms": round(elapsed * 1000, 2),
                        }
                    )

                results[metric_name] = metric_results
            except Exception as e:
                logger.warning(f"Distance metric '{metric_name}' failed: {e}")
                results[metric_name] = []

        return results

    # ── Analytics & Logging ──────────────────────────────

    def _log_search(
        self,
        query: str,
        method: str,
        results_count: int,
        elapsed_time: float,
        filter_category: Optional[str] = None,
        filter_tag: Optional[str] = None,
    ):
        """Log search event for analytics."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "method": method,
            "results_count": results_count,
            "elapsed_ms": round(elapsed_time * 1000, 2),
            "filter_category": filter_category,
            "filter_tag": filter_tag,
        }
        self._search_history.append(entry)
        logger.info(
            f"SEARCH | method={method} | query='{query[:80]}' | "
            f"results={results_count} | time={entry['elapsed_ms']}ms | "
            f"category={filter_category} | tag={filter_tag}"
        )

    def get_search_analytics(self) -> pd.DataFrame:
        """Return search history as a DataFrame for analysis."""
        if not self._search_history:
            return pd.DataFrame()
        return pd.DataFrame(self._search_history)

    def get_available_categories(self) -> List[str]:
        """Get all unique categories from the knowledge base."""
        sql = """
            SELECT DISTINCT e.cmetadata->>'category' AS category
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection
            AND e.cmetadata->>'category' IS NOT NULL
            ORDER BY category
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"collection": self.collection_name}).fetchall()
        return [row[0] for row in rows if row[0]]

    def get_available_tags(self, category: Optional[str] = None) -> List[str]:
        """Get all unique tags, optionally filtered by category."""
        sql = """
            SELECT DISTINCT e.cmetadata->>'tag' AS tag
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection
            AND e.cmetadata->>'tag' IS NOT NULL
        """
        params = {"collection": self.collection_name}
        if category:
            sql += " AND e.cmetadata->>'category' = :category"
            params["category"] = category
        sql += " ORDER BY tag"

        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return [row[0] for row in rows if row[0]]
