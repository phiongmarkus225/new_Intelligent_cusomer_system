"""
Database Setup & Data Ingestion Script
=======================================
Handles:
1. PostgreSQL + PgVector extension setup
2. Embedding model loading (all-MiniLM-L6-v2)
3. Text preprocessing & normalization before embedding
4. Batch processing for efficient embedding generation
5. HNSW index creation for optimized retrieval
6. Data validation & verification

Run:
    python setup_db.py
"""

import os
import re
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

COLLECTION_NAME = "customer_knowledge_base"
BATCH_SIZE = 10  # Process embeddings in batches of 10


def preprocess_text(text_val: str) -> str:
    """
    Normalize text before embedding:
    - Strip whitespace
    - Collapse multiple spaces
    - Remove non-printable characters
    """
    if not isinstance(text_val, str):
        return ""
    text_val = text_val.strip()
    text_val = re.sub(r"\s+", " ", text_val)
    text_val = re.sub(r"[^\w\s\?\!\.\,\-\/\@\'\:\;\(\)]", "", text_val)
    return text_val


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the knowledge base data."""
    print(f"\n Validating data ({len(df)} rows)...")

    # Check required columns
    required = ["category", "tag", "question", "answer"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        print(f"   Dropped {before - len(df)} rows with missing values")

    # Preprocess text fields
    for col in ["question", "answer"]:
        df[col] = df[col].apply(preprocess_text)

    # Strip category and tag
    df["category"] = df["category"].str.strip()
    df["tag"] = df["tag"].str.strip()

    # Validate data quality
    print(f"   Categories: {df['category'].nunique()} unique  {df['category'].unique().tolist()}")
    print(f"   Tags: {df['tag'].nunique()} unique  {df['tag'].unique().tolist()}")
    print(f"   Total valid entries: {len(df)}")

    # Check for duplicates
    dupes = df.duplicated(subset=["question"], keep="first").sum()
    if dupes > 0:
        print(f"   Found {dupes} duplicate questions, removing...")
        df = df.drop_duplicates(subset=["question"], keep="first")

    return df


def setup_database():
    """Create the PgVector extension in PostgreSQL."""
    print("\n Setting up PostgreSQL database...")
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Create extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
        print("   PgVector extension enabled")

        # Verify extension
        result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector';"))
        version = result.fetchone()
        if version:
            print(f"   PgVector version: {version[0]}")

    return engine


def create_embeddings_batch(
    vector_store: PGVector,
    documents: list,
    metadatas: list,
    ids: list,
    batch_size: int = BATCH_SIZE,
):
    """
    Add documents to vector store in batches for efficiency.
    This avoids memory issues with large datasets.
    """
    total = len(documents)
    total_batches = (total + batch_size - 1) // batch_size

    print(f"\n Processing {total} documents in {total_batches} batches (batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        batch_num = (i // batch_size) + 1
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        start = time.time()
        vector_store.add_texts(
            texts=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        elapsed = time.time() - start

        print(f"   Batch {batch_num}/{total_batches} ({len(batch_docs)} docs)  {elapsed:.2f}s")


def create_hnsw_index(engine, collection_name: str):
    """
    Create HNSW index on the embedding column for fast retrieval.

    HNSW (Hierarchical Navigable Small World) is chosen because:
    - Faster query time than IVFFlat for our dataset size
    - No need to call VACUUM for index training
    - Good recall at reasonable memory cost
    - Supports multiple distance metrics (cosine, L2, inner product)
    """
    print("\n Creating HNSW index...")

    with engine.connect() as conn:
        # Find collection UUID
        res = conn.execute(
            text("SELECT uuid FROM langchain_pg_collection WHERE name = :name"),
            {"name": collection_name},
        )
        collection_row = res.fetchone()
        if not collection_row:
            print("  ️ Collection not found, skipping index creation")
            return

        collection_id = collection_row[0]

        # LangChain PGVector creates the embedding column as VECTOR (no dimensions).
        # HNSW indexes require a fixed-dimension column, so we ALTER it first.
        # all-MiniLM-L6-v2 produces 384-dimensional vectors.
        print("   Setting embedding column to VECTOR(384)...")
        try:
            conn.execute(text(
                "ALTER TABLE langchain_pg_embedding "
                "ALTER COLUMN embedding TYPE vector(384);"
            ))
            conn.commit()
            print("   Embedding column set to VECTOR(384)")
        except Exception as e:
            conn.rollback()
            # Column may already have dimensions, that's OK
            print(f"  [INFO]️ Column already typed or skipped: {str(e)[:80]}")

        # Drop existing indexes if any (to recreate with correct params)
        conn.execute(text("DROP INDEX IF EXISTS knowledge_base_hnsw_cosine_idx;"))
        conn.execute(text("DROP INDEX IF EXISTS knowledge_base_hnsw_l2_idx;"))
        conn.execute(text("DROP INDEX IF EXISTS knowledge_base_hnsw_ip_idx;"))

        # Create HNSW index for cosine similarity (primary)
        conn.execute(text("""
            CREATE INDEX knowledge_base_hnsw_cosine_idx
            ON langchain_pg_embedding
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        print("   HNSW Cosine index created (m=16, ef_construction=64)")

        # Create HNSW index for L2 distance (for benchmarking)
        conn.execute(text("""
            CREATE INDEX knowledge_base_hnsw_l2_idx
            ON langchain_pg_embedding
            USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        print("   HNSW L2 index created")

        # Create HNSW index for Inner Product (for benchmarking)
        conn.execute(text("""
            CREATE INDEX knowledge_base_hnsw_ip_idx
            ON langchain_pg_embedding
            USING hnsw (embedding vector_ip_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        print("   HNSW Inner Product index created")

        conn.commit()

    print("   All HNSW indexes applied successfully")


def verify_ingestion(engine, collection_name: str):
    """Verify that data was properly ingested."""
    print("\n Verifying data ingestion...")

    with engine.connect() as conn:
        # Count documents
        res = conn.execute(text("""
            SELECT COUNT(*) FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :name
        """), {"name": collection_name})
        count = res.fetchone()[0]
        print(f"   Total documents in vector store: {count}")

        # Check embedding dimensions
        res = conn.execute(text("""
            SELECT vector_dims(embedding) FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :name
            LIMIT 1
        """), {"name": collection_name})
        dims = res.fetchone()
        if dims:
            print(f"   Embedding dimensions: {dims[0]}")

        # Check categories distribution
        res = conn.execute(text("""
            SELECT e.cmetadata->>'category' as category, COUNT(*) as cnt
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :name
            GROUP BY e.cmetadata->>'category'
            ORDER BY cnt DESC
        """), {"name": collection_name})
        print("   Category distribution:")
        for row in res:
            print(f"      {row[0]}: {row[1]} entries")

        # Check indexes
        res = conn.execute(text("""
            SELECT indexname, indexdef FROM pg_indexes
            WHERE tablename = 'langchain_pg_embedding'
            AND indexname LIKE '%hnsw%'
        """))
        indexes = res.fetchall()
        print(f"   HNSW indexes: {len(indexes)}")
        for idx in indexes:
            print(f"      {idx[0]}")


def main():
    print("=" * 60)
    print(" Intelligent Customer Assistant  Database Setup")
    print("=" * 60)

    # 1. Setup database
    engine = setup_database()

    # 2. Load embedding model
    print("\n Loading embedding model (all-MiniLM-L6-v2)...")
    start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"   Model loaded in {time.time() - start:.2f}s")

    # 3. Initialize vector store
    print("\n Initializing PGVector store...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    print("   Vector store ready")

    # 4. Load and validate data
    print("\n Loading knowledge base data...")
    csv_path = "data/knowledge_base.csv"
    if not os.path.exists(csv_path):
        print(f"   File not found: {csv_path}")
        print("  Run 'python generate_data.py' first!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = validate_data(df)

    # 5. Prepare documents for embedding
    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        # Combine question + answer for richer semantic representation
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        content = preprocess_text(content)

        metadata = {
            "category": row["category"],
            "tag": row["tag"],
            "question": row["question"],
            "answer": row["answer"],
        }

        documents.append(content)
        metadatas.append(metadata)
        ids.append(str(i))

    # 6. Batch embed and ingest
    create_embeddings_batch(vector_store, documents, metadatas, ids)

    # 7. Create HNSW indexes
    create_hnsw_index(engine, COLLECTION_NAME)

    # 8. Verify
    verify_ingestion(engine, COLLECTION_NAME)

    print("\n" + "=" * 60)
    print(" Database setup complete! You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
