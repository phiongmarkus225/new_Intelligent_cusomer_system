"""
Benchmarking Script for Search Performance
===========================================
Compares:
- Cosine similarity vs L2 distance vs Inner Product
- Semantic search vs Exact match vs Hybrid
- With and without metadata filters
- Measures latency, result quality, and recall

Run:
    python benchmark.py
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from search_engine import SemanticSearchEngine

load_dotenv()


# ── Test Queries ──────────────────────────────────────────
TEST_QUERIES = [
    {"query": "Bagaimana cara mereset password?", "expected_category": "FAQ", "expected_tag": "Account"},
    {"query": "Kebijakan pengembalian barang", "expected_category": "Policy", "expected_tag": "Return"},
    {"query": "Aplikasi sering crash", "expected_category": "Troubleshooting", "expected_tag": "App"},
    {"query": "Cara menghubungi customer service", "expected_category": "Contact Information", "expected_tag": "CS"},
    {"query": "Metode pembayaran yang tersedia", "expected_category": "FAQ", "expected_tag": "Payment"},
    {"query": "Proses refund berapa lama?", "expected_category": "Policy", "expected_tag": "Refund"},
    {"query": "Gagal bayar kartu kredit", "expected_category": "Troubleshooting", "expected_tag": "Payment"},
    {"query": "Alamat kantor pusat", "expected_category": "Contact Information", "expected_tag": "Office"},
    {"query": "Garansi produk tidak berlaku", "expected_category": "Policy", "expected_tag": "Warranty"},
    {"query": "Tidak bisa login akun", "expected_category": "Troubleshooting", "expected_tag": "Login"},
]


def benchmark_distance_metrics(engine: SemanticSearchEngine):
    """Compare performance across distance metrics."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Distance Metrics Comparison")
    print("=" * 70)

    results = []
    for tq in TEST_QUERIES:
        query = tq["query"]
        metrics_results = engine.search_with_distance_metrics(query, k=3)

        for metric_name, search_results in metrics_results.items():
            if search_results:
                top_result = search_results[0]
                results.append({
                    "query": query[:40],
                    "metric": metric_name,
                    "top_answer": top_result["question"][:50] if top_result["question"] else "N/A",
                    "similarity": top_result.get("similarity", 0),
                    "latency_ms": top_result.get("elapsed_ms", 0),
                })

    df = pd.DataFrame(results)
    print("\n--- Results by Distance Metric ---")
    print(df.to_string(index=False))

    # Summary statistics
    print("\n--- Average Latency by Metric ---")
    summary = df.groupby("metric").agg(
        avg_latency_ms=("latency_ms", "mean"),
        avg_similarity=("similarity", "mean"),
    ).round(3)
    print(summary.to_string())

    return df


def benchmark_search_methods(engine: SemanticSearchEngine):
    """Compare semantic, exact match, and hybrid search."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Search Methods Comparison")
    print("=" * 70)

    results = []
    for tq in TEST_QUERIES:
        query = tq["query"]
        expected_cat = tq["expected_category"]

        # Semantic Search
        t0 = time.time()
        sem_results = engine.semantic_search(query, k=3)
        sem_time = (time.time() - t0) * 1000

        # Exact Match Search
        t0 = time.time()
        exact_results = engine.exact_match_search(query)
        exact_time = (time.time() - t0) * 1000

        # Hybrid Search
        t0 = time.time()
        hybrid_results = engine.hybrid_search(query, k=3)
        hybrid_time = (time.time() - t0) * 1000

        # Check if expected category is in top result
        sem_cat = sem_results[0]["category"] if sem_results else "N/A"
        exact_cat = exact_results[0]["category"] if exact_results else "N/A"
        hybrid_cat = hybrid_results[0]["category"] if hybrid_results else "N/A"

        results.append({
            "query": query[:40],
            "expected_category": expected_cat,
            "semantic_results": len(sem_results),
            "semantic_ms": round(sem_time, 2),
            "semantic_cat_match": sem_cat == expected_cat,
            "exact_results": len(exact_results),
            "exact_ms": round(exact_time, 2),
            "exact_cat_match": exact_cat == expected_cat,
            "hybrid_results": len(hybrid_results),
            "hybrid_ms": round(hybrid_time, 2),
            "hybrid_cat_match": hybrid_cat == expected_cat,
        })

    df = pd.DataFrame(results)
    print("\n--- Method Comparison ---")
    print(df[["query", "semantic_ms", "exact_ms", "hybrid_ms",
              "semantic_cat_match", "exact_cat_match", "hybrid_cat_match"]].to_string(index=False))

    # Summary
    print("\n--- Summary ---")
    print(f"Semantic Avg Latency:    {df['semantic_ms'].mean():.2f} ms")
    print(f"Exact Match Avg Latency: {df['exact_ms'].mean():.2f} ms")
    print(f"Hybrid Avg Latency:      {df['hybrid_ms'].mean():.2f} ms")
    print(f"Semantic Category Match:  {df['semantic_cat_match'].sum()}/{len(df)}")
    print(f"Exact Match Category Match: {df['exact_cat_match'].sum()}/{len(df)}")
    print(f"Hybrid Category Match:    {df['hybrid_cat_match'].sum()}/{len(df)}")

    return df


def benchmark_filtered_search(engine: SemanticSearchEngine):
    """Compare unfiltered vs filtered search performance."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Filtered vs Unfiltered Search")
    print("=" * 70)

    results = []
    for tq in TEST_QUERIES:
        query = tq["query"]
        expected_cat = tq["expected_category"]

        # Unfiltered
        t0 = time.time()
        unfiltered = engine.semantic_search(query, k=3)
        unfiltered_time = (time.time() - t0) * 1000

        # Filtered by category
        t0 = time.time()
        filtered = engine.semantic_search(query, k=3, category_filter=expected_cat)
        filtered_time = (time.time() - t0) * 1000

        results.append({
            "query": query[:40],
            "category_filter": expected_cat,
            "unfiltered_results": len(unfiltered),
            "unfiltered_ms": round(unfiltered_time, 2),
            "unfiltered_top_score": unfiltered[0]["relevance_score"] if unfiltered else 0,
            "filtered_results": len(filtered),
            "filtered_ms": round(filtered_time, 2),
            "filtered_top_score": filtered[0]["relevance_score"] if filtered else 0,
        })

    df = pd.DataFrame(results)
    print("\n--- Filtered vs Unfiltered ---")
    print(df.to_string(index=False))

    print("\n--- Summary ---")
    print(f"Unfiltered Avg Latency: {df['unfiltered_ms'].mean():.2f} ms")
    print(f"Filtered Avg Latency:   {df['filtered_ms'].mean():.2f} ms")
    print(f"Unfiltered Avg Top Score: {df['unfiltered_top_score'].mean():.4f}")
    print(f"Filtered Avg Top Score:   {df['filtered_top_score'].mean():.4f}")

    return df


def main():
    print("Initializing Search Engine for Benchmarking...")
    engine = SemanticSearchEngine()

    print(f"\nAvailable categories: {engine.get_available_categories()}")
    print(f"Available tags: {engine.get_available_tags()}")

    df1 = benchmark_distance_metrics(engine)
    df2 = benchmark_search_methods(engine)
    df3 = benchmark_filtered_search(engine)

    # Save results
    os.makedirs("benchmark_results", exist_ok=True)
    df1.to_csv("benchmark_results/distance_metrics.csv", index=False)
    df2.to_csv("benchmark_results/search_methods.csv", index=False)
    df3.to_csv("benchmark_results/filtered_search.csv", index=False)

    # Search analytics
    analytics = engine.get_search_analytics()
    if not analytics.empty:
        analytics.to_csv("benchmark_results/search_analytics.csv", index=False)

    print("\n" + "=" * 70)
    print("All benchmark results saved to benchmark_results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
