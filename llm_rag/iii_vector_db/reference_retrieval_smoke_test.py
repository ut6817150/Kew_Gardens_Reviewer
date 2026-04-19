"""Command-line smoke test for reference retrieval.

Purpose:
    This module runs one local reference-side retrieval query and prints a
    readable debug view. It is meant for quick human inspection after rebuilding
    retrieval assets or changing retrieval logic, not as a formal benchmark.
"""

from __future__ import annotations

import argparse

from retrieval_engine import answer_query


def parse_args():
    """Parse command-line options for the local retrieval smoke test.

    This helper accepts:
    - one optional free-text query
    - dense retrieval depth
    - sparse retrieval depth
    - final result count

    The defaults are tuned for quick manual inspection rather than formal
    benchmarking.
    """
    parser = argparse.ArgumentParser(description="Run the reference retrieval pipeline.")
    parser.add_argument(
        "query",
        nargs="?",
        default="What supporting information is required for a threatened species assessment?",
    )
    parser.add_argument("--dense-k", type=int, default=24)
    parser.add_argument("--sparse-k", type=int, default=24)
    parser.add_argument("--k", type=int, default=8)
    return parser.parse_args()


def main():
    """Run one reference-side retrieval query and print a readable debug view.

    This script is intended for local inspection of the retrieval layer.
    It prints:
    - the selected route
    - any threshold-lookup answer
    - the internal subqueries
    - the answer scaffold
    - the final retrieved candidates and parent context
    """
    args = parse_args()
    payload = answer_query(args.query, dense_k=args.dense_k, sparse_k=args.sparse_k, final_k=args.k)

    print("\nRoute")
    print("-----")
    print(payload["route"])

    if payload["route"] == "deterministic_threshold_lookup":
        print("\nAnswer")
        print("------")
        print(payload["threshold_answer"])
        return

    if payload.get("threshold_fallback_miss"):
        print("\nNote")
        print("----")
        print("Threshold lookup did not match exactly, so the query fell back to hybrid RAG.")

    print("\nInternal retrieval queries")
    print("------------------------")
    for q in payload["subqueries"]:
        print(f"- {q}")

    print("\nQuery")
    print("-----")
    print(payload["query"])

    print("\n" + payload["answer_scaffold"])

    print("\nResults")
    print("-------")
    for i, cand in enumerate(payload["results"], start=1):
        meta = cand.metadata
        print(f"\n--- Result {i} ---")
        print("Metadata:")
        print(
            {
                "source_file": meta.get("source_file"),
                "page": meta.get("page"),
                "block_type": meta.get("block_type"),
                "table_id": meta.get("table_id"),
                "row_id": meta.get("row_id"),
                "section_title": meta.get("section_title"),
                "table_title": meta.get("table_title"),
                "fallback": meta.get("fallback"),
                "forced_for_coverage": cand.forced_for_coverage,
                "chunk_id": meta.get("chunk_id"),
                "dense_score": round(cand.dense_score, 4),
                "bm25_score": round(cand.bm25_score, 4),
                "rerank_score": round(cand.rerank_score, 4),
            }
        )
        print("\nChild text:")
        print(cand.text[:1400])
        if cand.parent_text:
            print("\nParent context:")
            print(cand.parent_text[:1600])


if __name__ == "__main__":
    main()
