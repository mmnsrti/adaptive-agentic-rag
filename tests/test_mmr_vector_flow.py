from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever
)


retriever = RerankedRetriever(
    hybrid_top_k=20,
    rerank_top_k=10,
    final_top_k=5
)


results = retriever.search(
    "amazon black friday deals"
)


print("\n===== FINAL MMR OUTPUT =====")


for item in results:

    print(
        item["id"],
        "vector:",
        "vector" in item,
        "score:",
        item["score"]
    )


retriever.close()