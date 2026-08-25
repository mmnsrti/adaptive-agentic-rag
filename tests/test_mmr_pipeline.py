from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever
)



retriever = RerankedRetriever()


results = retriever.search(
    "best amazon cyber monday deals",
    top_k=5
)


for i,r in enumerate(results,1):

    print(
        i,
        r["id"],
        r["metadata"]["title"]
    )


retriever.close()