from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever
)



retriever = RerankedRetriever()


query = (
    "Amazon Cyber Monday deals"
)



results = retriever.search(
    query,
    top_k=5
)



for rank, item in enumerate(
    results,
    start=1
):

    print(
        rank,
        item["id"],
        item["score"],
        item["metadata"]["title"]
    )



retriever.close()