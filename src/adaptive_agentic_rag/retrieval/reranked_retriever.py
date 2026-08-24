from adaptive_agentic_rag.retrieval.hybrid_retriever import (
    HybridRetriever
)

from adaptive_agentic_rag.retrieval.reranker import (
    BGEReranker
)



class RerankedRetriever:


    def __init__(
        self,
        hybrid_top_k=20,
        final_top_k=5
    ):


        self.hybrid = HybridRetriever(
            final_top_k=hybrid_top_k
        )


        self.reranker = BGEReranker()


        self.final_top_k = final_top_k



    def search(
        self,
        query,
        top_k=None
    ):


        if top_k is None:

            top_k = self.final_top_k



        #
        # Step 1
        # Hybrid Retrieval
        #

        candidates = (
            self.hybrid.search(
                query,
                top_k=20
            )
        )



        #
        # Step 2
        # Cross Encoder Reranking
        #

        reranked = (
            self.reranker.rerank(
                query,
                candidates,
                top_k=top_k
            )
        )



        #
        # Normalize score
        #

        for item in reranked:


            item["score"] = (
                item["rerank_score"]
            )


        return reranked



    def close(self):

        self.hybrid.close()