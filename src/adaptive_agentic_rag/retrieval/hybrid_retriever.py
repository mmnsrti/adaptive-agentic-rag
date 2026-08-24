from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever
)

from adaptive_agentic_rag.retrieval.bm25_retriever import (
    BM25Retriever
)

from adaptive_agentic_rag.retrieval.rrf import (
    reciprocal_rank_fusion
)



class HybridRetriever:


    def __init__(
        self,
        dense_top_k=20,
        bm25_top_k=20,
        final_top_k=20
    ):


        self.dense = DenseRetriever()

        self.bm25 = BM25Retriever()


        self.dense_top_k = dense_top_k

        self.bm25_top_k = bm25_top_k

        self.final_top_k = final_top_k





    def search(
        self,
        query,
        top_k=None
    ):


        if top_k is None:

            top_k = self.final_top_k



        #
        # Dense retrieval
        #

        dense_results = self.dense.search(
            query,
            top_k=self.dense_top_k
        )



        #
        # BM25 retrieval
        #

        bm25_results = self.bm25.search(
            query,
            top_k=self.bm25_top_k
        )




        #
        # Reciprocal Rank Fusion
        #

        fused_results = reciprocal_rank_fusion(

            [
                dense_results,
                bm25_results
            ],

            top_k=top_k

        )



        return fused_results





    def close(self):


        self.dense.close()
