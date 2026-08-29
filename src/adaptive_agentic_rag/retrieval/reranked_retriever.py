from adaptive_agentic_rag.retrieval import dense_retriever
from adaptive_agentic_rag.retrieval.hybrid_retriever import (
    HybridRetriever
)

from adaptive_agentic_rag.retrieval.reranker import (
    BGEReranker
)

from adaptive_agentic_rag.retrieval.mmr import (
    mmr_select
)

from adaptive_agentic_rag.retrieval.multi_query_retriever import (
    MultiQueryRetriever
)

class RerankedRetriever:


    def __init__(
        self,
        dense_retriever,
        hybrid_top_k=20,
        rerank_top_k=10,
        final_top_k=5,
        mmr_lambda=0.7
    ):

        self.hybrid = HybridRetriever(
            dense_retriever=dense_retriever,
            final_top_k=hybrid_top_k
        )

        self.multi_query = (
            MultiQueryRetriever(
                hybrid_retriever=self.hybrid
            )
        )

        self.reranker = BGEReranker()


        self.rerank_top_k = rerank_top_k

        self.final_top_k = final_top_k

        self.mmr_lambda = mmr_lambda



    def search(
        self,
        query,
        top_k=None
    ):

        if top_k is None:

            top_k = self.final_top_k


        #
        # The caller may request more final results
        # than the retriever's default rerank budget.
        #
        # Example:
        #
        # default rerank_top_k = 10
        # caller top_k = 20
        #
        # We must rerank at least 20 candidates,
        # otherwise MMR can never return 20 results.
        #

        rerank_candidate_k = max(
            self.rerank_top_k,
            top_k * 2
        )


        #
        # Hybrid retrieval must provide at least
        # as many candidates as the reranker needs.
        #

        hybrid_candidate_k = max(
            self.hybrid.final_top_k,
            rerank_candidate_k
        )


        # =====================================================
        # Step 1
        # Hybrid Retrieval
        # =====================================================

        #
        # Multi-query candidate generation.
        #
        # Hybrid retrieval is performed over the
        # original query plus a few deterministic
        # facets.
        #
        # Cross-encoder reranking still happens
        # only once below.
        #

        multi_query_candidate_k = max(
            hybrid_candidate_k * 2,
            40
        )


        candidates = (
            self.multi_query.search(

                query,

                top_k=(
                    multi_query_candidate_k
                )
            )
        )

        # =====================================================
        # Step 2
        # Cross Encoder Reranking
        # =====================================================

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=rerank_candidate_k
        )


        # =====================================================
        # Ensure all documents have vectors
        # =====================================================

        for item in reranked:

            if "vector" not in item:

                item["vector"] = (
                    self.hybrid
                    .dense
                    .embedder
                    .encode_documents(
                        [
                            item["text"]
                        ]
                    )[0]
                )


        # =====================================================
        # Step 3
        # Build MMR inputs
        # =====================================================

        document_embeddings = []

        valid_documents = []


        for item in reranked:

            if "vector" not in item:

                vector = (
                    self.hybrid
                    .dense
                    .embedder
                    .encode_documents(
                        [
                            item["text"]
                        ]
                    )[0]
                )

                item["vector"] = vector


            document_embeddings.append(
                item["vector"]
            )

            valid_documents.append(
                item
            )


        reranked = valid_documents


        query_embedding = (
            self.hybrid
            .dense
            .embedder
            .encode_queries(
                [query]
            )[0]
        )


        # =====================================================
        # Step 4
        # MMR diversity selection
        # =====================================================

        selected = mmr_select(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            documents=reranked,
            top_k=top_k,
            lambda_param=self.mmr_lambda
        )


        # =====================================================
        # Step 5
        # Normalize score
        # =====================================================

        for item in selected:

            item["score"] = (
                item["rerank_score"]
            )


        return selected



    def close(self):

        self.hybrid.close()