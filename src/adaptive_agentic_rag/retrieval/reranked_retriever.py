from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from adaptive_agentic_rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from adaptive_agentic_rag.retrieval.reranker import (
    BGEReranker,
)

from adaptive_agentic_rag.retrieval.mmr import (
    mmr_select,
)

from adaptive_agentic_rag.retrieval.multi_query_retriever import (
    MultiQueryRetriever,
)


DEFAULT_BM25_CORPUS_PATH = (
    "data/processed/processed_corpus.json"
)


class RerankedRetriever:

    def __init__(
        self,
        dense_retriever=None,
        hybrid_top_k: int = 20,
        rerank_top_k: int = 10,
        final_top_k: int = 5,
        mmr_lambda: float = 0.7,
        bm25_corpus_path: str = DEFAULT_BM25_CORPUS_PATH,
    ):

        # ====================================================
        # Backward-compatible Dense ownership
        #
        # Preferred production usage:
        #
        # RerankedRetriever(
        #     dense_retriever=shared_dense
        # )
        #
        # Standalone usage remains supported:
        #
        # RerankedRetriever()
        # ====================================================

        if dense_retriever is None:

            dense_retriever = (
                DenseRetriever()
            )


        self.hybrid = (
            HybridRetriever(
                dense_retriever=
                    dense_retriever,
                final_top_k=
                    hybrid_top_k,
                bm25_corpus_path=
                    bm25_corpus_path,
            )
        )


        self.multi_query = (
            MultiQueryRetriever(
                hybrid_retriever=
                    self.hybrid
            )
        )


        self.reranker = (
            BGEReranker()
        )


        self.rerank_top_k = (
            rerank_top_k
        )

        self.final_top_k = (
            final_top_k
        )

        self.mmr_lambda = (
            mmr_lambda
        )

        self.bm25_corpus_path = (
            bm25_corpus_path
        )


        self._closed = False


    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:

        if top_k is None:

            top_k = (
                self.final_top_k
            )


        if (
            not query
            or
            top_k <= 0
        ):

            return []


        # ====================================================
        # Candidate budgets
        # ====================================================

        rerank_candidate_k = max(
            self.rerank_top_k,
            top_k * 2,
        )


        hybrid_candidate_k = max(
            self.hybrid.final_top_k,
            rerank_candidate_k,
        )


        multi_query_candidate_k = max(
            hybrid_candidate_k * 2,
            40,
        )


        # ====================================================
        # 1. Multi-query hybrid candidate generation
        # ====================================================

        candidates = (
            self.multi_query.search(
                query,
                top_k=
                    multi_query_candidate_k,
            )
        )


        if not candidates:

            return []


        # ====================================================
        # 2. ONE cross-encoder rerank
        #
        # Always against the original query.
        # ====================================================

        reranked = (
            self.reranker.rerank(
                query,
                candidates,
                top_k=
                    rerank_candidate_k,
            )
        )


        if not reranked:

            return []


        # ====================================================
        # 3. Ensure every candidate has a Dense vector
        #
        # Dense-origin candidates normally already have one.
        #
        # BM25-only candidates may not, so only those missing
        # vectors are embedded here.
        # ====================================================

        valid_documents = []

        document_embeddings = []


        for item in reranked:

            vector = item.get(
                "vector"
            )


            if vector is None:

                vector = (
                    self.hybrid
                    .dense
                    .embedder
                    .encode_documents(
                        [
                            item[
                                "text"
                            ]
                        ]
                    )[0]
                )


                item[
                    "vector"
                ] = vector


            valid_documents.append(
                item
            )

            document_embeddings.append(
                vector
            )


        reranked = (
            valid_documents
        )


        if not reranked:

            return []


        # ====================================================
        # 4. Query embedding for MMR
        # ====================================================

        query_embedding = (
            self.hybrid
            .dense
            .embedder
            .encode_queries(
                [query]
            )[0]
        )


        # ====================================================
        # 5. MMR diversity selection
        # ====================================================

        selected = (
            mmr_select(
                query_embedding=
                    query_embedding,
                document_embeddings=
                    document_embeddings,
                documents=
                    reranked,
                top_k=
                    top_k,
                lambda_param=
                    self.mmr_lambda,
            )
        )


        # ====================================================
        # 6. Public final score
        #
        # Downstream consumers should see the strongest
        # semantic ranking signal:
        #
        #     Cross-Encoder rerank score
        #
        # Earlier retrieval diagnostics such as RRF scores
        # remain available in their own metadata fields.
        # ====================================================

        for item in selected:

            rerank_score = (
                item.get(
                    "rerank_score"
                )
            )


            if rerank_score is not None:

                item[
                    "score"
                ] = (
                    rerank_score
                )


        return selected


    # ========================================================
    # Cleanup
    # ========================================================

    def close(
        self,
    ):

        if self._closed:

            return


        self.hybrid.close()

        self._closed = True