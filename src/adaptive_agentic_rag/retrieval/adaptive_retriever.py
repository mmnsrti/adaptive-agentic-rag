from adaptive_agentic_rag.agents.query_router import (
    QueryRouter,
)

from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever,
)


# ============================================================
# Canonical retrieval configuration
#
# V2-A won the frozen retrieval evaluation:
#
# - 1000-char chunks
# - 100-char overlap
# - Dense embedding = raw chunk text
# - BM25 = title + source + chunk text
# ============================================================

DEFAULT_DENSE_COLLECTION = (
    "multihop_chunks_v2"
)

DEFAULT_BM25_CORPUS_PATH = (
    "data/processed/"
    "processed_corpus_v2.json"
)


class AdaptiveRetriever:

    def __init__(
        self,
        collection_name: str = (
            DEFAULT_DENSE_COLLECTION
        ),
        bm25_corpus_path: str = (
            DEFAULT_BM25_CORPUS_PATH
        ),
    ):

        self.collection_name = (
            collection_name
        )

        self.bm25_corpus_path = (
            bm25_corpus_path
        )

        # ====================================================
        # Router
        # ====================================================

        self.router = QueryRouter()

        # ====================================================
        # Shared Dense retriever
        #
        # The same embedding model / Qdrant client is reused
        # by both the simple Dense path and the heavy path.
        # ====================================================

        self.dense = DenseRetriever(
            collection_name=
                collection_name
        )

        # ====================================================
        # Heavy retrieval path
        #
        # Dense and BM25 MUST use the same chunking version.
        #
        # V2 Dense:
        #     multihop_chunks_v2
        #
        # V2 BM25:
        #     processed_corpus_v2.json
        # ====================================================

        self.reranked = (
            RerankedRetriever(
                dense_retriever=
                    self.dense,

                bm25_corpus_path=
                    bm25_corpus_path,
            )
        )

        self._closed = False

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict:

        if (
            not query
            or
            top_k <= 0
        ):

            return {
                "decision":
                    None,

                "results":
                    [],
            }

        # ====================================================
        # Step 1
        # Decide retrieval strategy
        # ====================================================

        decision = (
            self.router.route(
                query
            )
        )

        # ====================================================
        # Step 2
        # Simple query
        # ====================================================

        if (
            decision[
                "retrieval_strategy"
            ]
            ==
            "dense"
        ):

            results = (
                self.dense.search(
                    query,
                    top_k=top_k,
                )
            )

        # ====================================================
        # Step 3
        # Multi-hop / complex query
        # ====================================================

        else:

            results = (
                self.reranked.search(
                    query,
                    top_k=top_k,
                )
            )

        return {
            "decision":
                decision,

            "results":
                results,
        }

    # ========================================================
    # Cleanup
    # ========================================================

    def close(
        self,
    ):

        if self._closed:
            return

        # RerankedRetriever
        # -> HybridRetriever
        # -> shared DenseRetriever
        #
        # Therefore closing reranked already closes the
        # shared Qdrant client. Do NOT close self.dense
        # a second time.

        self.reranked.close()

        self._closed = True