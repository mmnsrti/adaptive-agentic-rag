from adaptive_agentic_rag.agents.query_router import QueryRouter
from adaptive_agentic_rag.agents.evidence_grader import EvidenceGrader
from adaptive_agentic_rag.agents.query_rewriter import QueryRewriter

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder
)

from adaptive_agentic_rag.orchestration.state import (
    AgentState
)


ABSTENTION_MESSAGE = (
    "I don't have enough evidence in the provided "
    "sources to answer reliably."
)


class RAGNodes:
    """
    LangGraph node implementations.

    This class owns shared application services such as
    retrievers and graders.

    AgentState only contains per-request data.
    """

    def __init__(self):

        # --------------------------------------------------
        # Shared services
        # --------------------------------------------------

        self.router = QueryRouter()

        self.retriever = AdaptiveRetriever()

        self.context_builder = ContextBuilder()

        self.evidence_grader = EvidenceGrader()

        self.query_rewriter = QueryRewriter()


    # ======================================================
    # Node 1
    # Query routing
    # ======================================================

    def route_query(
        self,
        state: AgentState
    ) -> dict:

        query = state["current_query"]

        decision = self.router.route(
            query
        )

        return {
            "query_type":
                decision["query_type"],

            "retrieval_strategy":
                decision["retrieval_strategy"],

            "use_reranker":
                decision["rerank"],

            "use_mmr":
                decision["mmr"]
        }


    # ======================================================
    # Node 2
    # Retrieval
    # ======================================================

    def retrieve(
        self,
        state: AgentState
    ) -> dict:

        query = state["current_query"]

        retrieval_output = (
            self.retriever.search(
                query
            )
        )

        results = (
            retrieval_output[
                "results"
            ]
        )

        return {
            "retrieved_results":
                results
        }


    # ======================================================
    # Node 3
    # Context construction
    # ======================================================

    def build_context(
        self,
        state: AgentState
    ) -> dict:

        results = (
            state[
                "retrieved_results"
            ]
        )

        context = (
            self.context_builder.build(
                results
            )
        )

        return {
            "context":
                context
        }


    # ======================================================
    # Node 4
    # Evidence grading
    # ======================================================

    def grade_evidence(
        self,
        state: AgentState
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )

        context = (
            state[
                "context"
            ]
        )

        query_type = (
            state[
                "query_type"
            ]
        )

        if context is None:

            raise ValueError(
                "Cannot grade evidence "
                "without a built context."
            )

        if query_type is None:

            raise ValueError(
                "Cannot grade evidence "
                "without query_type."
            )

        grade = (
            self.evidence_grader.grade(
                query=original_query,
                context=context,
                query_type=query_type
            )
        )

        return {
            "evidence_sufficient":
                grade.sufficient,

            "evidence_score":
                grade.evidence_score,

            "evidence_reasons":
                grade.reasons
        }


    # ======================================================
    # Node 5
    # Query rewriting
    # ======================================================

    def rewrite_query(
        self,
        state: AgentState
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )

        query_type = (
            state[
                "query_type"
            ]
        )

        if query_type is None:

            raise ValueError(
                "Cannot rewrite query "
                "without query_type."
            )

        #
        # retry_count represents completed rewrites.
        #
        # First rewrite:
        #
        # retry_count = 0
        # attempt = 1
        #

        attempt = (
            state[
                "retry_count"
            ]
            + 1
        )

        rewritten_query = (
            self.query_rewriter.rewrite(
                query=original_query,
                query_type=query_type,
                attempt=attempt
            )
        )

        return {
            "current_query":
                rewritten_query,

            "retry_count":
                attempt,

            "rewritten":
                True,

            #
            # Reset round-specific state.
            #
            # The next retrieval round will
            # fill these values again.
            #

            "retrieved_results":
                [],

            "context":
                None,

            "evidence_sufficient":
                None,

            "evidence_score":
                None,

            "evidence_reasons":
                []
        }


    # ======================================================
    # Node 6
    # Abstention
    # ======================================================

    def abstain(
        self,
        state: AgentState
    ) -> dict:

        return {
            "abstained":
                True,

            "raw_answer":
                None,

            "final_answer":
                ABSTENTION_MESSAGE
        }


    # ======================================================
    # Resource cleanup
    # ======================================================

    def close(self):

        close_method = getattr(
            self.retriever,
            "close",
            None
        )

        if callable(close_method):
            close_method()