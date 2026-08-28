from typing import Any

from adaptive_agentic_rag.agents.query_router import (
    QueryRouter
)

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder
)

from adaptive_agentic_rag.orchestration.state import (
    AgentState
)


def _enum_value(
    value: Any
) -> Any:
    """
    Convert Enum values to plain strings
    while leaving normal values unchanged.
    """

    return getattr(
        value,
        "value",
        value
    )


class RAGNodes:
    """
    Node implementations used by the LangGraph workflow.

    Important:
    Models and services live here,
    NOT inside AgentState.
    """

    def __init__(self):

        self.router = (
            QueryRouter()
        )

        self.retriever = (
            AdaptiveRetriever()
        )

        self.context_builder = (
            ContextBuilder()
        )


    # =====================================================
    # Node 1
    # Query routing
    # =====================================================

    def route_query(
        self,
        state: AgentState
    ) -> dict:

        query = (
            state["current_query"]
        )


        decision = (
            self.router.route(
                query
            )
        )


        return {

            "query_type":
                _enum_value(
                    decision.query_type
                ),

            "retrieval_strategy":
                _enum_value(
                    decision.retrieval_strategy
                ),

            "use_reranker":
                decision.use_reranker,

            "use_mmr":
                decision.use_mmr
        }


    # =====================================================
    # Node 2
    # Retrieval
    # =====================================================

    def retrieve(
        self,
        state: AgentState
    ) -> dict:

        query = (
            state["current_query"]
        )


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


    # =====================================================
    # Node 3
    # Context construction
    # =====================================================

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


    # =====================================================
    # Resource cleanup
    # =====================================================

    def close(self):

        close_method = getattr(
            self.retriever,
            "close",
            None
        )


        if callable(
            close_method
        ):

            close_method()