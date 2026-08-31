from langgraph.graph import (
    START,
    END,
    StateGraph,
)

from adaptive_agentic_rag.orchestration.state import (
    AgentState,
    create_initial_state,
)

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.orchestration.adaptive_retry_policy import (
    AdaptiveRetryPolicy,
    RetryAction,
)


# ==========================================================
# Conditional routing after evidence grading
# ==========================================================

def route_after_evidence(
    state: AgentState,
) -> str:
    """
    Decide what the graph should do after evidence grading.

    Architecture
    ------------

        grade_evidence
              ↓
        AdaptiveRetryPolicy
              │
        ┌─────┼─────┐
        │     │     │
        ▼     ▼     ▼
    generate retry abstain

    Important
    ---------
    EvidenceGrader decides:

        "Is the current evidence sufficient?"

    AdaptiveRetryPolicy decides:

        "If it is not sufficient, is another retrieval
        attempt structurally justified?"

    These responsibilities intentionally remain separate.
    """

    # ======================================================
    # Evidence decision must exist at this point.
    # ======================================================

    evidence_sufficient = (
        state.get(
            "evidence_sufficient"
        )
    )


    if (
        evidence_sufficient
        is None
    ):

        raise ValueError(
            "Cannot route after evidence grading "
            "without evidence_sufficient."
        )


    # ======================================================
    # Retry configuration
    # ======================================================

    retry_count = (
        state.get(
            "retry_count",
            0,
        )
        or 0
    )


    max_retries = (
        state.get(
            "max_retries",
            0,
        )
        or 0
    )


    evidence_reasons = list(
        state.get(
            "evidence_reasons",
            [],
        )
        or []
    )


    # ======================================================
    # Policy is intentionally lightweight.
    #
    # It loads:
    #
    # - no model
    # - no embeddings
    # - no reranker
    # - no files
    #
    # Therefore creating it here with the request-specific
    # retry budget is cheap and keeps routing deterministic.
    # ======================================================

    retry_policy = (
        AdaptiveRetryPolicy(
            max_retries=(
                int(
                    max_retries
                )
            )
        )
    )


    decision = (
        retry_policy.decide(
            evidence_sufficient=(
                bool(
                    evidence_sufficient
                )
            ),

            retry_count=(
                int(
                    retry_count
                )
            ),

            evidence_reasons=(
                evidence_reasons
            ),
        )
    )


    # ======================================================
    # Policy → LangGraph route
    # ======================================================

    if (
        decision.action
        ==
        RetryAction.GENERATE
    ):

        return "generate"


    if (
        decision.action
        ==
        RetryAction.RETRY
    ):

        return "rewrite"


    if (
        decision.action
        ==
        RetryAction.ABSTAIN
    ):

        return "abstain"


    # ======================================================
    # Defensive programming.
    #
    # A future RetryAction must never silently fall into an
    # existing graph route.
    # ======================================================

    raise RuntimeError(
        (
            "Unsupported retry-policy action: "
            f"{decision.action!r}"
        )
    )


# ==========================================================
# Main Graph
# ==========================================================

class AdaptiveRAGGraph:

    def __init__(
        self,
    ):

        # --------------------------------------------------
        # Shared application services
        # --------------------------------------------------

        self.nodes = (
            RAGNodes()
        )


        # --------------------------------------------------
        # Graph definition
        # --------------------------------------------------

        builder = (
            StateGraph(
                AgentState
            )
        )


        # ==================================================
        # Nodes
        # ==================================================

        builder.add_node(
            "route_query",
            self.nodes.route_query,
        )


        builder.add_node(
            "retrieve",
            self.nodes.retrieve,
        )


        builder.add_node(
            "build_context",
            self.nodes.build_context,
        )


        builder.add_node(
            "grade_evidence",
            self.nodes.grade_evidence,
        )


        builder.add_node(
            "rewrite_query",
            self.nodes.rewrite_query,
        )


        builder.add_node(
            "generate",
            self.nodes.generate,
        )


        builder.add_node(
            "abstain",
            self.nodes.abstain,
        )


        builder.add_node(
            "grade_answer",
            self.nodes.grade_answer,
        )


        # ==================================================
        # Main forward path
        # ==================================================

        builder.add_edge(
            START,
            "route_query",
        )


        builder.add_edge(
            "route_query",
            "retrieve",
        )


        builder.add_edge(
            "retrieve",
            "build_context",
        )


        builder.add_edge(
            "build_context",
            "grade_evidence",
        )


        # ==================================================
        # Adaptive evidence / retry routing
        #
        # OLD:
        #
        # evidence sufficient?
        #   yes → generate
        #   no  → retry whenever budget remains
        #
        #
        # NEW:
        #
        # evidence sufficient
        #       ↓
        #    generate
        #
        # evidence insufficient
        #       ↓
        # AdaptiveRetryPolicy
        #       │
        #       ├── structural retrieval miss
        #       │       → rewrite
        #       │
        #       └── no justified retrieval miss
        #               → abstain
        # ==================================================

        builder.add_conditional_edges(
            "grade_evidence",

            route_after_evidence,

            {
                "generate":
                    "generate",

                "rewrite":
                    "rewrite_query",

                "abstain":
                    "abstain",
            },
        )


        # ==================================================
        # Self-correction loop
        #
        # Only AdaptiveRetryPolicy can enter this path now.
        # ==================================================

        builder.add_edge(
            "rewrite_query",
            "retrieve",
        )


        # ==================================================
        # Both successful answers and abstentions
        # must be graded.
        # ==================================================

        builder.add_edge(
            "generate",
            "grade_answer",
        )


        builder.add_edge(
            "abstain",
            "grade_answer",
        )


        # ==================================================
        # Final exit
        # ==================================================

        builder.add_edge(
            "grade_answer",
            END,
        )


        # ==================================================
        # Compile
        # ==================================================

        self.graph = (
            builder.compile()
        )


    # ======================================================
    # Execute
    # ======================================================

    def run(
        self,
        query: str,
        max_retries: int = 1,
    ) -> AgentState:

        initial_state = (
            create_initial_state(
                query=(
                    query
                ),

                max_retries=(
                    max_retries
                ),
            )
        )


        final_state = (
            self.graph.invoke(
                initial_state
            )
        )


        return final_state


    # ======================================================
    # Cleanup
    # ======================================================

    def close(
        self,
    ):

        self.nodes.close()