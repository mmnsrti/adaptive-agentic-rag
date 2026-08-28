from langgraph.graph import (
    START,
    END,
    StateGraph
)

from adaptive_agentic_rag.orchestration.state import (
    AgentState,
    create_initial_state
)

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes
)


# ==========================================================
# Conditional routing
# ==========================================================

def route_after_evidence(
    state: AgentState
) -> str:
    """
    Decide what happens after evidence grading.

    Possible routes:

    sufficient
        -> current retrieval has enough evidence

    rewrite
        -> evidence insufficient,
           but retry budget remains

    abstain
        -> evidence insufficient,
           retry budget exhausted
    """

    if (
        state[
            "evidence_sufficient"
        ]
        is True
    ):
        return "sufficient"


    retry_count = (
        state[
            "retry_count"
        ]
    )

    max_retries = (
        state[
            "max_retries"
        ]
    )


    if (
        retry_count
        <
        max_retries
    ):
        return "rewrite"


    return "abstain"


# ==========================================================
# Graph
# ==========================================================

class AdaptiveRAGGraph:

    def __init__(self):

        # --------------------------------------------------
        # Shared node dependencies
        # --------------------------------------------------

        self.nodes = (
            RAGNodes()
        )


        # --------------------------------------------------
        # Graph builder
        # --------------------------------------------------

        builder = (
            StateGraph(
                AgentState
            )
        )


        # ==================================================
        # Register nodes
        # ==================================================

        builder.add_node(
            "route_query",
            self.nodes.route_query
        )

        builder.add_node(
            "retrieve",
            self.nodes.retrieve
        )

        builder.add_node(
            "build_context",
            self.nodes.build_context
        )

        builder.add_node(
            "grade_evidence",
            self.nodes.grade_evidence
        )

        builder.add_node(
            "rewrite_query",
            self.nodes.rewrite_query
        )

        builder.add_node(
            "abstain",
            self.nodes.abstain
        )


        # ==================================================
        # Normal edges
        # ==================================================

        builder.add_edge(
            START,
            "route_query"
        )

        builder.add_edge(
            "route_query",
            "retrieve"
        )

        builder.add_edge(
            "retrieve",
            "build_context"
        )

        builder.add_edge(
            "build_context",
            "grade_evidence"
        )


        # ==================================================
        # Conditional edge
        # ==================================================

        builder.add_conditional_edges(
            "grade_evidence",

            route_after_evidence,

            {
                "sufficient":
                    END,

                "rewrite":
                    "rewrite_query",

                "abstain":
                    "abstain"
            }
        )


        # ==================================================
        # Self-correction loop
        # ==================================================

        builder.add_edge(
            "rewrite_query",
            "retrieve"
        )


        # ==================================================
        # Abstention exit
        # ==================================================

        builder.add_edge(
            "abstain",
            END
        )


        # ==================================================
        # Compile graph
        # ==================================================

        self.graph = (
            builder.compile()
        )


    # ======================================================
    # Execute workflow
    # ======================================================

    def run(
        self,
        query: str,
        max_retries: int = 1
    ) -> AgentState:

        initial_state = (
            create_initial_state(
                query=query,
                max_retries=max_retries
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

    def close(self):

        self.nodes.close()