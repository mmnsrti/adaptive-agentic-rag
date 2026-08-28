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
# Conditional routing after evidence grading
# ==========================================================

def route_after_evidence(
    state: AgentState
) -> str:

    #
    # Evidence sufficient:
    #
    # move to grounded generation.
    #

    if (
        state[
            "evidence_sufficient"
        ]
        is True
    ):

        return "generate"


    #
    # Evidence insufficient.
    #
    # Check whether self-correction
    # still has retry budget.
    #

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


    #
    # No evidence and retry budget
    # exhausted.
    #

    return "abstain"


# ==========================================================
# Main Graph
# ==========================================================

class AdaptiveRAGGraph:

    def __init__(self):

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
            "generate",
            self.nodes.generate
        )


        builder.add_node(
            "abstain",
            self.nodes.abstain
        )


        builder.add_node(
            "grade_answer",
            self.nodes.grade_answer
        )


        # ==================================================
        # Main forward path
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
        # Conditional evidence routing
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
        # Both successful answers and abstentions
        # must be graded.
        # ==================================================

        builder.add_edge(
            "generate",
            "grade_answer"
        )


        builder.add_edge(
            "abstain",
            "grade_answer"
        )


        # ==================================================
        # Final exit
        # ==================================================

        builder.add_edge(
            "grade_answer",
            END
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