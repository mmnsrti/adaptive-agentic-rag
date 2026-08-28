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


class AdaptiveRAGGraph:


    def __init__(self):

        #
        # Shared application dependencies
        #

        self.nodes = (
            RAGNodes()
        )


        #
        # Build graph
        #

        builder = (
            StateGraph(
                AgentState
            )
        )


        # ---------------------------------------------
        # Register nodes
        # ---------------------------------------------

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


        # ---------------------------------------------
        # Define graph flow
        # ---------------------------------------------

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
            END
        )


        #
        # Compile
        #

        self.graph = (
            builder.compile()
        )


    # =====================================================
    # Execute graph
    # =====================================================

    def run(
        self,
        query: str,
        max_retries: int = 1
    ) -> AgentState:


        initial_state = (
            create_initial_state(

                query=query,

                max_retries=(
                    max_retries
                )

            )
        )


        final_state = (
            self.graph.invoke(
                initial_state
            )
        )


        return final_state


    # =====================================================
    # Cleanup
    # =====================================================

    def close(self):

        self.nodes.close()