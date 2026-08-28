from adaptive_agentic_rag.orchestration.graph import (
    AdaptiveRAGGraph
)


def print_state_summary(
    title: str,
    state: dict
):

    print(
        "\n"
        "================================"
    )

    print(title)


    print(
        "\n===== QUERY ====="
    )

    print(
        "Original:",
        state[
            "original_query"
        ]
    )

    print(
        "Current:",
        state[
            "current_query"
        ]
    )


    print(
        "\n===== ROUTING ====="
    )

    print(
        "Query type:",
        state[
            "query_type"
        ]
    )

    print(
        "Retrieval strategy:",
        state[
            "retrieval_strategy"
        ]
    )

    print(
        "Use reranker:",
        state[
            "use_reranker"
        ]
    )

    print(
        "Use MMR:",
        state[
            "use_mmr"
        ]
    )


    print(
        "\n===== RETRIEVAL ====="
    )

    print(
        "Retrieved chunks:",
        len(
            state[
                "retrieved_results"
            ]
        )
    )


    print(
        "\n===== EVIDENCE ====="
    )

    print(
        "Sufficient:",
        state[
            "evidence_sufficient"
        ]
    )

    print(
        "Score:",
        state[
            "evidence_score"
        ]
    )

    print(
        "Reasons:",
        state[
            "evidence_reasons"
        ]
    )


    print(
        "\n===== SELF CORRECTION ====="
    )

    print(
        "Retry count:",
        state[
            "retry_count"
        ]
    )

    print(
        "Max retries:",
        state[
            "max_retries"
        ]
    )

    print(
        "Rewritten:",
        state[
            "rewritten"
        ]
    )


    print(
        "\n===== FINAL ====="
    )

    print(
        "Abstained:",
        state[
            "abstained"
        ]
    )

    print(
        "Final answer:",
        state[
            "final_answer"
        ]
    )


def main():

    workflow = (
        AdaptiveRAGGraph()
    )


    try:

        # ==================================================
        # CASE 1
        # Evidence should already be sufficient
        # ==================================================

        query_1 = (
            "Compare Amazon and Walmart deals"
        )

        state_1 = (
            workflow.run(
                query=query_1,
                max_retries=1
            )
        )


        print_state_summary(
            title=(
                "CASE 1 — "
                "SUFFICIENT EVIDENCE"
            ),
            state=state_1
        )


        assert (
            state_1[
                "evidence_sufficient"
            ]
            is True
        )


        assert (
            state_1[
                "retry_count"
            ]
            ==
            0
        )


        assert (
            state_1[
                "rewritten"
            ]
            is False
        )


        assert (
            state_1[
                "abstained"
            ]
            is False
        )


        #
        # Generation is not connected
        # to LangGraph yet.
        #

        assert (
            state_1[
                "final_answer"
            ]
            is None
        )


        # ==================================================
        # CASE 2
        # Evidence should remain insufficient,
        # even after one rewrite.
        # ==================================================

        query_2 = (
            "Compare Amazon and AcmeMart "
            "loyalty cashback shipping deals"
        )


        state_2 = (
            workflow.run(
                query=query_2,
                max_retries=1
            )
        )


        print_state_summary(
            title=(
                "CASE 2 — "
                "REWRITE + ABSTENTION"
            ),
            state=state_2
        )


        assert (
            state_2[
                "evidence_sufficient"
            ]
            is False
        )


        assert (
            state_2[
                "retry_count"
            ]
            ==
            1
        )


        assert (
            state_2[
                "rewritten"
            ]
            is True
        )


        assert (
            state_2[
                "current_query"
            ]
            !=
            state_2[
                "original_query"
            ]
        )


        assert (
            state_2[
                "abstained"
            ]
            is True
        )


        assert (
            state_2[
                "final_answer"
            ]
            is not None
        )


        print(
            "\n"
            "================================"
        )

        print(
            "Evidence graph execution: OK"
        )


    finally:

        workflow.close()


if __name__ == "__main__":

    main()