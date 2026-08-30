import json

from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


DATASET_PATH = Path(
    "evaluation/datasets/"
    "frozen_e2e_smoke_20.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "e2e_smoke_retrieval_gate_diagnostic.json"
)


# ============================================================
# Dataset
# ============================================================

def load_examples():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Document helpers
# ============================================================

def unique_document_ids_from_results(
    results,
):

    output = []


    for item in (
        results
        or []
    ):

        document_id = (
            item.get(
                "document_id"
            )
        )


        if (
            document_id
            and
            document_id
            not in output
        ):

            output.append(
                document_id
            )


    return output


def unique_document_ids_from_context(
    context,
):

    output = []


    if context is None:

        return output


    for item in (
        context.items
        or []
    ):

        document_id = (
            item.document_id
        )


        if (
            document_id
            and
            document_id
            not in output
        ):

            output.append(
                document_id
            )


    return output


def gold_recall(
    predicted_ids,
    gold_ids,
):

    gold = set(
        gold_ids
        or []
    )


    predicted = set(
        predicted_ids
        or []
    )


    if not gold:

        return None


    return (
        len(
            gold
            &
            predicted
        )
        /
        len(
            gold
        )
    )


def complete_gold(
    predicted_ids,
    gold_ids,
):

    gold = set(
        gold_ids
        or []
    )


    if not gold:

        return None


    predicted = set(
        predicted_ids
        or []
    )


    return gold.issubset(
        predicted
    )


def mean(
    values,
):

    values = [
        value
        for value
        in values
        if value is not None
    ]


    if not values:

        return None


    return (
        sum(
            values
        )
        /
        len(
            values
        )
    )


# ============================================================
# Failure classification
# ============================================================

def classify_initial_failure(
    retrieval_recall,
    context_recall,
    context_complete,
):

    if (
        context_complete
        is True
    ):

        return (
            "grader_reject_complete_gold_context"
        )


    if (
        context_recall
        is not None
        and
        context_recall
        >=
        0.75
    ):

        return (
            "grader_reject_high_gold_context"
        )


    if (
        retrieval_recall
        is not None
        and
        retrieval_recall
        >=
        0.75
        and
        (
            context_recall
            is None
            or
            context_recall
            <
            0.75
        )
    ):

        return (
            "context_loss_candidate"
        )


    return (
        "retrieval_miss_candidate"
    )


# ============================================================
# One example
# ============================================================

def run_example(
    nodes,
    example,
):

    question = (
        example[
            "question"
        ]
    )


    question_type = (
        example[
            "question_type"
        ]
    )


    null_example = (
        question_type
        ==
        "null_query"
    )


    # ========================================================
    # IMPORTANT:
    #
    # Correct frozen-eval field.
    # ========================================================

    gold_document_ids = (
        example.get(
            "evidence_document_ids",
            [],
        )
    )


    state = {
        "original_query":
            question,

        "current_query":
            question,

        "retry_count":
            0,
    }


    # ========================================================
    # Route
    # ========================================================

    state.update(
        nodes.route_query(
            state
        )
    )


    router_query_type = (
        state[
            "query_type"
        ]
    )


    retrieval_strategy = (
        state[
            "retrieval_strategy"
        ]
    )


    # ========================================================
    # Original retrieval
    # ========================================================

    state.update(
        nodes.retrieve(
            state
        )
    )


    initial_retrieved_ids = (
        unique_document_ids_from_results(
            state[
                "retrieved_results"
            ]
        )
    )


    state.update(
        nodes.build_context(
            state
        )
    )


    initial_context_ids = (
        unique_document_ids_from_context(
            state[
                "context"
            ]
        )
    )


    state.update(
        nodes.grade_evidence(
            state
        )
    )


    initial_sufficient = (
        state[
            "evidence_sufficient"
        ]
    )


    initial_score = (
        state[
            "evidence_score"
        ]
    )


    initial_reasons = (
        state[
            "evidence_reasons"
        ]
    )


    initial_retrieval_recall = (
        gold_recall(
            predicted_ids=
                initial_retrieved_ids,
            gold_ids=
                gold_document_ids,
        )
    )


    initial_context_recall = (
        gold_recall(
            predicted_ids=
                initial_context_ids,
            gold_ids=
                gold_document_ids,
        )
    )


    initial_context_complete = (
        complete_gold(
            predicted_ids=
                initial_context_ids,
            gold_ids=
                gold_document_ids,
        )
    )


    # ========================================================
    # Rewrite
    # ========================================================

    rewrite_attempted = False

    rewritten_query = None

    final_retrieved_ids = (
        initial_retrieved_ids
    )

    final_context_ids = (
        initial_context_ids
    )

    final_retrieval_recall = (
        initial_retrieval_recall
    )

    final_context_recall = (
        initial_context_recall
    )

    final_context_complete = (
        initial_context_complete
    )

    final_sufficient = (
        initial_sufficient
    )

    final_score = (
        initial_score
    )

    final_reasons = (
        initial_reasons
    )


    if not initial_sufficient:

        rewrite_attempted = True


        state.update(
            nodes.rewrite_query(
                state
            )
        )


        rewritten_query = (
            state[
                "current_query"
            ]
        )


        state.update(
            nodes.retrieve(
                state
            )
        )


        final_retrieved_ids = (
            unique_document_ids_from_results(
                state[
                    "retrieved_results"
                ]
            )
        )


        state.update(
            nodes.build_context(
                state
            )
        )


        final_context_ids = (
            unique_document_ids_from_context(
                state[
                    "context"
                ]
            )
        )


        state.update(
            nodes.grade_evidence(
                state
            )
        )


        final_sufficient = (
            state[
                "evidence_sufficient"
            ]
        )


        final_score = (
            state[
                "evidence_score"
            ]
        )


        final_reasons = (
            state[
                "evidence_reasons"
            ]
        )


        final_retrieval_recall = (
            gold_recall(
                predicted_ids=
                    final_retrieved_ids,
                gold_ids=
                    gold_document_ids,
            )
        )


        final_context_recall = (
            gold_recall(
                predicted_ids=
                    final_context_ids,
                gold_ids=
                    gold_document_ids,
            )
        )


        final_context_complete = (
            complete_gold(
                predicted_ids=
                    final_context_ids,
                gold_ids=
                    gold_document_ids,
            )
        )


    # ========================================================
    # Failure category
    # ========================================================

    failure_category = None


    if null_example:

        failure_category = (
            "null_false_accept"
            if final_sufficient
            else "null_correct_reject"
        )


    elif not initial_sufficient:

        failure_category = (
            classify_initial_failure(
                retrieval_recall=
                    initial_retrieval_recall,
                context_recall=
                    initial_context_recall,
                context_complete=
                    initial_context_complete,
            )
        )


    else:

        failure_category = (
            "initial_gate_accept"
        )


    rewrite_recall_delta = None

    rewrite_context_delta = None


    if (
        rewrite_attempted
        and
        initial_retrieval_recall
        is not None
        and
        final_retrieval_recall
        is not None
    ):

        rewrite_recall_delta = (
            final_retrieval_recall
            -
            initial_retrieval_recall
        )


    if (
        rewrite_attempted
        and
        initial_context_recall
        is not None
        and
        final_context_recall
        is not None
    ):

        rewrite_context_delta = (
            final_context_recall
            -
            initial_context_recall
        )


    return {
        "id":
            example[
                "id"
            ],

        "question":
            question,

        "question_type":
            question_type,

        "gold_answer":
            example.get(
                "answer"
            ),

        "gold_document_ids":
            gold_document_ids,

        "router_query_type":
            router_query_type,

        "retrieval_strategy":
            retrieval_strategy,

        # ----------------------------------------------------
        # Initial
        # ----------------------------------------------------

        "initial_retrieved_document_ids":
            initial_retrieved_ids,

        "initial_context_document_ids":
            initial_context_ids,

        "initial_retrieval_gold_recall":
            initial_retrieval_recall,

        "initial_context_gold_recall":
            initial_context_recall,

        "initial_context_gold_complete":
            initial_context_complete,

        "initial_evidence_sufficient":
            initial_sufficient,

        "initial_evidence_score":
            initial_score,

        "initial_evidence_reasons":
            initial_reasons,

        # ----------------------------------------------------
        # Rewrite
        # ----------------------------------------------------

        "rewrite_attempted":
            rewrite_attempted,

        "rewritten_query":
            rewritten_query,

        "final_retrieved_document_ids":
            final_retrieved_ids,

        "final_context_document_ids":
            final_context_ids,

        "final_retrieval_gold_recall":
            final_retrieval_recall,

        "final_context_gold_recall":
            final_context_recall,

        "final_context_gold_complete":
            final_context_complete,

        "final_evidence_sufficient":
            final_sufficient,

        "final_evidence_score":
            final_score,

        "final_evidence_reasons":
            final_reasons,

        "rewrite_retrieval_recall_delta":
            rewrite_recall_delta,

        "rewrite_context_recall_delta":
            rewrite_context_delta,

        "rewrite_rescued":
            (
                rewrite_attempted
                and
                final_sufficient
            ),

        "failure_category":
            failure_category,
    }


# ============================================================
# Summary
# ============================================================

def summarize(
    records,
):

    answerable = [
        record
        for record
        in records
        if (
            record[
                "question_type"
            ]
            !=
            "null_query"
        )
    ]


    null_examples = [
        record
        for record
        in records
        if (
            record[
                "question_type"
            ]
            ==
            "null_query"
        )
    ]


    rewrites = [
        record
        for record
        in records
        if (
            record[
                "rewrite_attempted"
            ]
        )
    ]


    rewrite_improved_retrieval = [
        record
        for record
        in rewrites
        if (
            record[
                "rewrite_retrieval_recall_delta"
            ]
            is not None
            and
            record[
                "rewrite_retrieval_recall_delta"
            ]
            >
            0
        )
    ]


    rewrite_harmed_retrieval = [
        record
        for record
        in rewrites
        if (
            record[
                "rewrite_retrieval_recall_delta"
            ]
            is not None
            and
            record[
                "rewrite_retrieval_recall_delta"
            ]
            <
            0
        )
    ]


    rewrite_equal_retrieval = [
        record
        for record
        in rewrites
        if (
            record[
                "rewrite_retrieval_recall_delta"
            ]
            ==
            0
        )
    ]


    complete_context_rejected = [
        record
        for record
        in answerable
        if (
            record[
                "initial_context_gold_complete"
            ]
            is True
            and
            record[
                "initial_evidence_sufficient"
            ]
            is False
        )
    ]


    high_context_rejected = [
        record
        for record
        in answerable
        if (
            record[
                "initial_context_gold_recall"
            ]
            is not None
            and
            record[
                "initial_context_gold_recall"
            ]
            >=
            0.75
            and
            record[
                "initial_evidence_sufficient"
            ]
            is False
        )
    ]


    categories = {}


    for record in records:

        category = (
            record[
                "failure_category"
            ]
        )


        categories[
            category
        ] = (
            categories.get(
                category,
                0,
            )
            +
            1
        )


    return {
        "total":
            len(
                records
            ),

        "answerable":
            len(
                answerable
            ),

        "null":
            len(
                null_examples
            ),

        "mean_initial_retrieval_gold_recall":
            mean(
                [
                    record[
                        "initial_retrieval_gold_recall"
                    ]
                    for record
                    in answerable
                ]
            ),

        "mean_initial_context_gold_recall":
            mean(
                [
                    record[
                        "initial_context_gold_recall"
                    ]
                    for record
                    in answerable
                ]
            ),

        "initial_gate_accepts_answerable":
            sum(
                1
                for record
                in answerable
                if (
                    record[
                        "initial_evidence_sufficient"
                    ]
                )
            ),

        "final_gate_accepts_answerable":
            sum(
                1
                for record
                in answerable
                if (
                    record[
                        "final_evidence_sufficient"
                    ]
                )
            ),

        "complete_gold_context_rejected":
            len(
                complete_context_rejected
            ),

        "high_gold_context_rejected":
            len(
                high_context_rejected
            ),

        "rewrite_attempts":
            len(
                rewrites
            ),

        "rewrite_rescues":
            sum(
                1
                for record
                in rewrites
                if (
                    record[
                        "rewrite_rescued"
                    ]
                )
            ),

        "rewrite_retrieval_improved":
            len(
                rewrite_improved_retrieval
            ),

        "rewrite_retrieval_harmed":
            len(
                rewrite_harmed_retrieval
            ),

        "rewrite_retrieval_unchanged":
            len(
                rewrite_equal_retrieval
            ),

        "mean_rewrite_retrieval_recall_delta":
            mean(
                [
                    record[
                        "rewrite_retrieval_recall_delta"
                    ]
                    for record
                    in rewrites
                ]
            ),

        "mean_rewrite_context_recall_delta":
            mean(
                [
                    record[
                        "rewrite_context_recall_delta"
                    ]
                    for record
                    in rewrites
                ]
            ),

        "null_correct_rejects":
            sum(
                1
                for record
                in null_examples
                if not (
                    record[
                        "final_evidence_sufficient"
                    ]
                )
            ),

        "null_false_accepts":
            sum(
                1
                for record
                in null_examples
                if (
                    record[
                        "final_evidence_sufficient"
                    ]
                )
            ),

        "failure_categories":
            categories,
    }


# ============================================================
# Main
# ============================================================

def main():

    examples = (
        load_examples()
    )


    nodes = (
        RAGNodes()
    )


    records = []


    try:

        for (
            index,
            example,
        ) in enumerate(
            examples,
            start=1,
        ):

            print(
                "\n"
                +
                "=" * 100
            )


            print(
                (
                    f"{index}/"
                    f"{len(examples)} | "
                    f"{example['question_type']}"
                )
            )


            print(
                example[
                    "question"
                ]
            )


            record = (
                run_example(
                    nodes=
                        nodes,
                    example=
                        example,
                )
            )


            records.append(
                record
            )


            print(
                "\nGold docs:",
                record[
                    "gold_document_ids"
                ],
            )


            print(
                "Initial retrieval recall:",
                record[
                    "initial_retrieval_gold_recall"
                ],
            )


            print(
                "Initial context recall:",
                record[
                    "initial_context_gold_recall"
                ],
            )


            print(
                "Initial evidence:",
                (
                    record[
                        "initial_evidence_sufficient"
                    ],
                    round(
                        record[
                            "initial_evidence_score"
                        ],
                        4,
                    ),
                ),
            )


            if (
                record[
                    "rewrite_attempted"
                ]
            ):

                print(
                    "Rewrite:",
                    record[
                        "rewritten_query"
                    ],
                )


                print(
                    "Final retrieval recall:",
                    record[
                        "final_retrieval_gold_recall"
                    ],
                )


                print(
                    "Final context recall:",
                    record[
                        "final_context_gold_recall"
                    ],
                )


                print(
                    "Final evidence:",
                    (
                        record[
                            "final_evidence_sufficient"
                        ],
                        round(
                            record[
                                "final_evidence_score"
                            ],
                            4,
                        ),
                    ),
                )


            print(
                "Classification:",
                record[
                    "failure_category"
                ],
            )


    finally:

        nodes.close()


    summary = (
        summarize(
            records
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 100
    )


    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "summary":
                    summary,

                "records":
                    records,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()