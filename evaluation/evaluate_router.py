import json

from collections import Counter, defaultdict
from pathlib import Path

from adaptive_agentic_rag.agents.query_router import (
    QueryRouter
)


DATASET_PATH = Path(
    "evaluation/datasets/router_dev_control_180.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/router_dev.json"
)


# ============================================================
# Load
# ============================================================

def load_examples():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Main evaluation
# ============================================================

def main():

    examples = (
        load_examples()
    )


    router = (
        QueryRouter()
    )


    predictions = []


    query_type_correct = 0

    strategy_correct = 0

    rerank_correct = 0

    mmr_correct = 0

    exact_correct = 0


    confusion = defaultdict(
        Counter
    )


    route_distribution = Counter()


    for example in examples:

        decision = (
            router.route(
                example[
                    "question"
                ]
            )
        )


        predicted_type = (
            decision[
                "query_type"
            ]
        )


        predicted_strategy = (
            decision[
                "retrieval_strategy"
            ]
        )


        predicted_rerank = (
            decision[
                "rerank"
            ]
        )


        predicted_mmr = (
            decision[
                "mmr"
            ]
        )


        gold_type = (
            example[
                "gold_query_type"
            ]
        )


        gold_strategy = (
            example[
                "gold_retrieval_strategy"
            ]
        )


        gold_rerank = (
            example[
                "gold_rerank"
            ]
        )


        gold_mmr = (
            example[
                "gold_mmr"
            ]
        )


        type_ok = (
            predicted_type
            ==
            gold_type
        )


        strategy_ok = (
            predicted_strategy
            ==
            gold_strategy
        )


        rerank_ok = (
            predicted_rerank
            ==
            gold_rerank
        )


        mmr_ok = (
            predicted_mmr
            ==
            gold_mmr
        )


        exact_ok = (
            type_ok
            and
            strategy_ok
            and
            rerank_ok
            and
            mmr_ok
        )


        query_type_correct += int(
            type_ok
        )

        strategy_correct += int(
            strategy_ok
        )

        rerank_correct += int(
            rerank_ok
        )

        mmr_correct += int(
            mmr_ok
        )

        exact_correct += int(
            exact_ok
        )


        confusion[
            gold_type
        ][
            predicted_type
        ] += 1


        route_key = (
            f"{predicted_type}"
            f"|{predicted_strategy}"
            f"|rerank={predicted_rerank}"
            f"|mmr={predicted_mmr}"
        )


        route_distribution[
            route_key
        ] += 1


        predictions.append(
            {
                "id":
                    example[
                        "id"
                    ],

                "question":
                    example[
                        "question"
                    ],

                "source":
                    example[
                        "source"
                    ],

                "gold":
                    {
                        "query_type":
                            gold_type,

                        "retrieval_strategy":
                            gold_strategy,

                        "rerank":
                            gold_rerank,

                        "mmr":
                            gold_mmr
                    },

                "predicted":
                    decision,

                "correct":
                    exact_ok
            }
        )


    total = len(
        examples
    )


    summary = {
        "total":
            total,

        "query_type_accuracy":
            (
                query_type_correct
                /
                total
            ),

        "strategy_accuracy":
            (
                strategy_correct
                /
                total
            ),

        "rerank_accuracy":
            (
                rerank_correct
                /
                total
            ),

        "mmr_accuracy":
            (
                mmr_correct
                /
                total
            ),

        "exact_decision_accuracy":
            (
                exact_correct
                /
                total
            ),

        "route_distribution":
            dict(
                route_distribution
            ),

        "confusion_matrix":
            {
                gold:
                    dict(
                        predictions_
                    )

                for (
                    gold,
                    predictions_
                )
                in confusion.items()
            }
    }


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "summary":
                    summary,

                "predictions":
                    predictions
            },
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\n===== ROUTER EVALUATION ====="
    )


    print(
        "Total:",
        total
    )


    print(
        "Query-type accuracy:",
        round(
            summary[
                "query_type_accuracy"
            ],
            4
        )
    )


    print(
        "Strategy accuracy:",
        round(
            summary[
                "strategy_accuracy"
            ],
            4
        )
    )


    print(
        "Rerank accuracy:",
        round(
            summary[
                "rerank_accuracy"
            ],
            4
        )
    )


    print(
        "MMR accuracy:",
        round(
            summary[
                "mmr_accuracy"
            ],
            4
        )
    )


    print(
        "Exact decision accuracy:",
        round(
            summary[
                "exact_decision_accuracy"
            ],
            4
        )
    )


    print(
        "\n===== CONFUSION MATRIX ====="
    )


    for gold_type, row in (
        summary[
            "confusion_matrix"
        ].items()
    ):

        print(
            gold_type,
            "→",
            row
        )


    print(
        "\n===== ROUTE DISTRIBUTION ====="
    )


    for route, count in (
        summary[
            "route_distribution"
        ].items()
    ):

        print(
            count,
            route
        )


    print(
        "\nSaved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()