import time
from statistics import mean

from adaptive_agentic_rag.evaluation.dataset_adapter import (
    get_relevant_document_keys,
    get_retrieved_document_keys
)

from adaptive_agentic_rag.evaluation.metrics import (
    recall_at_k,
    hit_at_k,
    complete_evidence_recall_at_k,
    reciprocal_rank,
    ndcg_at_k
)


DEFAULT_K_VALUES = [
    1,
    3,
    5,
    10,
    20
]


def evaluate_dense_retriever(
    dataset,
    retriever,
    limit: int | None = None,
    k_values: list[int] | None = None
) -> dict:

    if k_values is None:
        k_values = DEFAULT_K_VALUES

    max_k = max(k_values)

    if limit is not None:
        dataset = dataset.select(
            range(
                min(
                    limit,
                    len(dataset)
                )
            )
        )

    examples = [
        item
        for item in dataset
        if len(item["evidence_list"]) > 0
    ]

    skipped_no_evidence = (
        len(dataset)
        - len(examples)
    )

    queries = [
        item["query"]
        for item in examples
    ]

    print(
        f"Embedding {len(queries)} queries..."
    )

    embedding_start = time.perf_counter()

    query_vectors = retriever.embed_queries(
        queries,
        batch_size=32,
        show_progress_bar=True
    )

    embedding_seconds = (
        time.perf_counter()
        - embedding_start
    )

    metrics = {}

    for k in k_values:

        metrics[k] = {
            "recall": [],
            "hit": [],
            "complete_recall": [],
            "mrr": [],
            "ndcg": []
        }

    per_query = []

    retrieval_times = []

    print("Running retrieval...")

    for index, (
        example,
        query_vector
    ) in enumerate(
        zip(
            examples,
            query_vectors
        )
    ):

        start = time.perf_counter()

        results = retriever.search_by_vector(
            query_vector,
            top_k=max_k
        )

        retrieval_time = (
            time.perf_counter()
            - start
        )

        retrieval_times.append(
            retrieval_time
        )

        relevant_keys = (
            get_relevant_document_keys(
                example["evidence_list"]
            )
        )

        retrieved_keys = (
            get_retrieved_document_keys(
                results
            )
        )

        query_metrics = {}

        for k in k_values:

            recall = recall_at_k(
                retrieved_keys,
                relevant_keys,
                k
            )

            hit = hit_at_k(
                retrieved_keys,
                relevant_keys,
                k
            )

            complete_recall = (
                complete_evidence_recall_at_k(
                    retrieved_keys,
                    relevant_keys,
                    k
                )
            )

            mrr = reciprocal_rank(
                retrieved_keys,
                relevant_keys,
                k
            )

            ndcg = ndcg_at_k(
                retrieved_keys,
                relevant_keys,
                k
            )

            metrics[k]["recall"].append(
                recall
            )

            metrics[k]["hit"].append(
                hit
            )

            metrics[k]["complete_recall"].append(
                complete_recall
            )

            metrics[k]["mrr"].append(
                mrr
            )

            metrics[k]["ndcg"].append(
                ndcg
            )

            query_metrics[str(k)] = {
                "recall": recall,
                "hit": hit,
                "complete_recall":
                    complete_recall,
                "mrr": mrr,
                "ndcg": ndcg
            }

        per_query.append(
            {
                "index": index,
                "query":
                    example["query"],

                "question_type":
                    example["question_type"],

                "num_relevant_documents":
                    len(relevant_keys),

                "metrics":
                    query_metrics,

                "retrieved": [
                    {
                        "rank": rank,
                        "score":
                            result["score"],
                        "document_id":
                            result[
                                "document_id"
                            ],
                        "title":
                            result[
                                "metadata"
                            ]["title"],
                        "source":
                            result[
                                "metadata"
                            ]["source"]
                    }
                    for rank, result
                    in enumerate(
                        results,
                        start=1
                    )
                ]
            }
        )

        if (
            (index + 1) % 100
            == 0
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(examples)}"
            )

    summary_metrics = {}

    for k in k_values:

        summary_metrics[
            f"recall@{k}"
        ] = mean(
            metrics[k]["recall"]
        )

        summary_metrics[
            f"hit@{k}"
        ] = mean(
            metrics[k]["hit"]
        )

        summary_metrics[
            f"complete_recall@{k}"
        ] = mean(
            metrics[k][
                "complete_recall"
            ]
        )

        summary_metrics[
            f"mrr@{k}"
        ] = mean(
            metrics[k]["mrr"]
        )

        summary_metrics[
            f"ndcg@{k}"
        ] = mean(
            metrics[k]["ndcg"]
        )

    return {
        "num_examples":
            len(dataset),

        "num_evaluated":
            len(examples),

        "skipped_no_evidence":
            skipped_no_evidence,

        "embedding_seconds":
            embedding_seconds,

        "avg_retrieval_ms":
            mean(
                retrieval_times
            ) * 1000,

        "metrics":
            summary_metrics,

        "per_query":
            per_query
    }