def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:


    retrieved_top_k = set(
        retrieved_ids[:k]
    )


    relevant = set(
        relevant_ids
    )


    if len(relevant) == 0:
        return 0.0


    return len(
        retrieved_top_k.intersection(
            relevant
        )
    ) / len(relevant)