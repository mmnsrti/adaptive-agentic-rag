def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_k=20
):
    """
    Reciprocal Rank Fusion

    result_lists:
        [
          dense_results,
          bm25_results
        ]
    """

    scores = {}
    metadata = {}


    for results in result_lists:

        for rank, item in enumerate(
            results,
            start=1
        ):

            doc_id = item["id"]

            rrf_score = 1 / (
                k + rank
            )


            scores[doc_id] = (
                scores.get(doc_id, 0)
                +
                rrf_score
            )


            if doc_id not in metadata:
                metadata[doc_id] = item



    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    output = []


    for doc_id, score in ranked[:top_k]:

        item = metadata[doc_id].copy()

        item["rrf_score"] = score

        output.append(item)


    return output