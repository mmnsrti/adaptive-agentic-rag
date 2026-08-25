import numpy as np



def cosine_similarity(
    a,
    b
):

    return np.dot(a,b)



def mmr_select(
    query_embedding,
    document_embeddings,
    documents,
    top_k=5,
    lambda_param=0.7
):


    selected = []

    candidates = list(
        range(len(documents))
    )


    while len(selected) < top_k and candidates:


        best_score = -float("inf")

        best_candidate = None



        for idx in candidates:


            relevance = cosine_similarity(
                query_embedding,
                document_embeddings[idx]
            )



            if not selected:

                diversity = 0


            else:

                diversity = max(

                    cosine_similarity(
                        document_embeddings[idx],
                        document_embeddings[selected_idx]
                    )

                    for selected_idx in selected

                )



            score = (

                lambda_param * relevance

                -

                (1-lambda_param) * diversity

            )


            if score > best_score:

                best_score = score

                best_candidate = idx



        selected.append(
            best_candidate
        )


        candidates.remove(
            best_candidate
        )


    return [

        documents[idx]

        for idx in selected

    ]