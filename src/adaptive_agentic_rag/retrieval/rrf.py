from collections import defaultdict



def reciprocal_rank_fusion(
    result_lists,
    top_k=20,
    k=60
):


    scores = defaultdict(float)

    documents = {}



    for results in result_lists:


        for rank, doc in enumerate(
            results,
            start=1
        ):


            doc_id = doc["id"]


            #
            # RRF score
            #
            scores[doc_id] += (
                1 /
                (k + rank)
            )


            #
            # Keep best document object
            #
            if doc_id not in documents:

                documents[doc_id] = doc.copy()


            else:

                #
                # Dense result has vector
                # BM25 usually doesn't
                #
                if (
                    "vector" in doc
                    and
                    "vector" not in documents[doc_id]
                ):

                    documents[doc_id]["vector"] = (
                        doc["vector"]
                    )





    ranked = sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True

    )



    output = []



    for doc_id, score in ranked[:top_k]:


        item = documents[doc_id].copy()


        item["score"] = score


        output.append(item)



    return output