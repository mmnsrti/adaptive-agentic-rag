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


            scores[doc_id] += (
                1 /
                (k + rank)
            )


            documents[doc_id] = doc





    ranked = sorted(

        scores.items(),

        key=lambda x:x[1],

        reverse=True

    )



    output = []



    for doc_id, score in ranked[:top_k]:


        item = documents[doc_id].copy()


        item["score"] = score


        output.append(item)



    return output