from adaptive_agentic_rag.retrieval.rrf import (
    reciprocal_rank_fusion
)


dense = [
    {
        "id":"A",
        "score":0.9
    },
    {
        "id":"B",
        "score":0.8
    },
    {
        "id":"C",
        "score":0.7
    }
]


bm25 = [
    {
        "id":"C",
        "score":20
    },
    {
        "id":"A",
        "score":15
    },
    {
        "id":"D",
        "score":10
    }
]


result = reciprocal_rank_fusion(
    [
        dense,
        bm25
    ]
)


for r in result:
    print(
        r["id"],
        r["rrf_score"]
    )