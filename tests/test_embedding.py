from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel
)



def test_embedding():

    model = EmbeddingModel()


    vectors = model.encode(
        [
            "Sam Bankman-Fried founded FTX",
            "Amazon Cyber Monday deals"
        ]
    )


    assert len(vectors) == 2

    assert len(vectors[0]) > 0