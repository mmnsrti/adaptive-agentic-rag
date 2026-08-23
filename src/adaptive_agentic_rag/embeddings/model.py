from sentence_transformers import SentenceTransformer



MODEL_NAME = (
    "Qwen/Qwen3-Embedding-0.6B"
)



class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device="cuda"
        )


    def encode(
        self,
        texts: list[str]
    ):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings