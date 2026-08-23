from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel
)

from adaptive_agentic_rag.vectorstore.qdrant_store import (
    QdrantVectorStore
)



class DenseRetriever:


    def __init__(
        self,
        collection_name="multihop_chunks",
        vector_size=1024
    ):


        self.embedder = EmbeddingModel()


        self.store = QdrantVectorStore(

            collection_name=
                collection_name,

            vector_size=
                vector_size
        )



    def search(
        self,
        query: str,
        top_k: int = 5
    ):


        query_embedding = (

            self.embedder
            .encode(
                [query]
            )[0]

        )


        response = (

            self.store.client
            .query_points(

                collection_name=
                    self.store.collection_name,

                query=
                    query_embedding.tolist(),

                limit=
                    top_k
            )
        )


        return [

            {
                "score": point.score,

                **point.payload

            }

            for point in response.points

        ]