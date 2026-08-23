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


        query_vector = (
            self.embedder
            .encode([query])[0]
        )


        results = self.store.client.search(

            collection_name=
                self.store.collection_name,

            query_vector=
                query_vector.tolist(),

            limit=
                top_k
        )


        return [
            {
                "score": result.score,

                **result.payload
            }

            for result in results
        ]