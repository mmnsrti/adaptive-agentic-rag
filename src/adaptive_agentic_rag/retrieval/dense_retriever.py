from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel
)

from adaptive_agentic_rag.vectorstore.qdrant_store import (
    QdrantVectorStore
)



class DenseRetriever:


    def __init__(
        self,
        collection_name: str = "multihop_chunks"
    ):

        self.embedder = EmbeddingModel()

        self.store = QdrantVectorStore(
            collection_name=collection_name
        )



    def embed_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):

        return self.embedder.encode_queries(
            queries,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )



    def embed_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):

        return self.embedder.encode_documents(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )



    def search_by_vector(
        self,
        query_vector,
        top_k: int = 5
    ):


        response = self.store.client.query_points(
            collection_name=self.store.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=True
        )


        results = []


        for point in response.points:


            item = {

                "score": point.score,

                **point.payload

            }


            if point.vector is not None:

                item["vector"] = point.vector


            results.append(item)



        return results


    def search(
        self,
        query: str,
        top_k: int = 5
    ):


        query_vector = self.embedder.encode_queries(
            [query]
        )[0]


        return self.search_by_vector(
            query_vector=query_vector,
            top_k=top_k
        )



    def close(self):

        self.store.client.close()