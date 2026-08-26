from adaptive_agentic_rag.agents.query_router import (
    QueryRouter
)

from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever
)

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever
)



class AdaptiveRetriever:


    def __init__(self):


        self.router = QueryRouter()

        self.dense = DenseRetriever()


        self.reranked = RerankedRetriever(
            dense_retriever=self.dense
        )



    def search(
        self,
        query,
        top_k=5
    ):


        #
        # Step 1
        # Decide strategy
        #

        decision = self.router.route(
            query
        )



        #
        # Step 2
        # Simple query
        #

        if (
            decision["retrieval_strategy"]
            == "dense"
        ):


            results = self.dense.search(

                query,

                top_k=top_k

            )


        #
        # Step 3
        # Complex query
        #

        else:


            results = self.reranked.search(

                query,

                top_k=top_k

            )



        return {

            "decision": decision,

            "results": results

        }



    def close(self):


        self.dense.close()


        self.reranked.close()