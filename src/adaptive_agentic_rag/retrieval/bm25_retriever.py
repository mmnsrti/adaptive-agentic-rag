from rank_bm25 import BM25Okapi

import json



class BM25Retriever:


    def __init__(
        self,
        corpus_path="data/processed/processed_corpus.json"
    ):


        with open(
            corpus_path,
            encoding="utf-8"
        ) as f:

            self.documents = json.load(f)



        self.texts = [

            doc["text"].split()

            for doc in self.documents

        ]



        self.bm25 = BM25Okapi(
            self.texts
        )



        self.doc_map = {

            doc["id"]: doc

            for doc in self.documents

        }





    def search(
        self,
        query,
        top_k=20
    ):


        scores = self.bm25.get_scores(
            query.split()
        )


        ranked = sorted(

            enumerate(scores),

            key=lambda x:x[1],

            reverse=True

        )



        results = []



        for index, score in ranked[:top_k]:


            doc = self.documents[index]



            results.append(

                {

                    "id":
                        doc["id"],


                    "document_id":
                        doc["document_id"],


                    "text":
                        doc["text"],


                    "metadata":
                        doc["metadata"],


                    "score":
                        float(score)

                }

            )



        return results
