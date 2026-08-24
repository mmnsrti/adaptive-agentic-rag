import pickle
from pathlib import Path


class BM25Retriever:

    def __init__(
        self,
        index_path="data/bm25/bm25_index.pkl"
    ):

        self.index_path = Path(index_path)

        self._load()


    def _tokenize(self, text):

        return text.lower().split()


    def _load(self):

        with open(
            self.index_path,
            "rb"
        ) as f:

            data = pickle.load(f)


        self.bm25 = data["bm25"]
        self.ids = data["ids"]
        self.texts = data["texts"]


    def search(
        self,
        query,
        top_k=20
    ):

        tokens = self._tokenize(query)


        scores = self.bm25.get_scores(
            tokens
        )


        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )


        results = []


        for idx in ranked_indices[:top_k]:

            results.append(
                {
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "score": float(scores[idx])
                }
            )


        return results