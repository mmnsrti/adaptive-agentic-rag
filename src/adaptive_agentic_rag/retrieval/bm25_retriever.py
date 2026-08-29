import json
import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Lexical retriever over processed chunks.

    Index representation includes:

    - chunk text
    - document title
    - source name

    Tokenization is normalized so punctuation and casing
    do not create unnecessary lexical mismatches.
    """

    def __init__(
        self,
        corpus_path: str = (
            "data/processed/"
            "processed_corpus.json"
        )
    ):

        with open(
            corpus_path,
            encoding="utf-8"
        ) as f:

            self.documents = json.load(
                f
            )


        # ====================================================
        # Build normalized lexical representation
        # ====================================================

        self.texts = [

            self._document_tokens(
                document
            )

            for document in self.documents
        ]


        self.bm25 = BM25Okapi(
            self.texts
        )


        self.doc_map = {

            document["id"]:
                document

            for document
            in self.documents
        }


    # ========================================================
    # Normalize/tokenize
    # ========================================================

    @staticmethod
    def _tokenize(
        text: str
    ) -> list[str]:

        if not text:

            return []


        #
        # Lowercase and normalize punctuation.
        #
        # Examples:
        #
        # "fraud,"       -> fraud
        # "Google's"     -> google, s
        # "CBSSports.com"-> cbssports, com
        # "GPT-4"        -> gpt, 4
        #

        return re.findall(
            r"[a-z0-9]+",
            text.lower()
        )


    # ========================================================
    # Build searchable document representation
    # ========================================================

    def _document_tokens(
        self,
        document: dict
    ) -> list[str]:

        metadata = (
            document.get(
                "metadata",
                {}
            )
            or {}
        )


        title = metadata.get(
            "title",
            ""
        )


        source = metadata.get(
            "source",
            ""
        )


        text = document.get(
            "text",
            ""
        )


        #
        # Title/source are intentionally part of the
        # lexical index because many enterprise and
        # MultiHopRAG questions explicitly reference
        # publications or document titles.
        #
        # We do not repeat them artificially here:
        # BM25 itself should determine their weight.
        #

        searchable_text = (
            f"{title}\n"
            f"{source}\n"
            f"{text}"
        )


        return self._tokenize(
            searchable_text
        )


    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> list[dict]:

        if (
            not query
            or
            top_k <= 0
        ):

            return []


        query_tokens = (
            self._tokenize(
                query
            )
        )


        if not query_tokens:

            return []


        scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )


        ranked = sorted(

            enumerate(
                scores
            ),

            key=lambda item: (
                -float(
                    item[1]
                ),
                item[0]
            )
        )


        limit = min(
            top_k,
            len(
                ranked
            )
        )


        results = []


        for index, score in ranked[
            :limit
        ]:

            document = (
                self.documents[
                    index
                ]
            )


            results.append(
                {
                    "id":
                        document[
                            "id"
                        ],

                    "document_id":
                        document[
                            "document_id"
                        ],

                    "text":
                        document[
                            "text"
                        ],

                    "metadata":
                        document.get(
                            "metadata",
                            {}
                        ),

                    "score":
                        float(
                            score
                        )
                }
            )


        return results