from enum import Enum


class QueryType(Enum):

    SIMPLE = "simple"

    MULTIHOP = "multihop"

    COMPLEX = "complex"



class QueryRouter:


    def __init__(self):

        self.multihop_keywords = [

            "compare",
            "comparison",
            "difference",
            "versus",
            "vs",
            "relationship",
            "between",
            "why",
            "explain",
            "how does",
            "which one"

        ]


        self.complex_keywords = [

            "multiple",
            "several",
            "all",
            "summarize",
            "analyze",
            "reason"

        ]



    def classify(
        self,
        query: str
    ) -> QueryType:


        text = query.lower()



        #
        # Complex questions
        #

        for keyword in self.complex_keywords:

            if keyword in text:

                return QueryType.COMPLEX



        #
        # Multi-hop questions
        #

        for keyword in self.multihop_keywords:

            if keyword in text:

                return QueryType.MULTIHOP



        #
        # Long questions
        #

        if len(text.split()) > 15:

            return QueryType.MULTIHOP



        return QueryType.SIMPLE




    def route(
        self,
        query: str
    ):


        query_type = self.classify(
            query
        )



        if query_type == QueryType.SIMPLE:


            return {

                "query_type":
                    query_type.value,

                "retrieval_strategy":
                    "dense",

                "rerank":
                    False,

                "mmr":
                    False

            }



        elif query_type == QueryType.MULTIHOP:


            return {

                "query_type":
                    query_type.value,

                "retrieval_strategy":
                    "hybrid",

                "rerank":
                    True,

                "mmr":
                    True

            }



        else:


            return {

                "query_type":
                    query_type.value,

                "retrieval_strategy":
                    "hybrid",

                "rerank":
                    True,

                "mmr":
                    True

            }