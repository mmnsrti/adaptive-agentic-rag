from dataclasses import dataclass


@dataclass
class RelevantClaim:

    claim: str

    citation_id: int

    relevance_score: float


@dataclass
class RelevanceFilterResult:

    relevant_claims: list[RelevantClaim]

    filtered_claims: list[RelevantClaim]

    total_claims: int


class ClaimRelevanceFilter:


    def __init__(
        self,
        reranker,
        max_relevant_claims: int = 2
    ):

        self.reranker = reranker

        self.max_relevant_claims = (
            max_relevant_claims
        )


    def filter(
        self,
        query: str,
        grounded_claims
    ) -> RelevanceFilterResult:


        supported_claims = [

            claim

            for claim in grounded_claims.claims

            if claim.supported

        ]


        if not supported_claims:

            return RelevanceFilterResult(

                relevant_claims=[],

                filtered_claims=[],

                total_claims=0

            )


        #
        # Convert claims to the format
        # expected by BGEReranker
        #

        documents = []


        for index, claim in enumerate(
            supported_claims
        ):

            documents.append(
                {
                    "id": f"claim_{index}",

                    "text": claim.claim,

                    "claim": claim
                }
            )


        #
        # Cross-encoder ranking
        #

        ranked = (
            self.reranker.rerank(

                query=query,

                documents=documents,

                top_k=len(documents)

            )
        )


        relevant_claims = []

        filtered_claims = []


        for index, item in enumerate(
            ranked
        ):


            original_claim = (
                item["claim"]
            )


            result = RelevantClaim(

                claim=(
                    original_claim.claim
                ),

                citation_id=(
                    original_claim.citation_id
                ),

                relevance_score=round(
                    float(
                        item["rerank_score"]
                    ),
                    4
                )

            )


            if (
                index
                <
                self.max_relevant_claims
            ):

                relevant_claims.append(
                    result
                )

            else:

                filtered_claims.append(
                    result
                )


        return RelevanceFilterResult(

            relevant_claims=(
                relevant_claims
            ),

            filtered_claims=(
                filtered_claims
            ),

            total_claims=len(
                supported_claims
            )

        )