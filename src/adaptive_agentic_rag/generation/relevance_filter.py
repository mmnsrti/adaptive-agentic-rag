from dataclasses import dataclass

import numpy as np


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
        embedder,
        min_relevance_score: float = 0.40
    ):

        self.embedder = embedder

        self.min_relevance_score = (
            min_relevance_score
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
        # Query embedding
        #

        query_vector = (
            self.embedder
            .encode_queries(
                [query]
            )[0]
        )


        #
        # Batch encode all claims
        #

        claim_texts = [

            claim.claim

            for claim
            in supported_claims

        ]


        claim_vectors = (
            self.embedder
            .encode_documents(
                claim_texts
            )
        )


        relevant_claims = []

        filtered_claims = []


        for claim, vector in zip(

            supported_claims,

            claim_vectors

        ):


            #
            # Embeddings are normalized,
            # so dot product = cosine similarity
            #

            score = float(

                np.dot(
                    query_vector,
                    vector
                )

            )


            item = RelevantClaim(

                claim=claim.claim,

                citation_id=(
                    claim.citation_id
                ),

                relevance_score=round(
                    score,
                    4
                )

            )


            if (
                score
                >=
                self.min_relevance_score
            ):

                relevant_claims.append(
                    item
                )

            else:

                filtered_claims.append(
                    item
                )


        #
        # Keep most relevant claims first
        #

        relevant_claims.sort(

            key=lambda item:
                item.relevance_score,

            reverse=True

        )


        filtered_claims.sort(

            key=lambda item:
                item.relevance_score,

            reverse=True

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