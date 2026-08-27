from dataclasses import dataclass

import numpy as np


@dataclass
class AnswerGrade:

    passed: bool

    correct_abstention: bool

    citation_valid: bool

    supported_claim_ratio: float

    relevance_score: float | None

    reasons: list[str]


class AnswerGrader:


    def __init__(
        self,
        embedder,
        min_supported_ratio: float = 0.50,
        min_relevance_score: float = 0.35
    ):

        self.embedder = embedder

        self.min_supported_ratio = (
            min_supported_ratio
        )

        self.min_relevance_score = (
            min_relevance_score
        )


    def _relevance_score(
        self,
        query: str,
        answer: str
    ) -> float:

        query_vector = (
            self.embedder
            .encode_queries(
                [query]
            )[0]
        )


        answer_vector = (
            self.embedder
            .encode_documents(
                [answer]
            )[0]
        )


        #
        # Embeddings are normalized,
        # so dot product = cosine similarity
        #

        score = float(
            np.dot(
                query_vector,
                answer_vector
            )
        )


        return round(
            score,
            4
        )


    def grade(
        self,
        query: str,
        generation_result,
        evidence_sufficient: bool
    ) -> AnswerGrade:


        reasons = []


        #
        # =====================================
        # Case 1:
        # Evidence was insufficient
        # =====================================
        #

        if not evidence_sufficient:


            correct_abstention = (
                generation_result.abstained
            )


            if correct_abstention:

                reasons.append(
                    "Correctly abstained when evidence was insufficient."
                )

            else:

                reasons.append(
                    "The system generated an answer despite insufficient evidence."
                )


            return AnswerGrade(

                passed=correct_abstention,

                correct_abstention=(
                    correct_abstention
                ),

                citation_valid=(
                    generation_result
                    .citation_valid
                ),

                supported_claim_ratio=0.0,

                relevance_score=None,

                reasons=reasons
            )


        #
        # =====================================
        # Case 2:
        # Evidence existed but generation
        # abstained
        # =====================================
        #

        if generation_result.abstained:

            return AnswerGrade(

                passed=False,

                correct_abstention=False,

                citation_valid=(
                    generation_result
                    .citation_valid
                ),

                supported_claim_ratio=0.0,

                relevance_score=None,

                reasons=[
                    (
                        "Generation abstained even "
                        "though evidence was considered sufficient."
                    )
                ]
            )


        #
        # =====================================
        # Supported claim ratio
        # =====================================
        #

        total_claims = (

            generation_result.supported_claims
            +
            generation_result.unsupported_claims

        )


        if total_claims == 0:

            supported_ratio = 0.0

        else:

            supported_ratio = (

                generation_result.supported_claims
                /
                total_claims

            )


        supported_ratio = round(
            supported_ratio,
            4
        )


        enough_supported_claims = (

            supported_ratio
            >=
            self.min_supported_ratio

        )


        if not enough_supported_claims:

            reasons.append(

                (
                    "Too many generated claims were unsupported: "
                    f"{supported_ratio:.2f} "
                    f"< {self.min_supported_ratio:.2f}"
                )

            )


        #
        # =====================================
        # Citation validity
        # =====================================
        #

        citation_valid = (
            generation_result
            .citation_valid
        )


        if not citation_valid:

            reasons.append(
                "Final answer contains invalid or missing citations."
            )


        #
        # =====================================
        # Semantic relevance
        # =====================================
        #

        relevance_score = (
            self._relevance_score(

                query=query,

                answer=(
                    generation_result.answer
                )

            )
        )


        relevant = (

            relevance_score
            >=
            self.min_relevance_score

        )


        if not relevant:

            reasons.append(

                (
                    "Final answer has low semantic relevance "
                    f"to the original query: "
                    f"{relevance_score:.2f} "
                    f"< {self.min_relevance_score:.2f}"
                )

            )


        #
        # =====================================
        # Final decision
        # =====================================
        #

        passed = (

            citation_valid

            and

            enough_supported_claims

            and

            relevant

        )


        if passed:

            reasons.insert(
                0,
                "Final answer passed baseline quality checks."
            )


        return AnswerGrade(

            passed=passed,

            correct_abstention=False,

            citation_valid=(
                citation_valid
            ),

            supported_claim_ratio=(
                supported_ratio
            ),

            relevance_score=(
                relevance_score
            ),

            reasons=reasons
        )