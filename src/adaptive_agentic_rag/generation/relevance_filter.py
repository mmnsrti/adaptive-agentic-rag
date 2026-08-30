import re

from dataclasses import dataclass


DEFAULT_MIN_RELEVANCE_SCORE = (
    -6.311376452445984
)


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
    """
    Claim-level question relevance filter.

    Pipeline:

        supported claims
            ↓
        deterministic malformed guard
            ↓
        BGE cross-encoder ranking
            ↓
        calibrated catastrophic relevance floor
            ↓
        top-k surviving claims

    IMPORTANT:

    min_relevance_score is a RAW BGE reranker logit,
    not a probability.

    The default floor was calibrated to reject only
    catastrophic low-relevance claims while preserving
    every manually audited positive calibration example.
    """

    def __init__(
        self,
        reranker,
        max_relevant_claims: int = 2,
        min_relevance_score: float = (
            DEFAULT_MIN_RELEVANCE_SCORE
        ),
    ):

        if (
            max_relevant_claims
            <
            1
        ):

            raise ValueError(
                "max_relevant_claims must be >= 1."
            )


        self.reranker = (
            reranker
        )


        self.max_relevant_claims = (
            max_relevant_claims
        )


        self.min_relevance_score = float(
            min_relevance_score
        )


    # ========================================================
    # Deterministic malformed-output guard
    #
    # Some model failures produce placeholder-like claims:
    #
    #   [TechCrunch article mentions ...]
    #
    # These are generation artifacts rather than proper
    # factual statements.
    #
    # Do NOT make this guard aggressive.
    # ========================================================

    @staticmethod
    def _is_malformed_claim(
        text: str,
    ) -> bool:

        normalized = (
            " ".join(
                (text or "").split()
            )
        )


        if not normalized:

            return True


        # Entire claim is enclosed in square brackets.
        #
        # Example:
        #
        # [TechCrunch article mentions Google's antitrust suit]
        #
        if re.fullmatch(
            r"\[[^\[\]]+\]",
            normalized,
        ):

            return True


        return False


    # ========================================================
    # Convert scored item to public result
    # ========================================================

    @staticmethod
    def _to_result(
        item,
    ) -> RelevantClaim:

        original_claim = (
            item[
                "claim"
            ]
        )


        return RelevantClaim(

            claim=(
                original_claim.claim
            ),

            citation_id=(
                original_claim.citation_id
            ),

            relevance_score=round(
                float(
                    item[
                        "rerank_score"
                    ]
                ),
                4,
            ),
        )


    # ========================================================
    # Main filtering
    # ========================================================

    def filter(
        self,
        query: str,
        grounded_claims,
    ) -> RelevanceFilterResult:

        supported_claims = [

            claim

            for claim
            in grounded_claims.claims

            if claim.supported
        ]


        if not supported_claims:

            return RelevanceFilterResult(

                relevant_claims=[],

                filtered_claims=[],

                total_claims=0,
            )


        # ====================================================
        # Stage 1
        # Remove deterministic malformed generation artifacts
        # before asking BGE to rank them.
        # ====================================================

        valid_supported_claims = []

        malformed_claims = []


        for claim in (
            supported_claims
        ):

            if self._is_malformed_claim(
                claim.claim
            ):

                malformed_claims.append(
                    RelevantClaim(

                        claim=(
                            claim.claim
                        ),

                        citation_id=(
                            claim.citation_id
                        ),

                        # No BGE score exists because the
                        # malformed claim is rejected before
                        # model scoring.
                        relevance_score=float(
                            "-inf"
                        ),
                    )
                )

                continue


            valid_supported_claims.append(
                claim
            )


        if not valid_supported_claims:

            return RelevanceFilterResult(

                relevant_claims=[],

                filtered_claims=(
                    malformed_claims
                ),

                total_claims=len(
                    supported_claims
                ),
            )


        # ====================================================
        # Stage 2
        # Convert claims to BGEReranker format.
        # ====================================================

        documents = []


        for (
            index,
            claim,
        ) in enumerate(
            valid_supported_claims
        ):

            documents.append(
                {
                    "id":
                        f"claim_{index}",

                    "text":
                        claim.claim,

                    "claim":
                        claim,
                }
            )


        # ====================================================
        # Stage 3
        # Cross-encoder relevance ranking.
        # ====================================================

        ranked = (
            self.reranker.rerank(

                query=query,

                documents=documents,

                top_k=len(
                    documents
                ),
            )
        )


        relevant_claims = []

        filtered_claims = list(
            malformed_claims
        )


        # ====================================================
        # Stage 4
        #
        # First apply calibrated safety floor.
        #
        # THEN apply max_relevant_claims.
        #
        # Important:
        # Do not use the original ranked index for top-k.
        # A filtered claim must not consume one of the
        # surviving top-k slots.
        # ====================================================

        for item in ranked:

            raw_score = float(
                item[
                    "rerank_score"
                ]
            )


            result = (
                self._to_result(
                    item
                )
            )


            # ------------------------------------------------
            # Catastrophic relevance floor
            # ------------------------------------------------

            if (
                raw_score
                <
                self.min_relevance_score
            ):

                filtered_claims.append(
                    result
                )

                continue


            # ------------------------------------------------
            # Top-k among claims that SURVIVED the floor
            # ------------------------------------------------

            if (
                len(
                    relevant_claims
                )
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
            ),
        )