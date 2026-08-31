import ast

from dataclasses import dataclass
from enum import Enum


class RetryAction(
    str,
    Enum,
):

    GENERATE = "generate"

    RETRY = "retry"

    ABSTAIN = "abstain"


@dataclass
class RetryDecision:

    action: RetryAction

    reason: str

    evidence_path: str | None

    required_sources: list[str]

    covered_sources: list[str]

    missing_sources: list[str]

    retry_count: int


class AdaptiveRetryPolicy:
    """
    Conservative retry policy.

    Goal
    ----
    Retry retrieval only when the current evidence state
    contains a structural signal that another retrieval
    attempt may be useful.

    This V1 intentionally prefers precision over retry recall.

    Safe rules
    ----------

    1. Evidence sufficient
       -> GENERATE

    2. Retry budget exhausted
       -> ABSTAIN

    3. Explicit source coverage rejection with PARTIAL
       source coverage
       -> RETRY

    4. Explicit source coverage rejection with ZERO
       requested sources covered
       -> ABSTAIN

    5. Evidence/source coverage already satisfied but
       constrained semantic rescue still rejects
       -> ABSTAIN

    6. Unknown rejection state
       -> ABSTAIN

    Important
    ---------
    V1 does NOT use:

    - gold evidence
    - dataset labels
    - evidence-score thresholds
    - embedding thresholds
    - LLM classification

    It only consumes production-visible evidence telemetry.
    """

    def __init__(
        self,
        *,
        max_retries: int = 1,
    ):

        self.max_retries = (
            max_retries
        )


    # ========================================================
    # Telemetry helpers
    # ========================================================

    @staticmethod
    def _extract_value(
        reasons: list[str],
        prefix: str,
    ) -> str | None:

        for reason in (
            reasons
            or []
        ):

            if not isinstance(
                reason,
                str,
            ):

                continue


            if reason.startswith(
                prefix
            ):

                return (
                    reason[
                        len(
                            prefix
                        ):
                    ]
                    .strip()
                )


        return None


    @classmethod
    def _extract_list(
        cls,
        reasons: list[str],
        prefix: str,
    ) -> list[str]:

        raw_value = (
            cls._extract_value(
                reasons=
                    reasons,

                prefix=
                    prefix,
            )
        )


        if raw_value is None:

            return []


        try:

            parsed = (
                ast.literal_eval(
                    raw_value
                )
            )

        except (
            ValueError,
            SyntaxError,
        ):

            return []


        if not isinstance(
            parsed,
            list,
        ):

            return []


        return [
            str(
                item
            )

            for item
            in parsed
        ]


    @classmethod
    def _extract_path(
        cls,
        reasons: list[str],
    ) -> str | None:

        return (
            cls._extract_value(
                reasons=
                    reasons,

                prefix=
                    "evidence_path=",
            )
        )


    # ========================================================
    # Public policy
    # ========================================================

    def decide(
        self,
        *,
        evidence_sufficient: bool,
        retry_count: int,
        evidence_reasons: list[str],
    ) -> RetryDecision:

        evidence_path = (
            self._extract_path(
                evidence_reasons
            )
        )


        required_sources = (
            self._extract_list(
                evidence_reasons,
                "required_sources=",
            )
        )


        covered_sources = (
            self._extract_list(
                evidence_reasons,
                "covered_sources=",
            )
        )


        missing_sources = (
            self._extract_list(
                evidence_reasons,
                "missing_sources=",
            )
        )


        # ====================================================
        # Evidence already accepted
        # ====================================================

        if evidence_sufficient:

            return RetryDecision(
                action=
                    RetryAction.GENERATE,

                reason=(
                    "Evidence is already sufficient."
                ),

                evidence_path=
                    evidence_path,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,

                retry_count=
                    retry_count,
            )


        # ====================================================
        # Retry budget exhausted
        # ====================================================

        if (
            retry_count
            >=
            self.max_retries
        ):

            return RetryDecision(
                action=
                    RetryAction.ABSTAIN,

                reason=(
                    "Retry budget exhausted."
                ),

                evidence_path=
                    evidence_path,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,

                retry_count=
                    retry_count,
            )


        # ====================================================
        # Explicit source coverage failure
        # ====================================================

        if (
            evidence_path
            ==
            "explicit_source_coverage_reject"
        ):

            # ------------------------------------------------
            # Some required source evidence exists, while
            # another explicitly requested source is absent.
            #
            # This is the clearest retrieval-recoverable
            # signal currently available.
            # ------------------------------------------------

            if (
                missing_sources
                and
                covered_sources
            ):

                return RetryDecision(
                    action=
                        RetryAction.RETRY,

                    reason=(
                        "Partial explicit-source coverage "
                        "suggests a recoverable retrieval miss."
                    ),

                    evidence_path=
                        evidence_path,

                    required_sources=
                        required_sources,

                    covered_sources=
                        covered_sources,

                    missing_sources=
                        missing_sources,

                    retry_count=
                        retry_count,
                )


            # ------------------------------------------------
            # No explicitly requested source was found.
            #
            # Current Smoke diagnostics provide no evidence
            # that blindly rewriting such cases helps.
            #
            # Null queries also frequently exhibit this shape.
            # ------------------------------------------------

            return RetryDecision(
                action=
                    RetryAction.ABSTAIN,

                reason=(
                    "Explicit-source coverage failed without "
                    "partial source support; blind retry is "
                    "not justified."
                ),

                evidence_path=
                    evidence_path,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,

                retry_count=
                    retry_count,
            )


        # ====================================================
        # Source coverage passed, semantic rescue failed.
        #
        # Cases with complete gold context already showed
        # that re-running retrieval here can be pointless.
        # ====================================================

        if (
            evidence_path
            ==
            "constrained_semantic_rescue_reject"
        ):

            return RetryDecision(
                action=
                    RetryAction.ABSTAIN,

                reason=(
                    "Evidence remained insufficient after "
                    "source coverage and constrained semantic "
                    "rescue; no structural retrieval-miss "
                    "signal is present."
                ),

                evidence_path=
                    evidence_path,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,

                retry_count=
                    retry_count,
            )


        # ====================================================
        # Conservative fallback
        # ====================================================

        return RetryDecision(
            action=
                RetryAction.ABSTAIN,

            reason=(
                "No production-visible signal justifies "
                "another retrieval attempt."
            ),

            evidence_path=
                evidence_path,

            required_sources=
                required_sources,

            covered_sources=
                covered_sources,

            missing_sources=
                missing_sources,

            retry_count=
                retry_count,
        )