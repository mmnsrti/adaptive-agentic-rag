import re


class QueryRewriter:

    def __init__(self):
        pass


    def _normalize(
        self,
        query: str
    ) -> str:

        return " ".join(
            query.strip().split()
        )


    def _rewrite_comparison(
        self,
        query: str
    ) -> str:

        text = self._normalize(
            query
        )

        #
        # Example:
        #
        # Compare Amazon and AcmeMart loyalty cashback
        # shipping deals
        #
        # ->
        #
        # Amazon and AcmeMart loyalty cashback
        # shipping deals comparison
        #

        text = re.sub(
            r"^\s*compare\s+",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"^\s*comparison\s+(of|between)\s+",
            "",
            text,
            flags=re.IGNORECASE
        )

        if not re.search(
            r"\bcomparison\b",
            text,
            flags=re.IGNORECASE
        ):

            text = (
                text.rstrip(" ?.")
                +
                " comparison"
            )

        return self._normalize(
            text
        )


    def _rewrite_reasoning(
        self,
        query: str
    ) -> str:

        text = self._normalize(
            query
        )

        #
        # Keep the entities and important terms,
        # but make the retrieval intent more explicit.
        #

        if re.match(
            r"^\s*why\b",
            text,
            flags=re.IGNORECASE
        ):

            text = re.sub(
                r"^\s*why\b",
                "",
                text,
                count=1,
                flags=re.IGNORECASE
            )

            return self._normalize(
                f"{text} reasons explanation"
            )


        if re.match(
            r"^\s*how\b",
            text,
            flags=re.IGNORECASE
        ):

            return self._normalize(
                f"{text} explanation"
            )


        return self._normalize(
            f"{text} supporting evidence"
        )


    def _rewrite_complex(
        self,
        query: str
    ) -> str:

        text = self._normalize(
            query
        )

        if re.search(
            r"\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b",
            text,
            flags=re.IGNORECASE
        ):

            return self._rewrite_comparison(
                text
            )


        return self._normalize(
            f"{text} relevant facts evidence"
        )


    def rewrite(
        self,
        query: str,
        query_type: str,
        attempt: int = 1
    ) -> str:

        text = self._normalize(
            query
        )


        if not text:

            return text


        query_type = (
            query_type
            or ""
        ).strip().lower()


        #
        # Comparison queries are common in our
        # MultiHopRAG workload. Detect them even
        # when the router labels them multihop.
        #

        if re.search(
            r"^\s*compare\b|\bcomparison\b|\bversus\b|\bvs\.?\b",
            text,
            flags=re.IGNORECASE
        ):

            rewritten = (
                self._rewrite_comparison(
                    text
                )
            )


        elif query_type == "complex":

            rewritten = (
                self._rewrite_complex(
                    text
                )
            )


        elif query_type == "multihop":

            rewritten = (
                self._rewrite_reasoning(
                    text
                )
            )


        else:

            #
            # Simple queries usually do not need
            # aggressive rewriting.
            #

            rewritten = text


        #
        # A later retry may add a small retrieval hint,
        # while keeping the original entities intact.
        #

        if (
            attempt > 1
            and
            rewritten == text
        ):

            rewritten = self._normalize(
                f"{rewritten} relevant information"
            )


        return rewritten