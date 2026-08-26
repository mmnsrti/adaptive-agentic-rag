import re


class QueryRewriter:

    def _normalize(
        self,
        query: str
    ) -> str:

        query = query.strip()

        query = re.sub(
            r"\s+",
            " ",
            query
        )

        return query


    def rewrite(
        self,
        query: str,
        query_type: str,
        attempt: int = 1
    ) -> str:

        query = self._normalize(
            query
        )


        #
        # Multi-hop / comparison queries
        #

        if query_type == "multihop":

            rewritten = re.sub(
                r"^\s*compare\s+",
                "",
                query,
                flags=re.IGNORECASE
            )

            rewritten = re.sub(
                r"\b(vs\.?|versus)\b",
                " ",
                rewritten,
                flags=re.IGNORECASE
            )

            rewritten = re.sub(
                r"\s+",
                " ",
                rewritten
            ).strip()


            if attempt == 1:

                return (
                    f"{rewritten} comparison"
                )


            return (
                f"{rewritten} comparison "
                f"differences similarities"
            )


        #
        # Complex queries
        #

        if query_type == "complex":

            rewritten = re.sub(
                r"^\s*"
                r"(summarize|analyse|analyze|explain)"
                r"\s+",
                "",
                query,
                flags=re.IGNORECASE
            )

            rewritten = rewritten.strip()


            if attempt == 1:

                return (
                    f"{rewritten} detailed information"
                )


            return (
                f"{rewritten} key facts "
                f"details sources"
            )


        #
        # Simple query fallback
        #

        if attempt == 1:

            return (
                f"{query} detailed information"
            )


        return (
            f"{query} relevant facts evidence"
        )