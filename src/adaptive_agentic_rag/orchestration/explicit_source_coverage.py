import re

from dataclasses import dataclass


@dataclass
class ExplicitSourceCoverageResult:
    satisfied: bool
    required_sources: list[str]
    available_sources: list[str]
    covered_sources: list[str]
    missing_sources: list[str]


class ExplicitSourceCoverageGuard:
    """
    Hard structural evidence guard.

    If a query explicitly requires evidence from named
    publishers/sources, those sources must be represented
    in the built context.

    This component checks source presence only.

    It does NOT decide semantic support.
    """

    # ========================================================
    # Normalization
    # ========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return " ".join(
            re.findall(
                r"[a-z0-9]+",
                (text or "").lower(),
            )
        )


    # ========================================================
    # Source aliases
    # ========================================================

    @classmethod
    def _source_aliases(
        cls,
        source: str,
    ) -> set[str]:

        source = (
            source
            or ""
        ).strip()


        if not source:
            return set()


        # Example:
        #
        # Cnbc | World Business News Leader
        #
        # →
        #
        # full identity
        # primary identity = Cnbc

        primary = (
            source
            .split(
                "|",
                1,
            )[0]
            .strip()
        )


        aliases = {
            cls._normalize(
                source
            ),

            cls._normalize(
                primary
            ),
        }


        expanded = set(
            aliases
        )


        for alias in aliases:

            if alias.startswith(
                "the "
            ):

                without_the = (
                    alias[4:]
                    .strip()
                )


                # Safe:
                #
                # The Sydney Morning Herald
                # → Sydney Morning Herald
                #
                # Unsafe:
                #
                # The Age
                # → Age

                if (
                    len(
                        without_the.split()
                    )
                    >=
                    2
                ):

                    expanded.add(
                        without_the
                    )


        return {
            alias
            for alias in expanded
            if alias
        }


    # ========================================================
    # Candidate cleanup
    # ========================================================

    @staticmethod
    def _clean_candidate(
        candidate: str,
    ) -> str:

        candidate = (
            candidate
            or ""
        ).strip()


        candidate = candidate.strip(
            " \t\r\n"
            "\"'“”‘’"
            ".,;:()[]{}"
        )


        candidate = re.sub(
            r"\s+",
            " ",
            candidate,
        )


        # ----------------------------------------------------
        # Remove question prefixes.
        #
        # Does the Fortune
        # → Fortune
        # ----------------------------------------------------

        candidate = re.sub(
            (
                r"^(?:"
                r"does|did|has|have|had|"
                r"was|were|is|are|"
                r"can|could|would|should|will"
                r")\s+the\s+"
            ),
            "",
            candidate,
            flags=re.IGNORECASE,
        )


        # ----------------------------------------------------
        # Remove discourse/connective prefixes.
        #
        # while The Sydney Morning Herald
        # → The Sydney Morning Herald
        #
        # and TechCrunch
        # → TechCrunch
        # ----------------------------------------------------

        candidate = re.sub(
            (
                r"^(?:"
                r"while|whereas|but|"
                r"and with|and|"
                r"another"
                r")\s+"
            ),
            "",
            candidate,
            flags=re.IGNORECASE,
        )


        return candidate.strip(
            " \"'“”‘’.,;:"
        )


    # ========================================================
    # Candidate plausibility
    #
    # This prevents grammar around an article/report mention
    # from being mistaken for a publisher.
    #
    # Example:
    #
    # "Between the report from 'The Roar' ..."
    #
    # A permissive regex may also capture:
    #
    #     "Between the"
    #
    # before the word "report".
    #
    # That is grammatical structure, NOT a source.
    # ========================================================

    @classmethod
    def _is_plausible_source_candidate(
        cls,
        candidate: str,
    ) -> bool:

        candidate = (
            candidate
            or ""
        ).strip()


        if not candidate:
            return False


        normalized = (
            cls._normalize(
                candidate
            )
        )


        if not normalized:
            return False


        words = (
            normalized.split()
        )


        if not words:
            return False


        # Publisher/source names in this dataset are compact.
        # A large captured phrase means regex spillover.

        if len(words) > 10:
            return False


        # ----------------------------------------------------
        # Grammar/question/discourse words cannot begin a
        # publisher candidate after cleanup.
        # ----------------------------------------------------

        forbidden_leading_words = {
            "between",
            "does",
            "do",
            "did",
            "has",
            "have",
            "had",
            "is",
            "are",
            "was",
            "were",
            "can",
            "could",
            "would",
            "should",
            "will",
            "which",
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "while",
            "whereas",
            "considering",
            "after",
            "before",
            "and",
            "but",
            "or",
        }


        if (
            words[0]
            in
            forbidden_leading_words
        ):
            return False


        # ----------------------------------------------------
        # Captures ending with an article/determiner are
        # almost certainly incomplete grammar fragments.
        #
        # "Between the"
        # "Does the"
        # ----------------------------------------------------

        if words[-1] in {
            "the",
            "a",
            "an",
        }:
            return False


        # ----------------------------------------------------
        # Generic reporting words alone are not publishers.
        # ----------------------------------------------------

        generic_only = {
            "article",
            "report",
            "reports",
            "coverage",
            "reporting",
            "update",
            "updates",
            "source",
            "news",
        }


        if (
            len(words)
            ==
            1
            and
            words[0]
            in
            generic_only
        ):
            return False


        return True


    # ========================================================
    # Candidate registration
    #
    # Centralizes cleanup + validation so every regex path
    # receives the same protection.
    # ========================================================

    @classmethod
    def _add_candidate(
        cls,
        candidates: list[str],
        candidate: str,
    ) -> None:

        candidate = (
            cls._clean_candidate(
                candidate
            )
        )


        if not (
            cls._is_plausible_source_candidate(
                candidate
            )
        ):
            return


        candidates.append(
            candidate
        )


    # ========================================================
    # Explicit source extraction
    # ========================================================

    @classmethod
    def _extract_candidates(
        cls,
        query: str,
    ) -> list[str]:

        query = (
            query
            or ""
        )


        if not query.strip():

            return []


        candidates = []


        # ====================================================
        # Common grammatical words that terminate a source
        # phrase.
        #
        # Examples:
        #
        # reports by The Age remained consistent ...
        # article from TechCrunch focuses on ...
        # report by Reuters showed ...
        #
        # Without these terminators:
        #
        # "The Age remained consistent"
        #
        # could either fail extraction or be captured as part
        # of the publisher name.
        # ====================================================

        source_terminators = (
            r"on|after|before|about|regarding|"
            r"suggesting|suggested|suggests|"
            r"detailing|detailed|"
            r"focusing|focused|focuses|"
            r"discussing|discussed|discusses|"
            r"covering|covered|covers|"
            r"indicating|indicated|indicates|"
            r"showing|showed|shows|"
            r"mentioning|mentioned|mentions|"
            r"describing|described|describes|"
            r"remaining|remained|remains|remain|"
            r"being|"
            r"which|who|that|while|whereas|"
            r"was|were|is|are|"
            r"has|have|had"
        )


        # ====================================================
        # A)
        #
        # report from The Roar
        # article by Forbes
        # reports by The Age remained ...
        # coverage from TechCrunch
        # ====================================================

        after_marker_pattern = re.compile(
            (
                r"\b"
                r"(?:"
                r"article|articles|"
                r"report|reports|"
                r"coverage|reporting|"
                r"update|updates"
                r")"
                r"\s+"
                r"(?:from|by)"
                r"\s+"
                r"(?P<source>"
                r"[^,;?]{1,100}?"
                r")"
                r"(?="
                r",|;|\?|$|"
                r"\s+(?:"
                +
                source_terminators
                +
                r")\b"
                r")"
            ),
            flags=re.IGNORECASE,
        )


        for match in (
            after_marker_pattern
            .finditer(
                query
            )
        ):

            cls._add_candidate(
                candidates,
                match.group(
                    "source"
                ),
            )


        # ====================================================
        # B)
        #
        # according to Reuters,
        # according to CNBC on ...
        # ====================================================

        according_pattern = re.compile(
            (
                r"\b"
                r"according\s+to\s+"
                r"(?P<source>"
                r"[^,;?]{1,100}?"
                r")"
                r"(?="
                r",|;|\?|$|"
                r"\s+(?:"
                +
                source_terminators
                +
                r")\b"
                r")"
            ),
            flags=re.IGNORECASE,
        )


        for match in (
            according_pattern
            .finditer(
                query
            )
        ):

            cls._add_candidate(
                candidates,
                match.group(
                    "source"
                ),
            )


        # ====================================================
        # C)
        #
        # The Verge's coverage
        # TechCrunch's report
        # Fortune's reporting
        # ====================================================

        possessive_pattern = re.compile(
            (
                r"(?P<source>"
                r"(?:"
                r"[A-Z][A-Za-z0-9&|.-]*"
                r"(?:"
                r"\s+|"
                r"\s*\|\s*|"
                r"\s*-\s*"
                r")"
                r"){0,7}"
                r"[A-Z][A-Za-z0-9&|.-]*"
                r")"
                r"[’']s"
                r"\s+"
                r"(?:"
                r"article|report|coverage|"
                r"reporting|update"
                r")"
                r"\b"
            )
        )


        for match in (
            possessive_pattern
            .finditer(
                query
            )
        ):

            cls._add_candidate(
                candidates,
                match.group(
                    "source"
                ),
            )


        # ====================================================
        # D)
        #
        # Fortune article suggests ...
        # The Guardian article ...
        #
        # _add_candidate() protects us from grammatical
        # fragments such as:
        #
        # Between the report ...
        # → "Between the" is rejected.
        # ====================================================

        before_marker_pattern = re.compile(
            (
                r"(?:^|[,;])"
                r"\s*"
                r"(?P<source>"
                r"[^,;?]{1,100}?"
                r")"
                r"\s+"
                r"(?:article|report)"
                r"\b"
            ),
            flags=re.IGNORECASE,
        )


        for match in (
            before_marker_pattern
            .finditer(
                query
            )
        ):

            cls._add_candidate(
                candidates,
                match.group(
                    "source"
                ),
            )


        # ====================================================
        # E)
        #
        # 'The Guardian' article
        # 'The Roar | Sports Writers Blog' report
        # ====================================================

        quoted_pattern = re.compile(
            (
                r"[\"'“”‘’]"
                r"(?P<source>"
                r"[^\"'“”‘’]{2,100}"
                r")"
                r"[\"'“”‘’]"
                r"\s+"
                r"(?:"
                r"article|report|coverage"
                r")"
                r"\b"
            ),
            flags=re.IGNORECASE,
        )


        for match in (
            quoted_pattern
            .finditer(
                query
            )
        ):

            cls._add_candidate(
                candidates,
                match.group(
                    "source"
                ),
            )


        # ====================================================
        # Stable deduplication
        # ====================================================

        output = []

        seen = set()


        for candidate in candidates:

            key = (
                cls._normalize(
                    candidate
                )
            )


            if not key:

                continue


            if key in seen:

                continue


            seen.add(
                key
            )


            output.append(
                candidate
            )


        return output


    # ========================================================
    # Available context sources
    # ========================================================

    @staticmethod
    def _available_sources(
        context,
    ) -> list[str]:

        if context is None:
            return []


        items = (
            getattr(
                context,
                "items",
                [],
            )
            or []
        )


        output = []

        seen = set()


        for item in items:

            source = (
                getattr(
                    item,
                    "source",
                    "",
                )
                or ""
            ).strip()


            if not source:
                continue


            key = (
                source.lower()
            )


            if key in seen:
                continue


            seen.add(
                key
            )


            output.append(
                source
            )


        return output


    # ========================================================
    # Candidate → actual context source
    # ========================================================

    @classmethod
    def _match_available_source(
        cls,
        *,
        candidate: str,
        available_sources: list[str],
    ) -> str | None:

        candidate_normalized = (
            cls._normalize(
                candidate
            )
        )


        if not candidate_normalized:
            return None


        matches = []


        for source in available_sources:

            for alias in (
                cls._source_aliases(
                    source
                )
            ):

                if not alias:
                    continue


                # Exact source identity.

                if (
                    candidate_normalized
                    ==
                    alias
                ):

                    matches.append(
                        (
                            len(alias),
                            source,
                        )
                    )

                    continue


                candidate_words = (
                    candidate_normalized.split()
                )


                alias_words = (
                    alias.split()
                )


                # Allow a small amount of grammatical capture
                # around an otherwise exact publisher alias.

                if (
                    len(alias_words)
                    >=
                    1
                    and
                    len(candidate_words)
                    <=
                    len(alias_words)
                    +
                    3
                ):

                    pattern = (
                        r"(?:^|\s)"
                        +
                        re.escape(
                            alias
                        )
                        +
                        r"(?:$|\s)"
                    )


                    if re.search(
                        pattern,
                        candidate_normalized,
                    ):

                        matches.append(
                            (
                                len(alias),
                                source,
                            )
                        )


        if not matches:
            return None


        # Longest alias wins.

        matches.sort(
            reverse=True
        )


        return (
            matches[0][1]
        )


    # ========================================================
    # Detect available context source mentioned anywhere
    # in query.
    # ========================================================

    @classmethod
    def _query_mentions_available_source(
        cls,
        *,
        query: str,
        source: str,
    ) -> bool:

        normalized_query = (
            cls._normalize(
                query
            )
        )


        for alias in (
            cls._source_aliases(
                source
            )
        ):

            if not alias:
                continue


            pattern = (
                r"(?:^|\s)"
                +
                re.escape(
                    alias
                )
                +
                r"(?:$|\s)"
            )


            if re.search(
                pattern,
                normalized_query,
            ):

                return True


        return False


    # ========================================================
    # Stable source-list deduplication
    # ========================================================

    @classmethod
    def _deduplicate(
        cls,
        values: list[str],
    ) -> list[str]:

        output = []

        seen = set()


        for value in values:

            key = (
                cls._normalize(
                    value
                )
            )


            if not key:
                continue


            if key in seen:
                continue


            seen.add(
                key
            )


            output.append(
                value
            )


        return output


    # ========================================================
    # Main guard
    # ========================================================

    def check(
        self,
        *,
        query: str,
        context,
    ) -> ExplicitSourceCoverageResult:

        available_sources = (
            self._available_sources(
                context
            )
        )


        extracted = (
            self._extract_candidates(
                query
            )
        )


        required_sources = []

        covered_sources = []

        missing_sources = []


        # ====================================================
        # Resolve syntactically extracted source references.
        # ====================================================

        for candidate in extracted:

            matched = (
                self._match_available_source(
                    candidate=
                        candidate,

                    available_sources=
                        available_sources,
                )
            )


            if matched is not None:

                required_sources.append(
                    matched
                )


                covered_sources.append(
                    matched
                )


            else:

                required_sources.append(
                    candidate
                )


                missing_sources.append(
                    candidate
                )


        # ====================================================
        # Add known context sources explicitly appearing
        # anywhere in the query.
        #
        # Since they already exist in context, they can only
        # increase required/covered coverage, never create a
        # false missing source.
        # ====================================================

        for source in available_sources:

            if (
                self._query_mentions_available_source(
                    query=
                        query,

                    source=
                        source,
                )
            ):

                required_sources.append(
                    source
                )


                covered_sources.append(
                    source
                )


        required_sources = (
            self._deduplicate(
                required_sources
            )
        )


        covered_sources = (
            self._deduplicate(
                covered_sources
            )
        )


        missing_sources = (
            self._deduplicate(
                missing_sources
            )
        )


        satisfied = (
            len(
                missing_sources
            )
            ==
            0
        )


        return ExplicitSourceCoverageResult(
            satisfied=
                satisfied,

            required_sources=
                required_sources,

            available_sources=
                available_sources,

            covered_sources=
                covered_sources,

            missing_sources=
                missing_sources,
        )