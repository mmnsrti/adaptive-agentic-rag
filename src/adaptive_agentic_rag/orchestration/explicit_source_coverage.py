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

            if not alias.startswith(
                "the "
            ):

                continue


            without_the = (
                alias[
                    4:
                ]
                .strip()
            )


            words = (
                without_the.split()
            )


            # ------------------------------------------------
            # Safe multi-token aliases:
            #
            # The Sydney Morning Herald
            # -> Sydney Morning Herald
            # ------------------------------------------------

            if (
                len(
                    words
                )
                >=
                2
            ):

                expanded.add(
                    without_the
                )


            # ------------------------------------------------
            # Allow distinctive long single-token publisher
            # names.
            #
            # The Guardian -> Guardian
            # The Verge    -> Verge
            #
            # But:
            #
            # The Age -> NO "Age"
            # ------------------------------------------------

            elif (
                len(
                    words
                )
                ==
                1
                and
                len(
                    words[
                        0
                    ]
                )
                >=
                5
            ):

                expanded.add(
                    without_the
                )


        return {
            alias

            for alias
            in expanded

            if alias
        }


    # ========================================================
    # Referential phrases
    #
    # These describe another article/source grammatically.
    # They are NOT publisher names.
    # ========================================================

    @classmethod
    def _is_reference_phrase(
        cls,
        candidate: str,
    ) -> bool:

        normalized = (
            cls._normalize(
                candidate
            )
        )


        if not normalized:

            return True


        # ----------------------------------------------------
        # Strong prefix references.
        #
        # Important:
        #
        # This intentionally rejects the ENTIRE captured
        # phrase, not just "both".
        #
        # Example:
        #
        # according to both sources played the lead guitarist
        #
        # regex candidate:
        #
        #     both sources played the lead guitarist
        #
        # This is grammatical spillover, not a publisher.
        # ----------------------------------------------------

        if re.match(
            (
                r"^(?:"
                r"both|these|those"
                r")"
                r"\s+"
                r"(?:news\s+)?"
                r"sources?"
                r"\b"
            ),
            normalized,
            flags=re.IGNORECASE,
        ):

            return True


        if re.match(
            (
                r"^(?:the\s+)?"
                r"same"
                r"\s+"
                r"(?:news\s+)?"
                r"source"
                r"\b"
            ),
            normalized,
            flags=re.IGNORECASE,
        ):

            return True


        patterns = [

            # the subsequent
            # subsequent report

            (
                r"^(?:the\s+)?"
                r"subsequent"
                r"(?:\s+(?:article|report|coverage|update))?"
                r"$"
            ),

            # the other
            # other article

            (
                r"^(?:the\s+)?"
                r"other"
                r"(?:\s+(?:article|report|coverage|update))?"
                r"$"
            ),

            # the earlier / their subsequent

            (
                r"^(?:"
                r"the|their"
                r")\s+"
                r"(?:"
                r"earlier|later|subsequent"
                r")"
                r"(?:\s+(?:article|report|coverage|update))?"
                r"$"
            ),

            # according to another

            (
                r"^(?:"
                r"according\s+to\s+"
                r")?"
                r"another"
                r"$"
            ),

            # the information collected

            (
                r"^the\s+"
                r"(?:"
                r"information|data|details|findings"
                r")"
                r"\b.*$"
            ),

            # contradict the earlier
            # compared to their subsequent

            (
                r"^(?:"
                r"contradict|contradicts|contradicted|"
                r"compare|compared"
                r")"
                r"\b.*"
                r"(?:"
                r"earlier|later|subsequent"
                r")"
                r"$"
            ),
        ]


        return any(
            re.fullmatch(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )
            is not None

            for pattern
            in patterns
        )


    # ========================================================
    # Candidate cleanup
    # ========================================================

    @classmethod
    def _clean_candidate(
        cls,
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


        # ====================================================
        # Reject reference phrases BEFORE removing words such
        # as "both".
        #
        # This fixes:
        #
        # both sources played ...
        # these sources ...
        # the same source ...
        # ====================================================

        if (
            cls._is_reference_phrase(
                candidate
            )
        ):

            return ""


        # ====================================================
        # Remove question prefixes
        # ====================================================

        candidate = re.sub(
            (
                r"^(?:"
                r"does|do|did|"
                r"has|have|had|"
                r"was|were|is|are|"
                r"can|could|would|should|will"
                r")"
                r"\s+"
                r"(?:the\s+)?"
            ),
            "",
            candidate,
            flags=re.IGNORECASE,
        )


        # ====================================================
        # Remove discourse prefixes
        # ====================================================

        candidate = re.sub(
            (
                r"^(?:"
                r"while|whereas|but|"
                r"and with|and|"
                r"in contrast to|"
                r"compared to|"
                r"compared with|"
                r"versus|vs\.?"
                r")"
                r"\s+"
                r"(?:the\s+)?"
            ),
            "",
            candidate,
            flags=re.IGNORECASE,
        )


        # ====================================================
        # Remove source-list wrappers
        # ====================================================

        wrapper_patterns = [

            (
                r"^(?:"
                r"different\s+"
                r")?"
                r"(?:"
                r"articles?|"
                r"reports?|"
                r"coverage|"
                r"reporting|"
                r"updates?"
                r")"
                r"\s+"
                r"(?:from|by)"
                r"\s+"
            ),

            (
                r"^(?:a|an)"
                r"\s+"
                r"(?:analysis|study)"
                r"\s+"
                r"(?:from|by)"
                r"\s+"
            ),

            (
                r"^(?:a|an)"
                r"\s+"
                r"(?:analysis|study)"
                r"\s+"
                r"reported\s+by\s+"
                r"(?:sources?\s+)?"
                r"including"
                r"\s+"
            ),

            (
                r"^sources?\s+including\s+"
            ),

            (
                r"^according\s+to\s+"
            ),
        ]


        for _ in range(
            4
        ):

            before = (
                candidate
            )


            for pattern in (
                wrapper_patterns
            ):

                candidate = re.sub(
                    pattern,
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                )


            if (
                candidate
                ==
                before
            ):

                break


        candidate = candidate.strip()


        # ====================================================
        # Check reference phrases AGAIN after wrapper cleanup.
        #
        # Example:
        #
        # according to both sources
        # -> both sources
        # -> reject
        # ====================================================

        if (
            cls._is_reference_phrase(
                candidate
            )
        ):

            return ""


        # ====================================================
        # Leading "both" is safe only when followed by actual
        # publisher candidates:
        #
        # both Hacker News and Zee Business
        # -> Hacker News and Zee Business
        #
        # "both sources ..." was already rejected above.
        # ====================================================

        candidate = re.sub(
            r"^both\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )


        # ====================================================
        # Trim descriptive tails accidentally attached to a
        # publisher identity.
        # ====================================================

        trailing_terminators = (
            r"between|"
            r"published|"
            r"defending|defended|defends|"
            r"claiming|claimed|claims|"
            r"suggesting|suggested|suggests|"
            r"indicating|indicated|indicates|"
            r"discussing|discussed|discusses|"
            r"detailing|detailed|details|"
            r"focusing|focused|focuses|"
            r"covering|covered|covers|"
            r"showing|showed|shows|"
            r"mentioning|mentioned|mentions|"
            r"describing|described|describes|"
            r"arguing|argued|argues|"
            r"stating|stated|states|"
            r"explaining|explained|explains|"
            r"contrasting|contrasted|contrasts|"
            r"contradicting|contradicted|contradicts|"
            r"remaining|remained|remains|remain"
        )


        candidate = re.sub(
            (
                r"\s+"
                r"(?:"
                +
                trailing_terminators
                +
                r")"
                r"\b.*$"
            ),
            "",
            candidate,
            flags=re.IGNORECASE,
        )


        candidate = re.sub(
            (
                r"\s+both\s+"
                r"(?:"
                r"suggest|suggests|suggested|"
                r"indicate|indicates|indicated|"
                r"report|reports|reported|"
                r"discuss|discusses|discussed"
                r")"
                r"\b.*$"
            ),
            "",
            candidate,
            flags=re.IGNORECASE,
        )


        candidate = candidate.strip(
            " \t\r\n"
            "\"'“”‘’"
            ".,;:()[]{}"
        )


        candidate = re.sub(
            r"[’']s$",
            "",
            candidate,
            flags=re.IGNORECASE,
        )


        candidate = re.sub(
            r"\s+",
            " ",
            candidate,
        )


        return candidate.strip(
            " \"'“”‘’.,;:"
        )


    # ========================================================
    # Candidate plausibility
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


        if (
            cls._is_reference_phrase(
                candidate
            )
        ):

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


        if (
            len(
                words
            )
            >
            10
        ):

            return False


        if words[
            0
        ].isdigit():

            return False


        months = {
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        }


        if (
            words[
                0
            ]
            in months
        ):

            return False


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
            "in",
            "regarding",
            "according",
            "both",
            "these",
            "those",
            "different",
            "articles",
            "article",
            "reports",
            "report",
            "coverage",
            "analysis",
            "study",
            "sources",
            "source",
        }


        if (
            words[
                0
            ]
            in forbidden_leading_words
        ):

            return False


        if (
            len(
                words
            )
            >=
            2
            and
            words[
                0
            ]
            ==
            "the"
            and
            words[
                1
            ]
            in {
                "subsequent",
                "other",
                "same",
                "information",
                "data",
                "details",
                "findings",
            }
        ):

            return False


        if (
            words[
                -1
            ]
            in {
                "the",
                "a",
                "an",
            }
        ):

            return False


        generic_only = {
            "article",
            "report",
            "reports",
            "coverage",
            "reporting",
            "update",
            "updates",
            "source",
            "sources",
            "news",
        }


        if (
            len(
                words
            )
            ==
            1
            and
            words[
                0
            ]
            in generic_only
        ):

            return False


        return True


    # ========================================================
    # Candidate registration
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


        source_terminators = (
            r"on|after|before|about|regarding|between|"
            r"published|played|"
            r"suggesting|suggested|suggests|"
            r"detailing|detailed|"
            r"focusing|focused|focuses|"
            r"discussing|discussed|discusses|"
            r"covering|covered|covers|"
            r"indicating|indicated|indicates|"
            r"showing|showed|shows|"
            r"mentioning|mentioned|mentions|"
            r"describing|described|describes|"
            r"defending|defended|defends|"
            r"claiming|claimed|claims|"
            r"arguing|argued|argues|"
            r"stating|stated|states|"
            r"explaining|explained|explains|"
            r"remaining|remained|remains|remain|"
            r"being|"
            r"which|who|that|while|whereas|"
            r"was|were|is|are|"
            r"has|have|had"
        )


        # ====================================================
        # A)
        # report/article/coverage FROM/BY source
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
                r"[^,;?]{1,120}?"
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
            after_marker_pattern.finditer(
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
        # according to source
        # ====================================================

        according_pattern = re.compile(
            (
                r"\b"
                r"according\s+to\s+"
                r"(?P<source>"
                r"[^,;?]{1,120}?"
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
            according_pattern.finditer(
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
        # Compact publisher grammar
        # ====================================================

        source_name_pattern = (
            r"(?:The\s+)?"
            r"[A-Z][A-Za-z0-9&.]*"
            r"(?:"
            r"\s+(?:"
            r"[A-Z][A-Za-z0-9&.]*|"
            r"and|of|for"
            r")"
            r"|"
            r"\s*\|\s*"
            r"[A-Z][A-Za-z0-9&.]*"
            r"|"
            r"\s*-\s*"
            r"(?:"
            r"[A-Z][A-Za-z0-9&.]*|"
            r"and|of|for"
            r")"
            r"){0,8}"
        )


        # ====================================================
        # C)
        # Source's report / coverage
        # ====================================================

        possessive_pattern = re.compile(
            (
                r"(?P<source>"
                +
                source_name_pattern
                +
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
            possessive_pattern.finditer(
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
        # Source article / Source report
        # ====================================================

        before_marker_pattern = re.compile(
            (
                r"(?P<source>"
                +
                source_name_pattern
                +
                r")"
                r"\s+"
                r"(?:article|report)"
                r"\b"
            )
        )


        for match in (
            before_marker_pattern.finditer(
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
        # quoted source
        # ====================================================

        quoted_pattern = re.compile(
            (
                r"[\"'“”‘’]"
                r"(?P<source>"
                r"[^\"'“”‘’]{2,120}"
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
            quoted_pattern.finditer(
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


        for candidate in (
            candidates
        ):

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


        for item in (
            items
        ):

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
    # Exact candidate -> actual context source
    #
    # IMPORTANT:
    #
    # Exact matching is separated from fuzzy grammatical
    # matching.
    #
    # This prevents:
    #
    # "TechCrunch and The Verge"
    #
    # from prematurely matching only:
    #
    # "TechCrunch"
    # ========================================================

    @classmethod
    def _match_available_source_exact(
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


        for source in (
            available_sources
        ):

            for alias in (
                cls._source_aliases(
                    source
                )
            ):

                if (
                    candidate_normalized
                    ==
                    alias
                ):

                    matches.append(
                        (
                            len(
                                alias
                            ),
                            source,
                        )
                    )


        if not matches:

            return None


        matches.sort(
            reverse=True
        )


        return (
            matches[
                0
            ][
                1
            ]
        )


    # ========================================================
    # Candidate -> actual context source
    #
    # Allows limited grammatical spillover.
    # ========================================================

    @classmethod
    def _match_available_source(
        cls,
        *,
        candidate: str,
        available_sources: list[str],
    ) -> str | None:

        exact_match = (
            cls._match_available_source_exact(
                candidate=
                    candidate,

                available_sources=
                    available_sources,
            )
        )


        if exact_match is not None:

            return exact_match


        candidate_normalized = (
            cls._normalize(
                candidate
            )
        )


        if not candidate_normalized:

            return None


        matches = []


        candidate_words = (
            candidate_normalized.split()
        )


        for source in (
            available_sources
        ):

            for alias in (
                cls._source_aliases(
                    source
                )
            ):

                if not alias:

                    continue


                alias_words = (
                    alias.split()
                )


                if (
                    len(
                        candidate_words
                    )
                    >
                    (
                        len(
                            alias_words
                        )
                        +
                        3
                    )
                ):

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
                    candidate_normalized,
                ):

                    matches.append(
                        (
                            len(
                                alias
                            ),
                            source,
                        )
                    )


        if not matches:

            return None


        matches.sort(
            reverse=True
        )


        return (
            matches[
                0
            ][
                1
            ]
        )


    # ========================================================
    # Compound source candidate
    # ========================================================

    @classmethod
    def _split_compound_candidate(
        cls,
        candidate: str,
    ) -> list[str]:

        candidate = (
            cls._clean_candidate(
                candidate
            )
        )


        if not candidate:

            return []


        # ----------------------------------------------------
        # Pipe/hyphen publisher identities remain atomic.
        #
        # The Independent - Life and Style
        # ----------------------------------------------------

        if (
            "|"
            in candidate
            or
            " - "
            in candidate
        ):

            return [
                candidate
            ]


        raw_parts = re.split(
            r"\s+and\s+",
            candidate,
            flags=re.IGNORECASE,
        )


        if (
            len(
                raw_parts
            )
            <=
            1
        ):

            return [
                candidate
            ]


        parts = []


        for raw_part in (
            raw_parts
        ):

            part = (
                cls._clean_candidate(
                    raw_part
                )
            )


            if not (
                cls._is_plausible_source_candidate(
                    part
                )
            ):

                continue


            parts.append(
                part
            )


        if (
            len(
                parts
            )
            >=
            2
        ):

            return parts


        return [
            candidate
        ]


    # ========================================================
    # Resolve one candidate
    #
    # ORDER IS IMPORTANT:
    #
    # 1. exact whole publisher
    # 2. compound publisher list
    # 3. fuzzy singular publisher
    # 4. real-looking missing publisher
    # ========================================================

    @classmethod
    def _resolve_candidate(
        cls,
        *,
        candidate: str,
        available_sources: list[str],
    ) -> tuple[
        list[str],
        list[str],
        list[str],
    ]:

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

            return (
                [],
                [],
                [],
            )


        # ====================================================
        # 1. Exact whole match
        #
        # Protects true names which contain "and".
        # ====================================================

        exact_match = (
            cls._match_available_source_exact(
                candidate=
                    candidate,

                available_sources=
                    available_sources,
            )
        )


        if exact_match is not None:

            return (
                [
                    exact_match
                ],
                [
                    exact_match
                ],
                [],
            )


        # ====================================================
        # 2. Compound list
        #
        # TechCrunch and The Verge
        #
        # This MUST happen before fuzzy whole matching.
        # ====================================================

        parts = (
            cls._split_compound_candidate(
                candidate
            )
        )


        if (
            len(
                parts
            )
            >
            1
        ):

            required = []

            covered = []

            missing = []


            for part in (
                parts
            ):

                matched_part = (
                    cls._match_available_source(
                        candidate=
                            part,

                        available_sources=
                            available_sources,
                    )
                )


                if matched_part is not None:

                    required.append(
                        matched_part
                    )

                    covered.append(
                        matched_part
                    )


                else:

                    required.append(
                        part
                    )

                    missing.append(
                        part
                    )


            return (
                required,
                covered,
                missing,
            )


        # ====================================================
        # 3. Fuzzy singular publisher match
        # ====================================================

        fuzzy_match = (
            cls._match_available_source(
                candidate=
                    candidate,

                available_sources=
                    available_sources,
            )
        )


        if fuzzy_match is not None:

            return (
                [
                    fuzzy_match
                ],
                [
                    fuzzy_match
                ],
                [],
            )


        # ====================================================
        # 4. Singular explicit publisher missing from context
        #
        # Example:
        #
        # The Age
        #
        # This is the safety-critical Case15 path.
        # ====================================================

        return (
            [
                candidate
            ],
            [],
            [
                candidate
            ],
        )


    # ========================================================
    # Detect available context source mentioned in query
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
    # Stable deduplication
    # ========================================================

    @classmethod
    def _deduplicate(
        cls,
        values: list[str],
    ) -> list[str]:

        output = []

        seen = set()


        for value in (
            values
        ):

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
        # Resolve syntactically extracted source references
        # ====================================================

        for candidate in (
            extracted
        ):

            (
                candidate_required,
                candidate_covered,
                candidate_missing,
            ) = (
                self._resolve_candidate(
                    candidate=
                        candidate,

                    available_sources=
                        available_sources,
                )
            )


            required_sources.extend(
                candidate_required
            )


            covered_sources.extend(
                candidate_covered
            )


            missing_sources.extend(
                candidate_missing
            )


        # ====================================================
        # Add actual context sources explicitly mentioned
        # anywhere in the query.
        #
        # This can only increase required + covered.
        # ====================================================

        for source in (
            available_sources
        ):

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


        # ====================================================
        # Covered identity can never simultaneously remain
        # missing.
        # ====================================================

        covered_aliases = set()


        for source in (
            covered_sources
        ):

            covered_aliases.update(
                self._source_aliases(
                    source
                )
            )


        missing_sources = [
            source

            for source
            in missing_sources

            if (
                self._normalize(
                    source
                )
                not in covered_aliases
            )
        ]


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