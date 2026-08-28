import re

from dataclasses import dataclass


@dataclass
class AtomicClaims:

    original_text: str

    claims: list[str]


class AtomicClaimExtractor:


    def _clean(
        self,
        text: str
    ) -> str:

        text = text.strip()

        #
        # Remove bullet markers
        #

        text = re.sub(
            r"^[-*•]\s*",
            "",
            text
        )


        #
        # Remove existing citations
        #

        text = re.sub(
            r"\[\d+\]",
            "",
            text
        )


        #
        # Normalize whitespace
        #

        text = re.sub(
            r"\s+",
            " ",
            text
        )


        return text.strip()


    def _split_sentences(
        self,
        text: str
    ) -> list[str]:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )


        return [

            sentence.strip()

            for sentence in sentences

            if sentence.strip()

        ]


    def _split_semicolon(
        self,
        sentence: str
    ) -> list[str]:

        parts = re.split(
            r"\s*;\s*",
            sentence
        )

        parts = [
            part.strip()
            for part in parts
            if part.strip()
        ]


        if len(parts) <= 1:

            return parts


        cleaned_parts = []


        #
        # Clauses such as:
        #
        # however, it ...
        # but it ...
        # yet they ...
        #
        # depend on the previous clause and are
        # not safe standalone atomic claims.
        #

        dependent_pattern = re.compile(
            r"^(?:however|but|yet|nevertheless|nonetheless),?\s+"
            r"(?:it|they|this|these|those|he|she)\b",
            flags=re.IGNORECASE
        )


        #
        # If the connector is followed by an explicit
        # subject, we can keep the clause after removing
        # the discourse connector:
        #
        # however, Walmart offers ...
        # -> Walmart offers ...
        #

        connector_pattern = re.compile(
            r"^(?:however|but|yet|nevertheless|nonetheless),?\s+",
            flags=re.IGNORECASE
        )


        for index, part in enumerate(parts):

            #
            # First semicolon clause is already
            # self-contained in the normal case.
            #

            if index == 0:

                cleaned_parts.append(
                    part
                )

                continue


            #
            # Drop context-dependent clauses.
            #

            if dependent_pattern.match(
                part
            ):

                continue


            #
            # Remove unnecessary discourse connector
            # from otherwise self-contained clauses.
            #

            part = connector_pattern.sub(
                "",
                part
            ).strip()


            if not part:

                continue


            #
            # Cosmetic cleanup:
            #
            # users must ...
            # -> Users must ...
            #

            part = (
                part[0].upper()
                +
                part[1:]
            )


            cleaned_parts.append(
                part
            )


        return cleaned_parts


    def _split_explanatory_clause(
        self,
        sentence: str
    ) -> list[str]:

        """
        Remove dependent explanatory clauses such as:

        Walmart does not price-match competitors,
        limiting its ability to compete...

        -> Walmart does not price-match competitors.

        We intentionally discard the dependent clause
        in the deterministic baseline because expressions
        such as "limiting", "providing", or "enhancing"
        often require subject/coreference resolution and
        may contain generated interpretation.
        """

        pattern = (
            r",\s*"
            r"(allowing|providing|limiting|enhancing|"
            r"making|causing|creating|giving|offering|"
            r"bringing)"
            r"\s+"
        )


        match = re.search(
            pattern,
            sentence,
            flags=re.IGNORECASE
        )


        if not match:

            return [
                sentence.strip()
            ]


        main_claim = (
            sentence[
                :match.start()
            ]
            .strip()
        )


        if not main_claim:

            return []


        return [
            main_claim
        ]


    def _split_and_clause(
        self,
        sentence: str
    ) -> list[str]:

        """
        Split only when the text after "and"
        looks like a new independent clause.

        Example:

        Amazon hosts Black Friday sales and
        Walmart starts its promotion earlier.

        -> split

        But:

        Amazon Prime members receive free games
        and DLC throughout the year.

        -> do NOT split
        """

        parts = re.split(
            r"\s+\band\b\s+",
            sentence,
            maxsplit=1,
            flags=re.IGNORECASE
        )


        if len(parts) != 2:

            return [
                sentence
            ]


        left = parts[0].strip()

        right = parts[1].strip()


        left_words = left.split()

        right_words = right.split()


        if (
            len(left_words) < 3
            or
            len(right_words) < 3
        ):

            return [
                sentence
            ]


        #
        # Right side should begin with
        # something that can act as a subject.
        #

        first_word = re.sub(
            r"[^A-Za-z']",
            "",
            right_words[0]
        )


        pronouns = {
            "he",
            "she",
            "it",
            "they",
            "we",
            "you"
        }


        subject_like = (

            (
                first_word
                and
                first_word[0].isupper()
            )

            or

            first_word.lower()
            in pronouns

        )


        if not subject_like:

            return [
                sentence
            ]


        #
        # More importantly:
        # the right side must also contain
        # something that looks like a finite verb.
        #
        # This makes:
        #
        # "Walmart starts its promotion"
        #      -> clause
        #
        # but:
        #
        # "DLC throughout the year"
        #      -> NOT a clause
        #

        auxiliary_verbs = {

            "is",
            "are",
            "was",
            "were",

            "has",
            "have",
            "had",

            "does",
            "do",
            "did",

            "can",
            "could",

            "will",
            "would",

            "should",
            "may",
            "might",
            "must"

        }


        has_likely_verb = False


        #
        # We inspect only the beginning
        # of the candidate clause.
        #

        candidate_words = (
            right_words[1:6]
        )


        for word in candidate_words:

            clean_word = re.sub(
                r"[^A-Za-z']",
                "",
                word
            ).lower()


            if not clean_word:

                continue


            if clean_word in auxiliary_verbs:

                has_likely_verb = True

                break


            #
            # Conservative heuristic for
            # common finite verb forms:
            #
            # starts
            # matches
            # hosts
            # offers
            # provided / started ...
            #

            if (
                clean_word.endswith("ed")
                or
                (
                    clean_word.endswith("s")
                    and
                    len(clean_word) > 3
                )
            ):

                has_likely_verb = True

                break


        if not has_likely_verb:

            return [
                sentence
            ]


        return [
            left,
            right
        ]

    def extract(
        self,
        text: str
    ) -> AtomicClaims:


        original_text = text


        text = self._clean(
            text
        )


        if not text:

            return AtomicClaims(

                original_text=(
                    original_text
                ),

                claims=[]

            )


        atomic_claims = []


        #
        # Step 1:
        # sentence splitting
        #

        sentences = (
            self._split_sentences(
                text
            )
        )


        for sentence in sentences:


            #
            # Step 2:
            # semicolon splitting
            #

            semicolon_parts = (
                self._split_semicolon(
                    sentence
                )
            )


            for part in semicolon_parts:


                #
                # Step 3:
                # participial/explanatory clause
                #

                explanatory_parts = (
                    self
                    ._split_explanatory_clause(
                        part
                    )
                )


                for explanatory_part in (
                    explanatory_parts
                ):


                    #
                    # Step 4:
                    # conservative "and" split
                    #

                    final_parts = (
                        self._split_and_clause(
                            explanatory_part
                        )
                    )


                    for final_part in final_parts:

                        claim = (
                            final_part.strip(
                                " ,"
                            )
                        )


                        if not claim:

                            continue


                        #
                        # Add final punctuation
                        #

                        if claim[-1] not in ".!?":

                            claim += "."


                        atomic_claims.append(
                            claim
                        )


        #
        # Deduplicate while
        # preserving order
        #

        unique_claims = list(
            dict.fromkeys(
                atomic_claims
            )
        )


        return AtomicClaims(

            original_text=(
                original_text
            ),

            claims=unique_claims

        )