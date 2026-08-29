import re
from dataclasses import dataclass

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "about",
    "and",
    "or",
    "but",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "does",
    "do",
    "did",
    "has",
    "have",
    "had",
    "can",
    "could",
    "would",
    "should",
    "compare",
    "comparison",
    "explain",
    "summarize",
    "summary",
    "analyze",
    "analysis",
    "all"
}


@dataclass
class EvidenceGrade:

    sufficient: bool

    evidence_score: float

    query_term_coverage: float

    unique_documents: int

    chunk_count: int

    weak_citations: list[int]

    reasons: list[str]


class EvidenceGrader:


    def __init__(
        self,
        weak_item_threshold: float = 0.20
    ):

        self.weak_item_threshold = (
            weak_item_threshold
        )


    def _tokenize(
        self,
        text: str
    ) -> set[str]:

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

        return {
            token
            for token in tokens
            if (
                token not in STOPWORDS
                and len(token) > 1
            )
        }


    def _query_terms(
        self,
        query: str
    ) -> set[str]:

        return self._tokenize(
            query
        )


    def _coverage(
        self,
        query_terms: set[str],
        text: str
    ) -> float:

        if not query_terms:
            return 1.0


        text_terms = self._tokenize(
            text
        )


        matched_terms = (
            query_terms
            &
            text_terms
        )


        return (
            len(matched_terms)
            /
            len(query_terms)
        )


    def _requirements(
        self,
        query_type: str
    ) -> tuple[int, int, float]:

        """
        Returns:

        required_documents,
        required_chunks,
        minimum_query_coverage

        Evidence sufficiency currently uses two
        safety levels:

        SIMPLE:
            single-document lookup

        HARD:
            multihop / complex questions

        MULTIHOP and COMPLEX intentionally share
        the same evidence policy because fine-grained
        query-type classification is not reliable
        enough to change the abstention boundary.
        """

        if query_type == "simple":

            return (
                1,
                1,
                0.50
            )


        if query_type in {
            "multihop",
            "complex"
        }:

            return (
                2,
                2,
                0.60
            )


        #
        # Conservative fallback.
        #

        return (
            2,
            2,
            0.60
        )

    def grade(
        self,
        query: str,
        context: BuiltContext,
        query_type: str
    ) -> EvidenceGrade:


        query_terms = self._query_terms(
            query
        )


        required_documents, (
            required_chunks
        ), minimum_coverage = (
            self._requirements(
                query_type
            )
        )


        #
        # Empty context
        #

        if not context.items:

            return EvidenceGrade(

                sufficient=False,

                evidence_score=0.0,

                query_term_coverage=0.0,

                unique_documents=0,

                chunk_count=0,

                weak_citations=[],

                reasons=[
                    "No evidence was retrieved."
                ]
            )


        #
        # Overall lexical coverage
        #

        overall_coverage = (
            self._coverage(
                query_terms,
                context.text
            )
        )


        #
        # Document diversity
        #

        unique_document_ids = {

            item.document_id

            for item in context.items

        }


        unique_documents = len(
            unique_document_ids
        )


        chunk_count = len(
            context.items
        )


        #
        # Detect weak individual evidence
        #

        weak_citations = []


        for item in context.items:

            item_coverage = (
                self._coverage(
                    query_terms,
                    item.text
                )
            )


            if (
                item_coverage
                <
                self.weak_item_threshold
            ):

                weak_citations.append(
                    item.citation_id
                )


        #
        # Component scores
        #

        document_score = min(

            unique_documents
            /
            required_documents,

            1.0

        )


        chunk_score = min(

            chunk_count
            /
            required_chunks,

            1.0

        )


        #
        # Combined score
        #

        evidence_score = (

            0.55
            * overall_coverage

            +

            0.30
            * document_score

            +

            0.15
            * chunk_score

        )


        #
        # Hard requirements
        #

        enough_documents = (
            unique_documents
            >=
            required_documents
        )


        enough_chunks = (
            chunk_count
            >=
            required_chunks
        )


        enough_coverage = (
            overall_coverage
            >=
            minimum_coverage
        )


        sufficient = (

            enough_documents

            and

            enough_chunks

            and

            enough_coverage

            and

            evidence_score >= 0.70

        )


        #
        # Human-readable reasons
        #

        reasons = []


        if not enough_documents:

            reasons.append(

                (
                    "Insufficient document diversity: "
                    f"{unique_documents}/"
                    f"{required_documents}"
                )

            )


        if not enough_chunks:

            reasons.append(

                (
                    "Too few evidence chunks: "
                    f"{chunk_count}/"
                    f"{required_chunks}"
                )

            )


        if not enough_coverage:

            reasons.append(

                (
                    "Query term coverage is too low: "
                    f"{overall_coverage:.2f} "
                    f"< {minimum_coverage:.2f}"
                )

            )


        if weak_citations:

            reasons.append(

                (
                    "Potentially weak evidence citations: "
                    f"{weak_citations}"
                )

            )


        if sufficient:

            reasons.insert(

                0,

                "Evidence appears sufficient."

            )


        return EvidenceGrade(

            sufficient=sufficient,

            evidence_score=round(
                evidence_score,
                4
            ),

            query_term_coverage=round(
                overall_coverage,
                4
            ),

            unique_documents=(
                unique_documents
            ),

            chunk_count=chunk_count,

            weak_citations=(
                weak_citations
            ),

            reasons=reasons

        )