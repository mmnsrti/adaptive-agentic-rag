import re

from dataclasses import dataclass

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)


# ============================================================
# Generic lexical stopwords
# ============================================================

STOPWORDS = {

    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
}


# ============================================================
# Tokens that may be capitalized only because of grammar,
# dates, or generic question wording.
#
# These should NOT become evidence anchors.
# ============================================================

CRITICAL_TERM_EXCLUSIONS = (

    STOPWORDS

    |

    {
        "after",
        "before",
        "considering",
        "according",
        "based",
        "following",
        "given",

        "article",
        "articles",
        "report",
        "reports",
        "reported",
        "reporting",
        "published",

        "compare",
        "comparison",

        "yes",
        "no",

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

)


# ============================================================
# Result model
# ============================================================

@dataclass
class EvidenceGrade:

    sufficient: bool

    evidence_score: float

    query_term_coverage: float

    unique_documents: int

    chunk_count: int

    weak_citations: list[int]

    reasons: list[str]


# ============================================================
# Evidence grader
# ============================================================

class EvidenceGrader:


    def __init__(
        self,
        weak_item_threshold: float = 0.20,
        hard_critical_coverage: float = 0.75,
        simple_critical_coverage: float = 0.50
    ):

        self.weak_item_threshold = (
            weak_item_threshold
        )

        self.hard_critical_coverage = (
            hard_critical_coverage
        )

        self.simple_critical_coverage = (
            simple_critical_coverage
        )


    # ========================================================
    # General tokenization
    # ========================================================

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
                and
                len(token) > 1
            )
        }


    def _query_terms(
        self,
        query: str
    ) -> set[str]:

        return self._tokenize(
            query
        )


    # ========================================================
    # Critical evidence anchors
    #
    # Examples:
    #
    # Amazon
    # AcmeMart
    # Taylor
    # Swift
    # Travis
    # Kelce
    # TechCrunch
    # Google
    # YouTube
    #
    # These are more important than generic terms like
    # "shipping", "deals", "report", etc.
    # ========================================================

    def _critical_terms(
        self,
        query: str
    ) -> set[str]:

        #
        # Capture capitalized / proper-name-like tokens.
        #
        # This also catches:
        #
        # AcmeMart
        # TechCrunch
        # CBSSports
        # BBC
        # Google
        #

        raw_tokens = re.findall(
            r"\b[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*\b",
            query
        )


        critical_terms = set()


        for token in raw_tokens:

            normalized = (
                token
                .lower()
                .strip()
            )


            if not normalized:

                continue


            if len(normalized) <= 1:

                continue


            if (
                normalized
                in
                CRITICAL_TERM_EXCLUSIONS
            ):

                continue


            critical_terms.add(
                normalized
            )


        return critical_terms


    # ========================================================
    # Coverage
    # ========================================================

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
            len(
                matched_terms
            )
            /
            len(
                query_terms
            )
        )


    # ========================================================
    # Evidence requirements
    # ========================================================

    def _requirements(
        self,
        query_type: str
    ) -> tuple[int, int, float]:

        """
        Returns:

        required_documents,
        required_chunks,
        minimum_query_coverage

        Evidence sufficiency intentionally uses
        two safety levels:

        SIMPLE:
            single-document lookup

        HARD:
            multihop / complex

        MULTIHOP and COMPLEX share the same
        safety policy because fine-grained
        classification between them should not
        change the abstention boundary.
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


    # ========================================================
    # Critical-anchor threshold
    # ========================================================

    def _critical_coverage_requirement(
        self,
        query_type: str,
        critical_terms: set[str]
    ) -> float:

        if not critical_terms:

            return 0.0


        #
        # If there is only one critical anchor,
        # it must be present.
        #
        # Example:
        #
        # "Which company ... Valve?"
        #

        if len(
            critical_terms
        ) == 1:

            return 1.0


        if query_type == "simple":

            return (
                self.simple_critical_coverage
            )


        return (
            self.hard_critical_coverage
        )


    # ========================================================
    # Main grading
    # ========================================================

    def grade(
        self,
        query: str,
        context: BuiltContext,
        query_type: str
    ) -> EvidenceGrade:


        query_terms = (
            self._query_terms(
                query
            )
        )


        critical_terms = (
            self._critical_terms(
                query
            )
        )


        required_documents, (
            required_chunks
        ), minimum_coverage = (
            self._requirements(
                query_type
            )
        )


        # ====================================================
        # Empty context
        # ====================================================

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


        # ====================================================
        # Overall lexical coverage
        # ====================================================

        overall_coverage = (
            self._coverage(
                query_terms,
                context.text
            )
        )


        # ====================================================
        # Critical anchor coverage
        # ====================================================

        if critical_terms:

            context_terms = (
                self._tokenize(
                    context.text
                )
            )


            matched_critical_terms = (
                critical_terms
                &
                context_terms
            )


            missing_critical_terms = (
                critical_terms
                -
                context_terms
            )


            critical_coverage = (

                len(
                    matched_critical_terms
                )

                /

                len(
                    critical_terms
                )
            )


        else:

            matched_critical_terms = set()

            missing_critical_terms = set()

            critical_coverage = 1.0


        minimum_critical_coverage = (
            self._critical_coverage_requirement(
                query_type,
                critical_terms
            )
        )


        # ====================================================
        # Document diversity
        # ====================================================

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


        # ====================================================
        # Detect weak individual evidence
        # ====================================================

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


        # ====================================================
        # Component scores
        # ====================================================

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


        # ====================================================
        # Combined score
        #
        # NOTE:
        #
        # Critical anchor coverage is intentionally NOT
        # blended into this score.
        #
        # It acts as a hard safety gate below.
        #
        # Otherwise a high document/chunk score could
        # compensate for a missing named entity.
        # ====================================================

        evidence_score = (

            0.55
            *
            overall_coverage

            +

            0.30
            *
            document_score

            +

            0.15
            *
            chunk_score
        )


        # ====================================================
        # Hard requirements
        # ====================================================

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


        enough_critical_coverage = (

            critical_coverage

            >=

            minimum_critical_coverage
        )


        sufficient = (

            enough_documents

            and

            enough_chunks

            and

            enough_coverage

            and

            enough_critical_coverage

            and

            evidence_score >= 0.70
        )


        # ====================================================
        # Human-readable reasons
        # ====================================================

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


        if not enough_critical_coverage:

            missing_display = sorted(
                missing_critical_terms
            )


            reasons.append(

                (
                    "Critical query anchors are missing "
                    "from the evidence: "
                    f"coverage={critical_coverage:.2f} "
                    f"< {minimum_critical_coverage:.2f}; "
                    f"missing={missing_display}"
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