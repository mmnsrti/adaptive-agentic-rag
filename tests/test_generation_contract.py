from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
    ContextItem,
)

from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator,
)

from adaptive_agentic_rag.generation.relevance_filter import (
    RelevantClaim,
)


def test_parse_structured_generation():

    raw = """
DRAFT_ANSWER: Yes

FACTS:
- McTominay is Manchester United's top scorer.
- Haaland can become the overall top scorer in 2023.
""".strip()


    parsed = (
        GroundedGenerator
        ._parse_draft(
            raw
        )
    )


    assert (
        parsed.direct_answer
        ==
        "Yes"
    )


    assert parsed.evidence_claims == [
        "McTominay is Manchester United's top scorer.",
        "Haaland can become the overall top scorer in 2023.",
    ]


def test_accidental_model_citations_are_removed_and_not_trusted():

    raw = """
DRAFT_ANSWER: Yes [1]

FACTS:
- [2] McTominay is Manchester United's top scorer.
""".strip()


    parsed = (
        GroundedGenerator
        ._parse_draft(
            raw
        )
    )


    assert (
        parsed.direct_answer
        ==
        "Yes"
    )


    assert (
        parsed.evidence_facts[
            0
        ].text
        ==
        "McTominay is Manchester United's top scorer."
    )


    assert (
        parsed.evidence_facts[
            0
        ].citation_id
        is None
    )


def test_build_final_answer_attaches_grounder_citations():

    relevant_claims = [
        RelevantClaim(
            claim=(
                "McTominay is Manchester United's top scorer."
            ),
            citation_id=1,
            relevance_score=5.0,
        ),
        RelevantClaim(
            claim=(
                "Haaland can become the overall top scorer."
            ),
            citation_id=2,
            relevance_score=4.0,
        ),
    ]


    answer = (
        GroundedGenerator
        ._build_grounded_answer(
            direct_answer=
                "Yes",
            relevant_claims=
                relevant_claims,
        )
    )


    assert answer == (
        "Yes [1][2]\n"
        "- McTominay is Manchester United's top scorer. [1]\n"
        "- Haaland can become the overall top scorer. [2]"
    )


def test_legacy_bullet_output_still_parses():

    raw = """
- First supported fact.
- Second supported fact.
""".strip()


    parsed = (
        GroundedGenerator
        ._parse_draft(
            raw
        )
    )


    assert (
        parsed.direct_answer
        is None
    )


    assert parsed.evidence_claims == [
        "First supported fact.",
        "Second supported fact.",
    ]


def test_insufficient_answer_normalization():

    variants = [
        "INSUFFICIENT_EVIDENCE",
        "Insufficient Evidence",
        "insufficient-evidence",
        "Insufficient Evidence.",
    ]


    for value in variants:

        assert (
            GroundedGenerator
            ._is_insufficient_answer(
                value
            )
            is True
        )


def test_verified_facts_include_source_metadata():

    context = BuiltContext(
        text="",
        total_words=10,
        items=[
            ContextItem(
                citation_id=2,
                chunk_id="chunk_2",
                document_id="doc_2",
                title="Top goalscorers of 2023",
                source="TalkSport",
                url=None,
                text=(
                    "Haaland has time to become "
                    "the overall top scorer."
                ),
                score=1.0,
            )
        ],
    )


    relevant_claims = [
        RelevantClaim(
            claim=(
                "Haaland can become "
                "the overall top scorer."
            ),
            citation_id=2,
            relevance_score=5.0,
        )
    ]


    verified = (
        GroundedGenerator
        ._build_verified_facts(
            relevant_claims=
                relevant_claims,
            context=
                context,
        )
    )


    assert verified == [
        {
            "citation_id": 2,
            "source": "TalkSport",
            "title": "Top goalscorers of 2023",
            "text": (
                "Haaland can become "
                "the overall top scorer."
            ),
        }
    ]