from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator,
)

from adaptive_agentic_rag.generation.relevance_filter import (
    RelevantClaim,
)


def test_parse_structured_generation():

    raw = """
DIRECT_ANSWER: Yes

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


def test_old_draft_answer_header_remains_parseable():

    raw = """
DRAFT_ANSWER: No

FACTS:
- One supported fact.
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
        "No"
    )


    assert parsed.evidence_claims == [
        "One supported fact.",
    ]


def test_accidental_model_citations_are_removed():

    raw = """
DIRECT_ANSWER: Yes [1]

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
        "insufficient",
    ]


    for value in variants:

        assert (
            GroundedGenerator
            ._is_insufficient_answer(
                value
            )
            is True
        )