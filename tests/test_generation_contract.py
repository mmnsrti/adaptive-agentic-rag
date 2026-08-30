from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator,
)

from adaptive_agentic_rag.generation.relevance_filter import (
    RelevantClaim,
)


def test_parse_structured_generation():

    raw = """
DIRECT_ANSWER: Yes, the reports refer to different scoring scopes.

FACTS:
- Sporting News states that McTominay is United's top scorer.
- TalkSport states that Haaland can become the overall top scorer.
""".strip()

    parsed = GroundedGenerator._parse_draft(
        raw
    )

    assert parsed.direct_answer == (
        "Yes, the reports refer to "
        "different scoring scopes."
    )

    assert parsed.evidence_claims == [
        (
            "Sporting News states that "
            "McTominay is United's top scorer."
        ),
        (
            "TalkSport states that Haaland "
            "can become the overall top scorer."
        ),
    ]


def test_parse_generation_strips_existing_citations():

    raw = """
DIRECT_ANSWER: Yes. [1]

FACTS:
- McTominay is United's top scorer. [1]
""".strip()

    parsed = GroundedGenerator._parse_draft(
        raw
    )

    assert parsed.direct_answer == "Yes."

    assert parsed.evidence_claims == [
        "McTominay is United's top scorer."
    ]


def test_build_final_answer_attaches_grounded_citations():

    relevant_claims = [
        RelevantClaim(
            claim="McTominay is United's top scorer.",
            citation_id=1,
            relevance_score=5.0,
        ),
        RelevantClaim(
            claim="Haaland can become the overall top scorer.",
            citation_id=2,
            relevance_score=4.0,
        ),
    ]

    answer = GroundedGenerator._build_grounded_answer(
        direct_answer="Yes, their scoring scopes differ.",
        relevant_claims=relevant_claims,
    )

    assert answer == (
        "Yes, their scoring scopes differ. [1][2]\n"
        "- McTominay is United's top scorer. [1]\n"
        "- Haaland can become the overall top scorer. [2]"
    )


def test_legacy_bullet_output_still_parses():

    raw = """
- First supported fact.
- Second supported fact.
""".strip()

    parsed = GroundedGenerator._parse_draft(
        raw
    )

    assert parsed.direct_answer is None

    assert parsed.evidence_claims == [
        "First supported fact.",
        "Second supported fact.",
    ]

def test_parse_citation_linked_facts():

    raw = """
DRAFT_ANSWER: Yes

FACTS:
- [1] McTominay is Manchester United's top scorer.
- [2] Haaland can become the overall top scorer in 2023.
""".strip()


    parsed = (
        GroundedGenerator
        ._parse_draft(
            raw
        )
    )


    assert parsed.direct_answer == "Yes"

    assert (
        parsed.evidence_facts[
            0
        ].citation_id
        ==
        1
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
            1
        ].citation_id
        ==
        2
    )


def test_unlinked_legacy_fact_is_marked_untrusted():

    raw = """
DRAFT_ANSWER: Yes

FACTS:
- Some unsupported legacy fact.
""".strip()


    parsed = (
        GroundedGenerator
        ._parse_draft(
            raw
        )
    )


    assert (
        parsed.evidence_facts[
            0
        ].citation_id
        is None
    )