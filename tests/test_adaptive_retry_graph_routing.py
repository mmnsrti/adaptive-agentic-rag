from adaptive_agentic_rag.orchestration.graph import (
    route_after_evidence,
)


# ============================================================
# Helper
# ============================================================

def make_state(
    *,
    evidence_sufficient: bool,
    evidence_reasons: list[str],
    retry_count: int = 0,
    max_retries: int = 1,
):

    return {
        "evidence_sufficient":
            evidence_sufficient,

        "evidence_reasons":
            evidence_reasons,

        "retry_count":
            retry_count,

        "max_retries":
            max_retries,
    }


# ============================================================
# Sufficient evidence
#
# Must always go directly to generation.
# ============================================================

def test_sufficient_evidence_routes_to_generate():

    state = make_state(
        evidence_sufficient=True,

        evidence_reasons=[
            "evidence_path=v2",
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "generate"
    )


# ============================================================
# Canonical Case 15 shape
#
# Some required publishers are present,
# one is missing.
#
# This is a justified retrieval retry.
# ============================================================

def test_partial_explicit_source_miss_routes_to_rewrite():

    state = make_state(
        evidence_sufficient=False,

        evidence_reasons=[
            (
                "required_sources="
                "['The Age', 'The Verge', 'TechCrunch']"
            ),

            (
                "covered_sources="
                "['The Verge', 'TechCrunch']"
            ),

            (
                "missing_sources="
                "['The Age']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "rewrite"
    )


# ============================================================
# Complete context / semantic-gate rejection
#
# Cases 6 and 13 demonstrated this failure family.
#
# Re-running retrieval is not structurally justified.
# ============================================================

def test_semantic_rescue_reject_routes_to_abstain():

    state = make_state(
        evidence_sufficient=False,

        evidence_reasons=[
            (
                "explicit_source_coverage="
                "satisfied"
            ),

            (
                "required_sources="
                "['The Verge', 'TechCrunch']"
            ),

            (
                "covered_sources="
                "['The Verge', 'TechCrunch']"
            ),

            "semantic_rescue=insufficient",

            (
                "evidence_path="
                "constrained_semantic_rescue_reject"
            ),
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "abstain"
    )


# ============================================================
# Null-query style:
#
# Required publishers exist in the query,
# but none are covered.
#
# Blind rewriting is not justified in V1.
# ============================================================

def test_zero_explicit_source_coverage_routes_to_abstain():

    state = make_state(
        evidence_sufficient=False,

        evidence_reasons=[
            (
                "required_sources="
                "['Bloomberg', 'CNN']"
            ),

            "covered_sources=[]",

            (
                "missing_sources="
                "['Bloomberg', 'CNN']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "abstain"
    )


# ============================================================
# Retry budget exhausted
#
# Even a structurally recoverable source miss cannot loop
# indefinitely.
# ============================================================

def test_partial_source_miss_with_exhausted_budget_abstains():

    state = make_state(
        evidence_sufficient=False,

        retry_count=1,
        max_retries=1,

        evidence_reasons=[
            (
                "required_sources="
                "['The Age', 'TechCrunch']"
            ),

            (
                "covered_sources="
                "['TechCrunch']"
            ),

            (
                "missing_sources="
                "['The Age']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "abstain"
    )


# ============================================================
# Runtime max_retries must remain configurable.
#
# max_retries=2 and retry_count=1 still leaves one attempt.
# ============================================================

def test_runtime_retry_budget_is_honored():

    state = make_state(
        evidence_sufficient=False,

        retry_count=1,
        max_retries=2,

        evidence_reasons=[
            (
                "required_sources="
                "['The Age', 'TechCrunch']"
            ),

            (
                "covered_sources="
                "['TechCrunch']"
            ),

            (
                "missing_sources="
                "['The Age']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "rewrite"
    )


# ============================================================
# Unknown future rejection
#
# Fail closed.
# ============================================================

def test_unknown_evidence_rejection_routes_to_abstain():

    state = make_state(
        evidence_sufficient=False,

        evidence_reasons=[
            "unknown_future_gate=true",
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "abstain"
    )