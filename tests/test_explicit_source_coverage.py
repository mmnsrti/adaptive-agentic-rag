from types import SimpleNamespace

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)


def context(
    *sources,
):

    return SimpleNamespace(
        items=[
            SimpleNamespace(
                source=source
            )

            for source
            in sources
        ]
    )


def test_case15_missing_the_age_fails():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Has the portrayal of Google's market practices "
        "in reports by The Age after October 22, 2023, "
        "remained consistent with the depiction in "
        "The Verge's coverage of the Epic v. Google case, "
        "and with TechCrunch's report on the class action "
        "antitrust suit filed against Google?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Verge",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is False
    )


    assert any(
        "age"
        in source.lower()

        for source
        in result.missing_sources
    )


    assert (
        "The Verge"
        in
        result.covered_sources
    )


    assert (
        "TechCrunch"
        in
        result.covered_sources
    )


def test_case15_all_sources_present_passes():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Has the portrayal of Google's market practices "
        "in reports by The Age after October 22, 2023, "
        "remained consistent with the depiction in "
        "The Verge's coverage of the Epic v. Google case, "
        "and with TechCrunch's report on the class action "
        "antitrust suit filed against Google?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Age",
                "The Verge",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_case10_three_sources_present_passes():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Does the Mashable article suggest that Amazon's "
        "Cyber Monday includes both continued and new deals, "
        "while The Sydney Morning Herald article focuses on "
        "the impact of an antitrust lawsuit on Amazon's "
        "stock price, and the Cnbc | World Business News "
        "Leader article discusses the opportunity of "
        "selling on Amazon?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "Mashable",
                "The Sydney Morning Herald",
                (
                    "Cnbc | "
                    "World Business News Leader"
                ),
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_pipe_source_can_match_primary_alias():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "According to CNBC, Amazon sellers "
        "discussed their experience."
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                (
                    "Cnbc | "
                    "World Business News Leader"
                )
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        (
            "Cnbc | "
            "World Business News Leader"
        )
        in
        result.covered_sources
    )


def test_quoted_pipe_source_is_supported():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Between the report from "
        "'The Roar | Sports Writers Blog' "
        "on October 19, 2023 and another update, "
        "which source reported the larger lead?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                (
                    "The Roar | "
                    "Sports Writers Blog"
                )
            ),
    )


    assert (
        result.satisfied
        is True
    )


def test_missing_single_explicit_source_fails():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "According to Reuters, "
        "which country launched the initiative?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "Bloomberg"
            ),
    )


    assert (
        result.satisfied
        is False
    )


    assert any(
        "reuters"
        in source.lower()

        for source
        in result.missing_sources
    )


def test_query_without_explicit_source_passes():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Which company faced an "
        "antitrust lawsuit?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "TechCrunch",
                "The Verge",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )
    
def test_reports_by_source_followed_by_remained_is_detected():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Has Google's portrayal in reports by "
        "The Age remained consistent with "
        "The Verge's coverage and "
        "TechCrunch's report?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Verge",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is False
    )


    assert any(
        (
            "the age"
            ==
            guard._normalize(
                source
            )
        )

        for source
        in result.missing_sources
    )


    assert (
        "The Verge"
        in
        result.covered_sources
    )


    assert (
        "TechCrunch"
        in
        result.covered_sources
    )