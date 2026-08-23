import re


def normalize_text_key(
    value: str
) -> str:

    value = value.strip().casefold()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


def make_document_key(
    source: str,
    title: str
) -> str:

    source = normalize_text_key(
        source
    )

    title = normalize_text_key(
        title
    )

    return f"{source}::{title}"


def get_relevant_document_keys(
    evidence_list: list[dict]
) -> set[str]:

    return {
        make_document_key(
            evidence["source"],
            evidence["title"]
        )
        for evidence in evidence_list
    }


def get_retrieved_document_keys(
    results: list[dict]
) -> list[str]:

    keys = []

    for result in results:

        metadata = result["metadata"]

        key = make_document_key(
            metadata["source"],
            metadata["title"]
        )

        keys.append(key)

    return keys