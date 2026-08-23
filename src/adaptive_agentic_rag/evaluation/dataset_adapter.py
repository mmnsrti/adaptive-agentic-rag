from datasets import Dataset



def normalize_evidence(
    evidence_list: list[dict]
) -> list[dict]:

    normalized = []


    for evidence in evidence_list:

        normalized.append(
            {
                "source":
                    evidence["source"],

                "title":
                    evidence["title"],

                "fact":
                    evidence["fact"]
            }
        )


    return normalized



def get_evidence_keys(
    evidence_list: list[dict]
) -> set[str]:

    keys = set()


    for evidence in evidence_list:

        key = (
            evidence["source"]
            +
            "::"
            +
            evidence["title"]
        )

        keys.add(key)


    return keys