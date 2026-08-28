from adaptive_agentic_rag.generation.atomic_claim_extractor import (
    AtomicClaimExtractor
)


def main():

    extractor = (
        AtomicClaimExtractor()
    )


    examples = [

        (
            "Walmart does not price-match competitors' products, "
            "limiting its ability to compete effectively in price wars."
        ),

        (
            "Amazon Prime members receive free games and downloadable "
            "content throughout the year, enhancing their shopping experience."
        ),

        (
            "Apple does not offer a universal price-matching policy; "
            "users must check individual websites for specific deals."
        ),

        (
            "Amazon hosts Black Friday sales and Walmart starts "
            "its promotion earlier."
        ),

        (
            "Amazon Prime members receive free games and DLC throughout "
            "the year."
        ),
        (
        "Walmart does not price-match competitors' products; "
        "however, it provides price-matching on items purchased "
        "from its own stores."
    ),

    (
        "Amazon runs seasonal promotions; "
        "however, Walmart offers different discount policies."
    )

    ]


    for index, text in enumerate(
        examples,
        start=1
    ):

        result = extractor.extract(
            text
        )


        print(
            "\n"
            "================================"
        )

        print(
            f"EXAMPLE {index}"
        )


        print(
            "\nORIGINAL:"
        )

        print(
            text
        )


        print(
            "\nATOMIC CLAIMS:"
        )


        for claim_index, claim in enumerate(
            result.claims,
            start=1
        ):

            print(
                f"{claim_index}. "
                f"{claim}"
            )


if __name__ == "__main__":
    main()