import json
from pathlib import Path


from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


FROZEN_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "generation_stage_diagnostic.json"
)


# ============================================================
# Known problematic families from the previous E2E smoke run.
#
# We locate them from the frozen set instead of manually
# duplicating entire questions.
# ============================================================

TARGETS = {
    "taylor_kelce": [
        "taylor",
        "kelce",
    ],

    "sam_altman": [
        "sam altman",
    ],

    "mctominay_haaland": [
        "mctominay",
        "haaland",
    ],

    "google_youtube": [
        "google",
        "youtube",
    ],

    "google_epic": [
        "google",
        "epic",
    ],
}


# ============================================================
# Dataset loading
# ============================================================

def load_examples():

    with open(
        FROZEN_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )


    if isinstance(
        payload,
        list,
    ):

        return payload


    for key in (
        "examples",
        "records",
        "data",
    ):

        if key in payload:

            return payload[
                key
            ]


    raise ValueError(
        "Could not locate frozen examples "
        f"inside {FROZEN_PATH}"
    )


# ============================================================
# Find known smoke families
# ============================================================

def find_target_examples(
    examples,
):

    selected = []


    for (
        target_name,
        required_terms,
    ) in TARGETS.items():

        matches = []


        for example in examples:

            question = (
                example.get(
                    "question"
                )
                or
                example.get(
                    "query"
                )
                or
                ""
            )


            normalized = (
                question.lower()
            )


            if all(
                term in normalized
                for term in required_terms
            ):

                matches.append(
                    example
                )


        if not matches:

            print(
                f"[NOT FOUND] "
                f"{target_name}: "
                f"{required_terms}"
            )

            continue


        # ----------------------------------------------------
        # Deterministic choice.
        #
        # Prefer the shortest matching question because broad
        # keyword pairs can occasionally match multiple frozen
        # questions.
        # ----------------------------------------------------

        matches.sort(
            key=lambda item:
                len(
                    (
                        item.get(
                            "question"
                        )
                        or
                        item.get(
                            "query"
                        )
                        or
                        ""
                    )
                )
        )


        chosen = (
            matches[
                0
            ]
        )


        question = (
            chosen.get(
                "question"
            )
            or
            chosen.get(
                "query"
            )
            or
            ""
        )


        print(
            f"\n[{target_name}]"
        )

        print(
            question
        )

        print(
            "Matching frozen examples:",
            len(
                matches
            )
        )


        selected.append(
            (
                target_name,
                chosen,
            )
        )


    return selected


# ============================================================
# Context serialization
# ============================================================

def serialize_context(
    context,
):

    return {
        "total_words":
            context.total_words,

        "items": [
            {
                "citation_id":
                    item.citation_id,

                "chunk_id":
                    item.chunk_id,

                "document_id":
                    item.document_id,

                "title":
                    item.title,

                "source":
                    item.source,

                "url":
                    item.url,

                "score":
                    item.score,

                "text":
                    item.text,
            }

            for item
            in context.items
        ],
    }


# ============================================================
# Grounded claim serialization
# ============================================================

def serialize_grounded_claim(
    claim,
):

    return {
        "claim":
            claim.claim,

        "supported":
            claim.supported,

        "citation_id":
            claim.citation_id,

        "label":
            claim.label,

        "entailment_score":
            claim.entailment_score,

        "supporting_text":
            claim.supporting_text,
    }


# ============================================================
# Run retrieval + evidence gate
#
# This mirrors the graph:
#
# route
# → retrieve
# → context
# → evidence grade
# → optional rewrite
# → retrieve again
# → context again
# → evidence grade again
# ============================================================

def prepare_evidence(
    nodes,
    question,
):

    state = {
        "original_query":
            question,

        "current_query":
            question,

        "retry_count":
            0,
    }


    state.update(
        nodes.route_query(
            state
        )
    )


    state.update(
        nodes.retrieve(
            state
        )
    )


    state.update(
        nodes.build_context(
            state
        )
    )


    state.update(
        nodes.grade_evidence(
            state
        )
    )


    attempts = [
        {
            "attempt":
                0,

            "query":
                state[
                    "current_query"
                ],

            "evidence_sufficient":
                state[
                    "evidence_sufficient"
                ],

            "evidence_score":
                state[
                    "evidence_score"
                ],

            "evidence_reasons":
                state[
                    "evidence_reasons"
                ],

            "context":
                serialize_context(
                    state[
                        "context"
                    ]
                ),
        }
    ]


    # ========================================================
    # One rewrite maximum — matching current architecture.
    # ========================================================

    if not state[
        "evidence_sufficient"
    ]:

        state.update(
            nodes.rewrite_query(
                state
            )
        )


        state.update(
            nodes.retrieve(
                state
            )
        )


        state.update(
            nodes.build_context(
                state
            )
        )


        state.update(
            nodes.grade_evidence(
                state
            )
        )


        attempts.append(
            {
                "attempt":
                    state[
                        "retry_count"
                    ],

                "query":
                    state[
                        "current_query"
                    ],

                "evidence_sufficient":
                    state[
                        "evidence_sufficient"
                    ],

                "evidence_score":
                    state[
                        "evidence_score"
                    ],

                "evidence_reasons":
                    state[
                        "evidence_reasons"
                    ],

                "context":
                    serialize_context(
                        state[
                            "context"
                        ]
                    ),
            }
        )


    return (
        state,
        attempts,
    )


# ============================================================
# Stage-by-stage generation diagnostic
# ============================================================

def diagnose_generation(
    nodes,
    question,
    context,
):

    generator = (
        nodes.generator
    )


    # ========================================================
    # Stage 1
    # Raw Qwen generation
    # ========================================================

    generator._load_generator()


    raw_answer = (
        generator._generate_draft(
            query=
                question,
            context=
                context,
            max_new_tokens=
                160,
        )
    )


    # ========================================================
    # Stage 2
    # NLI grounding
    # ========================================================

    generator._load_claim_grounder()


    grounded = (
        generator
        .claim_grounder
        .ground(
            answer=
                raw_answer,
            context=
                context,
        )
    )


    # ========================================================
    # Stage 3
    # Relevance reranking
    # ========================================================

    relevance = (
        generator
        .relevance_filter
        .filter(
            query=
                question,
            grounded_claims=
                grounded,
        )
    )


    relevant_by_claim = {
        item.claim: item
        for item
        in relevance.relevant_claims
    }


    filtered_by_claim = {
        item.claim: item
        for item
        in relevance.filtered_claims
    }


    claim_rows = []


    for claim in grounded.claims:

        relevance_score = None

        relevance_decision = (
            "NOT_RANKED"
        )


        if (
            claim.claim
            in relevant_by_claim
        ):

            item = (
                relevant_by_claim[
                    claim.claim
                ]
            )

            relevance_score = (
                item.relevance_score
            )

            relevance_decision = (
                "KEEP"
            )


        elif (
            claim.claim
            in filtered_by_claim
        ):

            item = (
                filtered_by_claim[
                    claim.claim
                ]
            )

            relevance_score = (
                item.relevance_score
            )

            relevance_decision = (
                "FILTER"
            )


        claim_rows.append(
            {
                **serialize_grounded_claim(
                    claim
                ),

                "relevance_score":
                    relevance_score,

                "relevance_decision":
                    relevance_decision,
            }
        )


    return {
        "raw_answer":
            raw_answer,

        "extracted_claim_count":
            len(
                grounded.claims
            ),

        "supported_count":
            grounded.supported_count,

        "unsupported_count":
            grounded.unsupported_count,

        "relevant_count":
            len(
                relevance.relevant_claims
            ),

        "filtered_relevant_count":
            len(
                relevance.filtered_claims
            ),

        "claims":
            claim_rows,
    }


# ============================================================
# Pretty console output
# ============================================================

def print_generation_diagnostic(
    diagnostic,
):

    print(
        "\n===== RAW GENERATION ====="
    )

    print(
        diagnostic[
            "raw_answer"
        ]
    )


    print(
        "\n===== CLAIM PIPELINE ====="
    )


    for index, claim in enumerate(
        diagnostic[
            "claims"
        ],
        start=1,
    ):

        print(
            f"\nCLAIM {index}"
        )

        print(
            "Text:",
            claim[
                "claim"
            ]
        )

        print(
            "Supported:",
            claim[
                "supported"
            ]
        )

        print(
            "NLI label:",
            claim[
                "label"
            ]
        )

        print(
            "Entailment:",
            claim[
                "entailment_score"
            ]
        )

        print(
            "Citation:",
            claim[
                "citation_id"
            ]
        )

        print(
            "Relevance:",
            claim[
                "relevance_score"
            ]
        )

        print(
            "Decision:",
            claim[
                "relevance_decision"
            ]
        )


        if claim[
            "supporting_text"
        ]:

            print(
                "Best premise:"
            )

            print(
                claim[
                    "supporting_text"
                ]
            )


# ============================================================
# Main
# ============================================================

def main():

    examples = (
        load_examples()
    )


    selected = (
        find_target_examples(
            examples
        )
    )


    if not selected:

        raise RuntimeError(
            "No diagnostic targets "
            "were found."
        )


    nodes = (
        RAGNodes()
    )


    output = []


    try:

        for (
            target_name,
            example,
        ) in selected:

            question = (
                example.get(
                    "question"
                )
                or
                example.get(
                    "query"
                )
                or
                ""
            )


            print(
                "\n"
                +
                "=" * 100
            )

            print(
                target_name.upper()
            )

            print(
                "=" * 100
            )

            print(
                "Question:"
            )

            print(
                question
            )


            state, attempts = (
                prepare_evidence(
                    nodes,
                    question,
                )
            )


            print(
                "\n===== EVIDENCE ====="
            )


            for attempt in attempts:

                print(
                    (
                        f"Attempt "
                        f"{attempt['attempt']} | "
                        f"score="
                        f"{attempt['evidence_score']:.4f} | "
                        f"sufficient="
                        f"{attempt['evidence_sufficient']}"
                    )
                )

                print(
                    "Query:",
                    attempt[
                        "query"
                    ]
                )


            record = {
                "target":
                    target_name,

                "example_id":
                    example.get(
                        "id"
                    ),

                "question":
                    question,

                "gold_answer":
                    example.get(
                        "answer"
                    ),

                "question_type":
                    example.get(
                        "question_type"
                    ),

                "evidence_attempts":
                    attempts,

                "final_evidence_sufficient":
                    state[
                        "evidence_sufficient"
                    ],
            }


            # =================================================
            # If the EvidenceGrader rejects the case,
            # do NOT force generation.
            #
            # That is an evidence-gate failure, not a
            # generation/grounding failure.
            # =================================================

            if not state[
                "evidence_sufficient"
            ]:

                print(
                    "\nSKIP GENERATION:"
                )

                print(
                    "Final evidence gate "
                    "is insufficient."
                )


                record[
                    "generation"
                ] = None


                output.append(
                    record
                )

                continue


            diagnostic = (
                diagnose_generation(
                    nodes=
                        nodes,
                    question=
                        question,
                    context=
                        state[
                            "context"
                        ],
                )
            )


            print_generation_diagnostic(
                diagnostic
            )


            record[
                "generation"
            ] = (
                diagnostic
            )


            output.append(
                record
            )


    finally:

        nodes.close()


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "records":
                    output
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "SAVED"
    )

    print(
        "=" * 100
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()