import re

from dataclasses import dataclass

import numpy as np
import torch

from sentence_transformers import CrossEncoder

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
)

from adaptive_agentic_rag.generation.atomic_claim_extractor import (
    AtomicClaimExtractor,
)

from adaptive_agentic_rag.generation.sentence_splitter import (
    split_sentences,
)


NLI_MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-small"
)


LABELS = [
    "contradiction",
    "entailment",
    "neutral",
]


@dataclass
class ClaimSupport:

    claim: str

    supported: bool

    citation_id: int | None

    label: str

    entailment_score: float

    supporting_text: str | None

    evidence_relevance_score: float | None = None

    premise_mode: str | None = None


@dataclass
class GroundedClaims:

    claims: list[ClaimSupport]

    supported_count: int

    unsupported_count: int


class ClaimGrounder:

    def __init__(
        self,
        reranker,
        model_name: str = NLI_MODEL_NAME,
        device: str | None = None,
        max_candidate_units: int = 6,
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        self.device = (
            device
        )


        self.reranker = (
            reranker
        )


        self.max_candidate_units = (
            max_candidate_units
        )


        print(
            f"Loading NLI model: "
            f"{model_name}"
        )


        self.model = (
            CrossEncoder(
                model_name,
                device=device,
            )
        )


        self.atomic_extractor = (
            AtomicClaimExtractor()
        )


    # ========================================================
    # Claim extraction
    # ========================================================

    def extract_claims(
        self,
        answer: str,
    ) -> list[str]:

        atomic_claims = []


        for line in answer.splitlines():

            line = (
                line.strip()
            )


            if not line:

                continue


            if (
                line.lower()
                in {
                    "answer:",
                    "response:",
                    "final answer:",
                }
            ):

                continue


            line = re.sub(
                r"^[-*•]\s*",
                "",
                line,
            )


            line = re.sub(
                r"\[\d+\]",
                "",
                line,
            )


            line = (
                line.strip()
            )


            if not line:

                continue


            extraction = (
                self.atomic_extractor
                .extract(
                    line
                )
            )


            atomic_claims.extend(
                extraction.claims
            )


        return list(
            dict.fromkeys(
                atomic_claims
            )
        )


    # ========================================================
    # Evidence sentence splitting
    # ========================================================

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:

        return (
            split_sentences(
                text,
                split_newlines=False,
                min_chars=20,
            )
        )


    # ========================================================
    # Evidence units
    #
    # Each unit keeps:
    #
    # citation
    # source
    # title
    # plain text
    # provenance-aware text
    # ========================================================

    def _build_evidence_units(
        self,
        context: BuiltContext,
    ) -> list[dict]:

        units = []


        for item in context.items:

            sentences = (
                self._split_sentences(
                    item.text
                )
            )


            provenance_parts = []


            if item.source:

                provenance_parts.append(
                    f"Source: {item.source}."
                )


            if item.title:

                provenance_parts.append(
                    f"Title: {item.title}."
                )


            provenance = " ".join(
                provenance_parts
            )


            for (
                index,
                sentence,
            ) in enumerate(
                sentences
            ):

                self._append_unit(
                    units=units,
                    citation_id=
                        item.citation_id,
                    source=
                        item.source,
                    title=
                        item.title,
                    text=
                        sentence,
                    provenance=
                        provenance,
                    unit_type=
                        "sentence",
                )


                if (
                    index + 1
                    <
                    len(
                        sentences
                    )
                ):

                    window = (
                        sentence
                        +
                        " "
                        +
                        sentences[
                            index + 1
                        ]
                    )


                    self._append_unit(
                        units=units,
                        citation_id=
                            item.citation_id,
                        source=
                            item.source,
                        title=
                            item.title,
                        text=
                            window,
                        provenance=
                            provenance,
                        unit_type=
                            "window",
                    )


        return units


    @staticmethod
    def _append_unit(
        *,
        units,
        citation_id,
        source,
        title,
        text,
        provenance,
        unit_type,
    ):

        text = (
            text.strip()
        )


        if not text:

            return


        provenance_text = (
            (
                f"{provenance} "
                f"Evidence: {text}"
            )
            .strip()
        )


        units.append(
            {
                "id":
                    (
                        f"evidence_"
                        f"{len(units)}"
                    ),

                "citation_id":
                    citation_id,

                "source":
                    source or "",

                "title":
                    title or "",

                "text":
                    text,

                "provenance_text":
                    provenance_text,

                "unit_type":
                    unit_type,
            }
        )


    # ========================================================
    # Candidate retrieval BEFORE NLI
    #
    # This avoids:
    #
    # NLI(claim, every sentence)
    # then max(score)
    #
    # which can generate spurious high entailment scores.
    # ========================================================

    def _select_candidate_units(
        self,
        claim: str,
        units: list[dict],
    ) -> list[dict]:

        if not units:

            return []


        documents = []


        for index, unit in enumerate(
            units
        ):

            documents.append(
                {
                    "id":
                        f"grounding_unit_{index}",

                    "text":
                        unit[
                            "provenance_text"
                        ],

                    "unit":
                        unit,
                }
            )


        top_k = min(
            self.max_candidate_units,
            len(
                documents
            ),
        )


        ranked = (
            self.reranker.rerank(
                query=
                    claim,
                documents=
                    documents,
                top_k=
                    top_k,
            )
        )


        candidates = []


        for item in ranked:

            unit = (
                item[
                    "unit"
                ].copy()
            )


            unit[
                "relevance_score"
            ] = round(
                float(
                    item[
                        "rerank_score"
                    ]
                ),
                4,
            )


            candidates.append(
                unit
            )


        return candidates


    # ========================================================
    # Score one premise
    # ========================================================

    def _predict_nli(
        self,
        premise: str,
        claim: str,
    ) -> dict:

        probabilities = (
            self.model.predict(
                [
                    (
                        premise,
                        claim,
                    )
                ],
                apply_softmax=True,
            )
        )


        probabilities = (
            np.asarray(
                probabilities
            )
        )


        if (
            probabilities.ndim
            ==
            2
        ):

            probabilities = (
                probabilities[
                    0
                ]
            )


        label_index = int(
            np.argmax(
                probabilities
            )
        )


        return {
            "label":
                LABELS[
                    label_index
                ],

            "contradiction":
                float(
                    probabilities[
                        0
                    ]
                ),

            "entailment":
                float(
                    probabilities[
                        1
                    ]
                ),

            "neutral":
                float(
                    probabilities[
                        2
                    ]
                ),
        }


    # ========================================================
    # Evaluate candidate evidence
    #
    # We test BOTH:
    #
    # plain evidence
    #
    # and
    #
    # source + title + evidence
    #
    # because the diagnostic showed that neither representation
    # dominates universally.
    # ========================================================

    def _evaluate_candidate(
        self,
        claim: str,
        unit: dict,
    ) -> list[dict]:

        variants = [
            (
                "plain",
                unit[
                    "text"
                ],
            ),
            (
                "provenance",
                unit[
                    "provenance_text"
                ],
            ),
        ]


        results = []


        for (
            mode,
            premise,
        ) in variants:

            prediction = (
                self._predict_nli(
                    premise=
                        premise,
                    claim=
                        claim,
                )
            )


            results.append(
                {
                    "mode":
                        mode,

                    "premise":
                        premise,

                    "citation_id":
                        unit[
                            "citation_id"
                        ],

                    "relevance_score":
                        unit[
                            "relevance_score"
                        ],

                    **prediction,
                }
            )


        return results


    # ========================================================
    # Check one claim
    # ========================================================

    def _check_claim(
        self,
        claim: str,
        context: BuiltContext,
    ) -> ClaimSupport:

        units = (
            self._build_evidence_units(
                context
            )
        )


        if not units:

            return ClaimSupport(
                claim=
                    claim,
                supported=
                    False,
                citation_id=
                    None,
                label=
                    "neutral",
                entailment_score=
                    0.0,
                supporting_text=
                    None,
                evidence_relevance_score=
                    None,
                premise_mode=
                    None,
            )


        candidates = (
            self._select_candidate_units(
                claim=
                    claim,
                units=
                    units,
            )
        )


        if not candidates:

            return ClaimSupport(
                claim=
                    claim,
                supported=
                    False,
                citation_id=
                    None,
                label=
                    "neutral",
                entailment_score=
                    0.0,
                supporting_text=
                    None,
                evidence_relevance_score=
                    None,
                premise_mode=
                    None,
            )


        evaluations = []


        for unit in candidates:

            evaluations.extend(
                self._evaluate_candidate(
                    claim=
                        claim,
                    unit=
                        unit,
                )
            )


        # ====================================================
        # IMPORTANT:
        #
        # We still use "entailment is winning label"
        # as the support rule.
        #
        # No arbitrary entailment threshold is added yet.
        # ====================================================

        entailed = [

            evaluation

            for evaluation
            in evaluations

            if (
                evaluation[
                    "label"
                ]
                ==
                "entailment"
            )
        ]


        if entailed:

            # ------------------------------------------------
            # Among valid entailments:
            #
            # prioritize NLI entailment confidence,
            # then candidate relevance.
            # ------------------------------------------------

            best = max(

                entailed,

                key=lambda item: (
                    item[
                        "entailment"
                    ],
                    item[
                        "relevance_score"
                    ],
                ),
            )


            return ClaimSupport(
                claim=
                    claim,
                supported=
                    True,
                citation_id=
                    best[
                        "citation_id"
                    ],
                label=
                    "entailment",
                entailment_score=
                    round(
                        best[
                            "entailment"
                        ],
                        4,
                    ),
                supporting_text=
                    best[
                        "premise"
                    ],
                evidence_relevance_score=
                    best[
                        "relevance_score"
                    ],
                premise_mode=
                    best[
                        "mode"
                    ],
            )


        # ====================================================
        # Unsupported:
        #
        # Preserve the highest-entailment diagnostic candidate.
        # ====================================================

        best = max(

            evaluations,

            key=lambda item:
                item[
                    "entailment"
                ],
        )


        return ClaimSupport(
            claim=
                claim,
            supported=
                False,
            citation_id=
                None,
            label=
                best[
                    "label"
                ],
            entailment_score=
                round(
                    best[
                        "entailment"
                    ],
                    4,
                ),
            supporting_text=
                best[
                    "premise"
                ],
            evidence_relevance_score=
                best[
                    "relevance_score"
                ],
            premise_mode=
                best[
                    "mode"
                ],
        )


    # ========================================================
    # Ground full generated answer
    # ========================================================

    def ground(
        self,
        answer: str,
        context: BuiltContext,
    ) -> GroundedClaims:

        claims = (
            self.extract_claims(
                answer
            )
        )


        results = []


        for claim in claims:

            result = (
                self._check_claim(
                    claim=
                        claim,
                    context=
                        context,
                )
            )


            results.append(
                result
            )


        supported_count = sum(

            1

            for result
            in results

            if result.supported
        )


        unsupported_count = (
            len(
                results
            )
            -
            supported_count
        )


        return GroundedClaims(
            claims=
                results,
            supported_count=
                supported_count,
            unsupported_count=
                unsupported_count,
        )