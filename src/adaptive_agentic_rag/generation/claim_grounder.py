import re
from dataclasses import dataclass

import numpy as np
import torch

from sentence_transformers import CrossEncoder

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)

from adaptive_agentic_rag.generation.atomic_claim_extractor import (
    AtomicClaimExtractor
)


NLI_MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-small"
)


LABELS = [
    "contradiction",
    "entailment",
    "neutral"
]


@dataclass
class ClaimSupport:

    claim: str

    supported: bool

    citation_id: int | None

    label: str

    entailment_score: float

    supporting_text: str | None


@dataclass
class GroundedClaims:

    claims: list[ClaimSupport]

    supported_count: int

    unsupported_count: int


class ClaimGrounder:


    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        device: str | None = None
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        self.device = device


        print(
            f"Loading NLI model: "
            f"{model_name}"
        )


        self.model = CrossEncoder(
            model_name,
            device=device
        )


        #
        # Atomic claim extraction
        #

        self.atomic_extractor = (
            AtomicClaimExtractor()
        )


    # =========================================================
    # Extract atomic claims from generated answer
    # =========================================================

    def extract_claims(
        self,
        answer: str
    ) -> list[str]:


        atomic_claims = []


        #
        # Qwen normally produces bullet points.
        #
        # We process each bullet independently first.
        #

        for line in answer.splitlines():

            line = line.strip()


            if not line:

                continue


            #
            # Ignore common non-answer headings
            #

            if line.lower() in {
                "answer:",
                "response:",
                "final answer:"
            }:

                continue


            #
            # Remove bullet marker
            #

            line = re.sub(
                r"^[-*•]\s*",
                "",
                line
            )


            #
            # Existing citations are irrelevant
            # during claim verification.
            #

            line = re.sub(
                r"\[\d+\]",
                "",
                line
            )


            line = line.strip()


            if not line:

                continue


            #
            # -----------------------------
            # NEW:
            # Atomic claim extraction
            # -----------------------------
            #

            extraction = (
                self.atomic_extractor.extract(
                    line
                )
            )


            atomic_claims.extend(
                extraction.claims
            )


        #
        # Deduplicate while
        # preserving order
        #

        return list(
            dict.fromkeys(
                atomic_claims
            )
        )


    # =========================================================
    # Split evidence chunk into sentences
    # =========================================================

    def _split_sentences(
        self,
        text: str
    ) -> list[str]:


        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()


        if not text:

            return []


        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )


        return [

            sentence.strip()

            for sentence in sentences

            if len(
                sentence.strip()
            ) >= 20

        ]


    # =========================================================
    # Build small evidence units
    #
    # Instead of:
    #
    # claim ↔ 300-word chunk
    #
    # we compare:
    #
    # claim ↔ sentence
    # claim ↔ two-sentence window
    # =========================================================

    def _build_evidence_units(
        self,
        context: BuiltContext
    ) -> list[dict]:


        units = []


        for item in context.items:

            sentences = (
                self._split_sentences(
                    item.text
                )
            )


            for index, sentence in enumerate(
                sentences
            ):


                #
                # Single sentence
                #

                units.append(
                    {
                        "citation_id":
                            item.citation_id,

                        "text":
                            sentence
                    }
                )


                #
                # Two-sentence window
                #

                if (
                    index + 1
                    <
                    len(sentences)
                ):

                    window = (
                        sentence
                        + " "
                        + sentences[
                            index + 1
                        ]
                    )


                    units.append(
                        {
                            "citation_id":
                                item.citation_id,

                            "text":
                                window
                        }
                    )


        return units


    # =========================================================
    # Check ONE atomic claim against evidence
    # =========================================================

    def _check_claim(
        self,
        claim: str,
        context: BuiltContext
    ) -> ClaimSupport:


        evidence_units = (
            self._build_evidence_units(
                context
            )
        )


        if not evidence_units:

            return ClaimSupport(

                claim=claim,

                supported=False,

                citation_id=None,

                label="neutral",

                entailment_score=0.0,

                supporting_text=None
            )


        #
        # NLI format:
        #
        # premise    = evidence
        # hypothesis = claim
        #

        pairs = [

            (
                unit["text"],
                claim
            )

            for unit in evidence_units

        ]


        probabilities = (
            self.model.predict(

                pairs,

                apply_softmax=True

            )
        )


        probabilities = np.asarray(
            probabilities
        )


        #
        # Label order:
        #
        # 0 = contradiction
        # 1 = entailment
        # 2 = neutral
        #

        entailment_scores = (
            probabilities[:, 1]
        )


        #
        # Find evidence unit with
        # strongest entailment probability
        #

        best_index = int(
            np.argmax(
                entailment_scores
            )
        )


        best_probabilities = (
            probabilities[
                best_index
            ]
        )


        predicted_label_index = int(
            np.argmax(
                best_probabilities
            )
        )


        predicted_label = (
            LABELS[
                predicted_label_index
            ]
        )


        best_unit = (
            evidence_units[
                best_index
            ]
        )


        supported = (
            predicted_label
            ==
            "entailment"
        )


        return ClaimSupport(

            claim=claim,

            supported=supported,

            citation_id=(

                best_unit[
                    "citation_id"
                ]

                if supported

                else None
            ),

            label=(
                predicted_label
            ),

            entailment_score=round(

                float(
                    entailment_scores[
                        best_index
                    ]
                ),

                4
            ),

            supporting_text=(

                best_unit[
                    "text"
                ]

                if supported

                else None
            )
        )


    # =========================================================
    # Ground full generated answer
    # =========================================================

    def ground(
        self,
        answer: str,
        context: BuiltContext
    ) -> GroundedClaims:


        #
        # This now returns ATOMIC claims
        #

        claims = (
            self.extract_claims(
                answer
            )
        )


        results = []


        #
        # Each atomic claim is fact-checked
        # independently.
        #

        for claim in claims:

            result = (
                self._check_claim(

                    claim=claim,

                    context=context

                )
            )


            results.append(
                result
            )


        supported_count = sum(

            1

            for result in results

            if result.supported

        )


        unsupported_count = (

            len(results)

            -

            supported_count

        )


        return GroundedClaims(

            claims=results,

            supported_count=(
                supported_count
            ),

            unsupported_count=(
                unsupported_count
            )
        )