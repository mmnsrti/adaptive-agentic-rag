import re
from dataclasses import dataclass

import numpy as np
import torch

from sentence_transformers import CrossEncoder

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
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


    # -------------------------------------------------
    # Extract claims from generated answer
    # -------------------------------------------------

    def extract_claims(
        self,
        answer: str
    ) -> list[str]:

        claims = []


        for line in answer.splitlines():

            line = line.strip()


            if not line:
                continue


            #
            # Remove bullet markers
            #

            line = re.sub(
                r"^[-*•]\s*",
                "",
                line
            )


            #
            # Remove citations such as [1]
            #

            line = re.sub(
                r"\[\d+\]",
                "",
                line
            )


            line = line.strip()


            if line:

                claims.append(
                    line
                )


        return claims


    # -------------------------------------------------
    # Split one chunk into sentences
    # -------------------------------------------------

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


    # -------------------------------------------------
    # Build evidence units
    #
    # Each unit:
    #   sentence
    #   OR
    #   sentence + next sentence
    # -------------------------------------------------

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


    # -------------------------------------------------
    # Ground one claim
    # -------------------------------------------------

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
        # premise = evidence
        # hypothesis = claim
        #

        pairs = [

            (
                unit["text"],
                claim
            )

            for unit
            in evidence_units

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
        # entailment index = 1
        #

        entailment_scores = (
            probabilities[:, 1]
        )


        #
        # Find evidence unit with
        # strongest entailment score
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

            label=predicted_label,

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


    # -------------------------------------------------
    # Ground all claims
    # -------------------------------------------------

    def ground(
        self,
        answer: str,
        context: BuiltContext
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