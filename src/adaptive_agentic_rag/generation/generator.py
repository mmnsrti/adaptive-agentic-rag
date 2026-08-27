from dataclasses import dataclass

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)

from adaptive_agentic_rag.generation.prompts import (
    build_grounded_messages
)

from adaptive_agentic_rag.generation.claim_grounder import (
    ClaimGrounder
)

from adaptive_agentic_rag.generation.citation import (
    validate_citations
)


DEFAULT_MODEL = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)


ABSTENTION_MESSAGE = (
    "I don't have enough evidence in the "
    "provided sources to answer reliably."
)


GROUNDING_FAILURE_MESSAGE = (
    "I couldn't find enough supported claims "
    "in the retrieved evidence to answer reliably."
)


@dataclass
class GenerationResult:

    answer: str

    raw_answer: str | None

    abstained: bool

    cited_ids: list[int]

    invalid_citation_ids: list[int]

    citation_valid: bool

    model_name: str | None

    supported_claims: int

    unsupported_claims: int


class GroundedGenerator:


    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None
    ):

        self.model_name = model_name


        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        self.device = device


        #
        # Lazy loading
        #

        self.tokenizer = None

        self.model = None

        self.claim_grounder = None


    def _load_generator(
        self
    ):

        if (
            self.model is not None
            and
            self.tokenizer is not None
        ):

            return


        print(
            f"Loading generator: "
            f"{self.model_name}"
        )


        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )


        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )


        self.model = (
            AutoModelForCausalLM.from_pretrained(

                self.model_name,

                dtype=dtype

            )
        )


        self.model.to(
            self.device
        )


        self.model.eval()


    def _load_claim_grounder(
        self
    ):

        if self.claim_grounder is not None:

            return


        self.claim_grounder = (
            ClaimGrounder(
                device=self.device
            )
        )


    def _generate_draft(
        self,
        query: str,
        context: BuiltContext,
        max_new_tokens: int
    ) -> str:


        messages = (
            build_grounded_messages(

                query=query,

                context=context

            )
        )


        prompt = (
            self.tokenizer.apply_chat_template(

                messages,

                tokenize=False,

                add_generation_prompt=True

            )
        )


        inputs = self.tokenizer(

            prompt,

            return_tensors="pt"

        )


        inputs = {

            key: value.to(
                self.device
            )

            for key, value
            in inputs.items()

        }


        input_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )


        with torch.no_grad():

            output = (
                self.model.generate(

                    **inputs,

                    max_new_tokens=(
                        max_new_tokens
                    ),

                    do_sample=False,

                    pad_token_id=(
                        self.tokenizer
                        .eos_token_id
                    )

                )
            )


        generated_tokens = (
            output[0][
                input_length:
            ]
        )


        return (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True

            )
            .strip()
        )


    def _build_grounded_answer(
        self,
        grounded_claims
    ) -> str:


        lines = []


        for claim in grounded_claims.claims:


            if not claim.supported:

                continue


            line = (
                f"- {claim.claim} "
                f"[{claim.citation_id}]"
            )


            lines.append(
                line
            )


        return "\n".join(
            lines
        )


    def generate(
        self,
        query: str,
        context: BuiltContext,
        evidence_sufficient: bool,
        max_new_tokens: int = 160
    ) -> GenerationResult:


        #
        # =====================================
        # Gate 1:
        # Evidence is already insufficient
        # =====================================
        #

        if not evidence_sufficient:

            return GenerationResult(

                answer=ABSTENTION_MESSAGE,

                raw_answer=None,

                abstained=True,

                cited_ids=[],

                invalid_citation_ids=[],

                citation_valid=True,

                model_name=None,

                supported_claims=0,

                unsupported_claims=0

            )


        #
        # =====================================
        # Generate raw draft
        # =====================================
        #

        self._load_generator()


        raw_answer = (
            self._generate_draft(

                query=query,

                context=context,

                max_new_tokens=(
                    max_new_tokens
                )

            )
        )


        print(
            "\n===== RAW GENERATION ====="
        )

        print(
            raw_answer
        )


        #
        # =====================================
        # Ground every claim
        # =====================================
        #

        self._load_claim_grounder()


        grounded_claims = (
            self.claim_grounder.ground(

                answer=raw_answer,

                context=context

            )
        )


        #
        # =====================================
        # Gate 2:
        # No supported claims survived
        # =====================================
        #

        if (
            grounded_claims.supported_count
            ==
            0
        ):

            return GenerationResult(

                answer=(
                    GROUNDING_FAILURE_MESSAGE
                ),

                raw_answer=raw_answer,

                abstained=True,

                cited_ids=[],

                invalid_citation_ids=[],

                citation_valid=False,

                model_name=self.model_name,

                supported_claims=0,

                unsupported_claims=(
                    grounded_claims
                    .unsupported_count
                )

            )


        #
        # =====================================
        # Build final answer
        # =====================================
        #

        final_answer = (
            self._build_grounded_answer(
                grounded_claims
            )
        )


        #
        # Final safety validation
        #

        citation_validation = (
            validate_citations(

                answer=final_answer,

                context=context

            )
        )


        if not citation_validation.valid:

            return GenerationResult(

                answer=(
                    GROUNDING_FAILURE_MESSAGE
                ),

                raw_answer=raw_answer,

                abstained=True,

                cited_ids=(
                    citation_validation
                    .cited_ids
                ),

                invalid_citation_ids=(
                    citation_validation
                    .invalid_ids
                ),

                citation_valid=False,

                model_name=self.model_name,

                supported_claims=(
                    grounded_claims
                    .supported_count
                ),

                unsupported_claims=(
                    grounded_claims
                    .unsupported_count
                )

            )


        #
        # =====================================
        # Success
        # =====================================
        #

        return GenerationResult(

            answer=final_answer,

            raw_answer=raw_answer,

            abstained=False,

            cited_ids=(
                citation_validation
                .cited_ids
            ),

            invalid_citation_ids=(
                citation_validation
                .invalid_ids
            ),

            citation_valid=True,

            model_name=self.model_name,

            supported_claims=(
                grounded_claims
                .supported_count
            ),

            unsupported_claims=(
                grounded_claims
                .unsupported_count
            )

        )