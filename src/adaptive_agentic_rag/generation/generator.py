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
    build_grounded_messages,
    build_citation_repair_messages
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
    "I couldn't produce a sufficiently "
    "grounded answer from the provided evidence."
)


@dataclass
class GenerationResult:

    answer: str

    abstained: bool

    cited_ids: list[int]

    invalid_citation_ids: list[int]

    citation_valid: bool

    model_name: str | None

    citation_repaired: bool

    generation_attempts: int


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

        self.tokenizer = None

        self.model = None


    def _load_model(
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


    def _generate_from_messages(
        self,
        messages: list[dict],
        max_new_tokens: int
    ) -> str:

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

            output = self.model.generate(

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

    def generate(
        self,
        query: str,
        context: BuiltContext,
        evidence_sufficient: bool,
        max_new_tokens: int = 160
    ) -> GenerationResult:

        #
        # ---------------------------------
        # No sufficient evidence
        # ---------------------------------
        #

        if not evidence_sufficient:

            return GenerationResult(

                answer=ABSTENTION_MESSAGE,

                abstained=True,

                cited_ids=[],

                invalid_citation_ids=[],

                citation_valid=True,

                model_name=None,

                citation_repaired=False,

                generation_attempts=0
            )


        #
        # ---------------------------------
        # Load LLM
        # ---------------------------------
        #

        self._load_model()


        #
        # =================================
        # Attempt 1
        # Normal grounded generation
        # =================================
        #

        messages = build_grounded_messages(
            query=query,
            context=context
        )


        answer = self._generate_from_messages(
            messages=messages,
            max_new_tokens=max_new_tokens
        )


        print(
            "\n===== RAW GENERATION ====="
        )

        print(
            answer
        )


        validation = validate_citations(
            answer=answer,
            context=context
        )


        #
        # Generation succeeded immediately
        #

        if validation.valid:

            return GenerationResult(

                answer=answer,

                abstained=False,

                cited_ids=validation.cited_ids,

                invalid_citation_ids=(
                    validation.invalid_ids
                ),

                citation_valid=True,

                model_name=self.model_name,

                citation_repaired=False,

                generation_attempts=1
            )


        #
        # =================================
        # Attempt 2
        # Citation repair
        # =================================
        #

        repair_messages = (
            build_citation_repair_messages(

                query=query,

                context=context,

                draft_answer=answer
            )
        )


        repaired_answer = (
            self._generate_from_messages(

                messages=repair_messages,

                max_new_tokens=max_new_tokens
            )
        )


        print(
            "\n===== RAW REPAIR GENERATION ====="
        )

        print(
            repaired_answer
        )


        repaired_validation = (
            validate_citations(

                answer=repaired_answer,

                context=context
            )
        )


        #
        # Repair succeeded
        #

        if repaired_validation.valid:

            return GenerationResult(

                answer=repaired_answer,

                abstained=False,

                cited_ids=(
                    repaired_validation.cited_ids
                ),

                invalid_citation_ids=(
                    repaired_validation.invalid_ids
                ),

                citation_valid=True,

                model_name=self.model_name,

                citation_repaired=True,

                generation_attempts=2
            )


        #
        # ---------------------------------
        # Both attempts failed
        # ---------------------------------
        #

        return GenerationResult(

            answer=GROUNDING_FAILURE_MESSAGE,

            abstained=True,

            cited_ids=(
                repaired_validation.cited_ids
            ),

            invalid_citation_ids=(
                repaired_validation.invalid_ids
            ),

            citation_valid=False,

            model_name=self.model_name,

            citation_repaired=True,

            generation_attempts=2
        )