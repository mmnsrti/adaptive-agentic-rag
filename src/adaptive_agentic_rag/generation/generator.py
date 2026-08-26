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


@dataclass
class GenerationResult:

    answer: str

    abstained: bool

    cited_ids: list[int]

    invalid_citation_ids: list[int]

    citation_valid: bool

    model_name: str | None


class GroundedGenerator:


    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None
    ):

        self.model_name = (
            model_name
        )


        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        self.device = device


        #
        # Lazy loading:
        #
        # Generator model is NOT loaded here.
        #

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


        if self.device == "cuda":

            dtype = torch.float16

        else:

            dtype = torch.float32


        self.model = (
            AutoModelForCausalLM.from_pretrained(

                self.model_name,

                torch_dtype=dtype

            )
        )


        self.model.to(
            self.device
        )


        self.model.eval()


    def generate(
        self,
        query: str,
        context: BuiltContext,
        evidence_sufficient: bool,
        max_new_tokens: int = 300
    ) -> GenerationResult:


        #
        # =====================================
        # Do NOT generate without evidence
        # =====================================
        #

        if not evidence_sufficient:

            return GenerationResult(

                answer=ABSTENTION_MESSAGE,

                abstained=True,

                cited_ids=[],

                invalid_citation_ids=[],

                citation_valid=True,

                model_name=None

            )


        #
        # Load model only when needed
        #

        self._load_model()


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


        inputs = (
            self.tokenizer(

                prompt,

                return_tensors="pt"

            )
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


        answer = (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True

            )
            .strip()
        )


        citation_validation = (
            validate_citations(

                answer=answer,

                context=context

            )
        )


        return GenerationResult(

            answer=answer,

            abstained=False,

            cited_ids=(
                citation_validation
                .cited_ids
            ),

            invalid_citation_ids=(
                citation_validation
                .invalid_ids
            ),

            citation_valid=(
                citation_validation
                .valid
            ),

            model_name=(
                self.model_name
            )

        )