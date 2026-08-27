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

from adaptive_agentic_rag.generation.relevance_filter import (
    ClaimRelevanceFilter
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
    "I couldn't find enough supported and relevant claims "
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

    #
    # Claim grounding metrics
    #

    supported_claims: int

    unsupported_claims: int

    #
    # Relevance filtering metrics
    #

    relevant_claims: int

    filtered_irrelevant_claims: int


class GroundedGenerator:

    def __init__(
        self,
        reranker,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_relevant_claims: int = 2
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
        # Lazy-loaded generator components
        #

        self.tokenizer = None

        self.model = None

        self.claim_grounder = None


        #
        # Reuse the embedding model
        # already loaded by DenseRetriever
        #

        self.relevance_filter = (
            ClaimRelevanceFilter(

                reranker=reranker,

                max_relevant_claims=(
                    max_relevant_claims
                )

            )
        )


    # =========================================================
    # Load generator model
    # =========================================================

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


    # =========================================================
    # Load NLI Claim Grounder
    # =========================================================

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


    # =========================================================
    # Generate raw LLM draft
    # =========================================================

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


        return (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True

            )
            .strip()
        )


    # =========================================================
    # Build final answer from:
    #
    # supported
    # +
    # relevant
    #
    # claims only
    # =========================================================

    def _build_grounded_answer(
        self,
        relevant_claims
    ) -> str:


        lines = []


        for claim in relevant_claims:

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


    # =========================================================
    # Main generation pipeline
    # =========================================================

    def generate(
        self,
        query: str,
        context: BuiltContext,
        evidence_sufficient: bool,
        max_new_tokens: int = 160
    ) -> GenerationResult:


        # =====================================================
        # Gate 1
        #
        # Evidence Grader already decided
        # that evidence is insufficient.
        #
        # Do NOT run the LLM.
        # =====================================================

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

                unsupported_claims=0,

                relevant_claims=0,

                filtered_irrelevant_claims=0

            )


        # =====================================================
        # Step 1
        # Generate raw answer
        # =====================================================

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


        # =====================================================
        # Step 2
        # Claim Grounding
        #
        # Question:
        # Is each generated claim supported by evidence?
        # =====================================================

        self._load_claim_grounder()


        grounded_claims = (
            self.claim_grounder.ground(

                answer=raw_answer,

                context=context

            )
        )


        # =====================================================
        # Gate 2
        #
        # Nothing generated by the LLM
        # was actually supported.
        # =====================================================

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
                ),

                relevant_claims=0,

                filtered_irrelevant_claims=0

            )


        # =====================================================
        # Step 3
        # Claim-level relevance filtering
        #
        # A claim may be factually supported,
        # but still irrelevant to the user's question.
        # =====================================================

        relevance_result = (
            self.relevance_filter.filter(

                query=query,

                grounded_claims=(
                    grounded_claims
                )

            )
        )


        # =====================================================
        # Debug information
        # =====================================================

        print(
            "\n===== CLAIM RELEVANCE ====="
        )


        print(
            "Relevant claims:",
            len(
                relevance_result
                .relevant_claims
            )
        )


        print(
            "Filtered irrelevant claims:",
            len(
                relevance_result
                .filtered_claims
            )
        )


        for claim in (
            relevance_result
            .relevant_claims
        ):

            print(

                "KEEP:",

                claim.relevance_score,

                claim.claim

            )


        for claim in (
            relevance_result
            .filtered_claims
        ):

            print(

                "FILTER:",

                claim.relevance_score,

                claim.claim

            )


        # =====================================================
        # Gate 3
        #
        # Claims were supported,
        # but none were sufficiently relevant
        # to the original user question.
        # =====================================================

        if not relevance_result.relevant_claims:

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

                supported_claims=(
                    grounded_claims
                    .supported_count
                ),

                unsupported_claims=(
                    grounded_claims
                    .unsupported_count
                ),

                relevant_claims=0,

                filtered_irrelevant_claims=(
                    len(
                        relevance_result
                        .filtered_claims
                    )
                )

            )


        # =====================================================
        # Step 4
        # Build final answer
        #
        # Only:
        #
        # supported
        # AND
        # relevant
        #
        # claims survive.
        # =====================================================

        final_answer = (
            self._build_grounded_answer(

                relevance_result
                .relevant_claims

            )
        )


        # =====================================================
        # Step 5
        # Final citation safety validation
        # =====================================================

        citation_validation = (
            validate_citations(

                answer=final_answer,

                context=context

            )
        )


        # =====================================================
        # Gate 4
        #
        # Citation IDs are invalid.
        # =====================================================

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
                ),

                relevant_claims=(
                    len(
                        relevance_result
                        .relevant_claims
                    )
                ),

                filtered_irrelevant_claims=(
                    len(
                        relevance_result
                        .filtered_claims
                    )
                )

            )


        # =====================================================
        # Success
        # =====================================================

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
            ),

            relevant_claims=(
                len(
                    relevance_result
                    .relevant_claims
                )
            ),

            filtered_irrelevant_claims=(
                len(
                    relevance_result
                    .filtered_claims
                )
            )

        )