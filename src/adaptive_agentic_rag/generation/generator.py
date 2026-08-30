import re

from dataclasses import dataclass

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
)

from adaptive_agentic_rag.generation.prompts import (
    build_grounded_messages,
)

from adaptive_agentic_rag.generation.claim_grounder import (
    ClaimGrounder,
)

from adaptive_agentic_rag.generation.relevance_filter import (
    ClaimRelevanceFilter,
)

from adaptive_agentic_rag.generation.citation import (
    validate_citations,
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


INSUFFICIENT_TOKEN = (
    "INSUFFICIENT_EVIDENCE"
)


# ============================================================
# Parsed LLM draft
# ============================================================

@dataclass
class ParsedDraft:

    direct_answer: str | None

    evidence_claims: list[str]


# ============================================================
# Final generation result
# ============================================================

@dataclass
class GenerationResult:

    answer: str

    raw_answer: str | None

    direct_answer: str | None

    abstained: bool

    cited_ids: list[int]

    invalid_citation_ids: list[int]

    citation_valid: bool

    model_name: str | None

    # --------------------------------------------------------
    # Draft metrics
    # --------------------------------------------------------

    draft_claims: int

    # --------------------------------------------------------
    # Grounding metrics
    # --------------------------------------------------------

    supported_claims: int

    unsupported_claims: int

    # --------------------------------------------------------
    # Relevance filtering metrics
    # --------------------------------------------------------

    relevant_claims: int

    filtered_irrelevant_claims: int


class GroundedGenerator:

    def __init__(
        self,
        reranker,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_relevant_claims: int = 2,
    ):

        self.model_name = (
            model_name
        )


        self.reranker = (
            reranker
        )


        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        self.device = (
            device
        )


        # ====================================================
        # Lazy-loaded generator components
        # ====================================================

        self.tokenizer = None

        self.model = None

        self.claim_grounder = None


        # ====================================================
        # Reuse the BGE reranker already loaded by retrieval.
        # ====================================================

        self.relevance_filter = (
            ClaimRelevanceFilter(
                reranker=
                    reranker,
                max_relevant_claims=
                    max_relevant_claims,
            )
        )


    # ========================================================
    # Load generator
    # ========================================================

    def _load_generator(
        self,
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
                dtype=dtype,
            )
        )


        self.model.to(
            self.device
        )


        self.model.eval()


    # ========================================================
    # Load NLI Grounder
    # ========================================================

    def _load_claim_grounder(
        self,
    ):

        if (
            self.claim_grounder
            is not None
        ):

            return


        self.claim_grounder = (
            ClaimGrounder(
                reranker=
                    self.reranker,
                device=
                    self.device,
                max_candidate_units=
                    6,
            )
        )


    # ========================================================
    # Generate structured draft
    # ========================================================

    def _generate_draft(
        self,
        query: str,
        context: BuiltContext,
        max_new_tokens: int,
    ) -> str:

        messages = (
            build_grounded_messages(
                query=
                    query,
                context=
                    context,
            )
        )


        prompt = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


        inputs = (
            self.tokenizer(
                prompt,
                return_tensors="pt",
            )
        )


        inputs = {

            key:
                value.to(
                    self.device
                )

            for (
                key,
                value,
            ) in inputs.items()
        }


        input_length = (
            inputs[
                "input_ids"
            ].shape[
                1
            ]
        )


        with torch.no_grad():

            output = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=
                        max_new_tokens,
                    do_sample=False,
                    pad_token_id=(
                        self.tokenizer
                        .eos_token_id
                    ),
                )
            )


        generated_tokens = (
            output[
                0
            ][
                input_length:
            ]
        )


        return (
            self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )


    # ========================================================
    # Draft cleaning
    # ========================================================

    @staticmethod
    def _clean_generated_text(
        text: str,
    ) -> str:

        text = (
            text
            or ""
        )


        text = re.sub(
            r"\[\d+\]",
            "",
            text,
        )


        text = re.sub(
            r"\s+",
            " ",
            text,
        )


        return (
            text.strip()
        )


    # ========================================================
    # Parse structured Qwen output
    #
    # Expected:
    #
    # DIRECT_ANSWER: Yes ...
    #
    # FACTS:
    # - ...
    # - ...
    #
    # We keep a conservative fallback for occasional format
    # deviations from a small local model.
    # ========================================================

    @classmethod
    def _parse_draft(
        cls,
        raw_answer: str,
    ) -> ParsedDraft:

        direct_answer = None

        evidence_claims = []

        inside_facts = False


        for raw_line in (
            raw_answer.splitlines()
        ):

            line = (
                raw_line.strip()
            )


            if not line:

                continue


            # =================================================
            # Direct answer
            # =================================================

            direct_match = re.match(
                r"^DIRECT_ANSWER\s*:\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )


            if direct_match:

                value = (
                    cls._clean_generated_text(
                        direct_match.group(
                            1
                        )
                    )
                )


                if value:

                    direct_answer = (
                        value
                    )


                inside_facts = False

                continue


            # =================================================
            # FACTS header
            # =================================================

            facts_match = re.match(
                r"^FACTS\s*:\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )


            if facts_match:

                inside_facts = True


                inline_fact = (
                    cls._clean_generated_text(
                        facts_match.group(
                            1
                        )
                    )
                )


                if inline_fact:

                    evidence_claims.append(
                        inline_fact
                    )


                continue


            # =================================================
            # Bullet fact
            # =================================================

            bullet_match = re.match(
                r"^[-*•]\s*(.+)$",
                line,
            )


            if bullet_match:

                fact = (
                    cls._clean_generated_text(
                        bullet_match.group(
                            1
                        )
                    )
                )


                if fact:

                    evidence_claims.append(
                        fact
                    )


                continue


            # =================================================
            # Small-model fallback:
            #
            # Once FACTS has begun, accept a short standalone
            # line even if the bullet marker is missing.
            # =================================================

            if inside_facts:

                fact = (
                    cls._clean_generated_text(
                        line
                    )
                )


                if fact:

                    evidence_claims.append(
                        fact
                    )


        # ====================================================
        # Legacy fallback:
        #
        # If Qwen ignored the new contract completely but
        # produced ordinary bullet points, keep those bullets.
        # ====================================================

        if not evidence_claims:

            for raw_line in (
                raw_answer.splitlines()
            ):

                bullet_match = re.match(
                    r"^\s*[-*•]\s*(.+)$",
                    raw_line,
                )


                if not bullet_match:

                    continue


                fact = (
                    cls._clean_generated_text(
                        bullet_match.group(
                            1
                        )
                    )
                )


                if fact:

                    evidence_claims.append(
                        fact
                    )


        # ====================================================
        # Deduplicate preserving order
        # ====================================================

        evidence_claims = list(
            dict.fromkeys(
                evidence_claims
            )
        )


        return ParsedDraft(
            direct_answer=
                direct_answer,
            evidence_claims=
                evidence_claims,
        )


    # ========================================================
    # Convert parsed facts to the format expected by
    # ClaimGrounder.
    # ========================================================

    @staticmethod
    def _render_claims_for_grounding(
        claims: list[str],
    ) -> str:

        return "\n".join(

            f"- {claim}"

            for claim
            in claims
        )


    # ========================================================
    # Final answer construction
    #
    # DIRECT ANSWER:
    # synthesis / requested answer target
    #
    # FACTS:
    # independently grounded support
    # ========================================================

    @staticmethod
    def _build_grounded_answer(
        direct_answer: str | None,
        relevant_claims,
    ) -> str:

        lines = []


        citation_ids = list(
            dict.fromkeys(

                claim.citation_id

                for claim
                in relevant_claims

                if (
                    claim.citation_id
                    is not None
                )
            )
        )


        citation_suffix = "".join(

            f"[{citation_id}]"

            for citation_id
            in citation_ids
        )


        # ====================================================
        # Direct answer
        # ====================================================

        if direct_answer:

            direct_answer = (
                direct_answer.strip()
            )


            if direct_answer:

                line = (
                    direct_answer
                )


                if citation_suffix:

                    line += (
                        f" {citation_suffix}"
                    )


                lines.append(
                    line
                )


        # ====================================================
        # Grounded support
        # ====================================================

        for claim in relevant_claims:

            lines.append(
                (
                    f"- {claim.claim} "
                    f"[{claim.citation_id}]"
                )
            )


        return "\n".join(
            lines
        )


    # ========================================================
    # Main pipeline
    # ========================================================

    def generate(
        self,
        query: str,
        context: BuiltContext,
        evidence_sufficient: bool,
        max_new_tokens: int = 220,
    ) -> GenerationResult:

        # ====================================================
        # Gate 1:
        # Evidence already rejected upstream.
        # ====================================================

        if not evidence_sufficient:

            return GenerationResult(
                answer=
                    ABSTENTION_MESSAGE,
                raw_answer=
                    None,
                direct_answer=
                    None,
                abstained=
                    True,
                cited_ids=
                    [],
                invalid_citation_ids=
                    [],
                citation_valid=
                    True,
                model_name=
                    None,
                draft_claims=
                    0,
                supported_claims=
                    0,
                unsupported_claims=
                    0,
                relevant_claims=
                    0,
                filtered_irrelevant_claims=
                    0,
            )


        # ====================================================
        # Step 1:
        # Structured generation
        # ====================================================

        self._load_generator()


        raw_answer = (
            self._generate_draft(
                query=
                    query,
                context=
                    context,
                max_new_tokens=
                    max_new_tokens,
            )
        )


        print(
            "\n===== RAW GENERATION ====="
        )

        print(
            raw_answer
        )


        # ====================================================
        # Step 2:
        # Parse direct answer separately from evidence facts.
        # ====================================================

        parsed = (
            self._parse_draft(
                raw_answer
            )
        )


        print(
            "\n===== PARSED DRAFT ====="
        )

        print(
            "Direct answer:",
            parsed.direct_answer,
        )

        print(
            "Evidence claims:",
            len(
                parsed.evidence_claims
            ),
        )


        for claim in (
            parsed.evidence_claims
        ):

            print(
                "FACT:",
                claim,
            )


        # ====================================================
        # Explicit model-side abstention
        # ====================================================

        if (
            parsed.direct_answer
            and
            parsed.direct_answer
            .strip()
            .upper()
            ==
            INSUFFICIENT_TOKEN
        ):

            return GenerationResult(
                answer=
                    ABSTENTION_MESSAGE,
                raw_answer=
                    raw_answer,
                direct_answer=
                    parsed.direct_answer,
                abstained=
                    True,
                cited_ids=
                    [],
                invalid_citation_ids=
                    [],
                citation_valid=
                    True,
                model_name=
                    self.model_name,
                draft_claims=
                    len(
                        parsed.evidence_claims
                    ),
                supported_claims=
                    0,
                unsupported_claims=
                    0,
                relevant_claims=
                    0,
                filtered_irrelevant_claims=
                    0,
            )


        # ====================================================
        # No evidence facts produced.
        # ====================================================

        if not parsed.evidence_claims:

            return GenerationResult(
                answer=
                    GROUNDING_FAILURE_MESSAGE,
                raw_answer=
                    raw_answer,
                direct_answer=
                    parsed.direct_answer,
                abstained=
                    True,
                cited_ids=
                    [],
                invalid_citation_ids=
                    [],
                citation_valid=
                    False,
                model_name=
                    self.model_name,
                draft_claims=
                    0,
                supported_claims=
                    0,
                unsupported_claims=
                    0,
                relevant_claims=
                    0,
                filtered_irrelevant_claims=
                    0,
            )


        # ====================================================
        # Step 3:
        # Ground ONLY evidence facts.
        #
        # The direct answer is synthesis and must not pollute
        # atomic factual verification.
        # ====================================================

        self._load_claim_grounder()


        grounding_input = (
            self._render_claims_for_grounding(
                parsed.evidence_claims
            )
        )


        grounded_claims = (
            self.claim_grounder.ground(
                answer=
                    grounding_input,
                context=
                    context,
            )
        )


        # ====================================================
        # Gate 2:
        # No factual support survived.
        # ====================================================

        if (
            grounded_claims.supported_count
            ==
            0
        ):

            return GenerationResult(
                answer=
                    GROUNDING_FAILURE_MESSAGE,
                raw_answer=
                    raw_answer,
                direct_answer=
                    parsed.direct_answer,
                abstained=
                    True,
                cited_ids=
                    [],
                invalid_citation_ids=
                    [],
                citation_valid=
                    False,
                model_name=
                    self.model_name,
                draft_claims=
                    len(
                        parsed.evidence_claims
                    ),
                supported_claims=
                    0,
                unsupported_claims=(
                    grounded_claims
                    .unsupported_count
                ),
                relevant_claims=
                    0,
                filtered_irrelevant_claims=
                    0,
            )


        # ====================================================
        # Step 4:
        # Relevance filter over supported FACTS only.
        # ====================================================

        relevance_result = (
            self.relevance_filter.filter(
                query=
                    query,
                grounded_claims=
                    grounded_claims,
            )
        )


        print(
            "\n===== CLAIM RELEVANCE ====="
        )


        print(
            "Relevant claims:",
            len(
                relevance_result
                .relevant_claims
            ),
        )


        print(
            "Filtered irrelevant claims:",
            len(
                relevance_result
                .filtered_claims
            ),
        )


        for claim in (
            relevance_result
            .relevant_claims
        ):

            print(
                "KEEP:",
                claim.relevance_score,
                claim.claim,
            )


        for claim in (
            relevance_result
            .filtered_claims
        ):

            print(
                "FILTER:",
                claim.relevance_score,
                claim.claim,
            )


        # ====================================================
        # Gate 3:
        # Supported facts existed but none answered the query.
        # ====================================================

        if not (
            relevance_result
            .relevant_claims
        ):

            return GenerationResult(
                answer=
                    GROUNDING_FAILURE_MESSAGE,
                raw_answer=
                    raw_answer,
                direct_answer=
                    parsed.direct_answer,
                abstained=
                    True,
                cited_ids=
                    [],
                invalid_citation_ids=
                    [],
                citation_valid=
                    False,
                model_name=
                    self.model_name,
                draft_claims=
                    len(
                        parsed.evidence_claims
                    ),
                supported_claims=(
                    grounded_claims
                    .supported_count
                ),
                unsupported_claims=(
                    grounded_claims
                    .unsupported_count
                ),
                relevant_claims=
                    0,
                filtered_irrelevant_claims=(
                    len(
                        relevance_result
                        .filtered_claims
                    )
                ),
            )


        # ====================================================
        # Step 5:
        # Direct answer + grounded evidence facts.
        # ====================================================

        final_answer = (
            self._build_grounded_answer(
                direct_answer=
                    parsed.direct_answer,
                relevant_claims=(
                    relevance_result
                    .relevant_claims
                ),
            )
        )


        # ====================================================
        # Step 6:
        # Citation validation.
        # ====================================================

        citation_validation = (
            validate_citations(
                answer=
                    final_answer,
                context=
                    context,
            )
        )


        if not (
            citation_validation.valid
        ):

            return GenerationResult(
                answer=
                    GROUNDING_FAILURE_MESSAGE,
                raw_answer=
                    raw_answer,
                direct_answer=
                    parsed.direct_answer,
                abstained=
                    True,
                cited_ids=(
                    citation_validation
                    .cited_ids
                ),
                invalid_citation_ids=(
                    citation_validation
                    .invalid_ids
                ),
                citation_valid=
                    False,
                model_name=
                    self.model_name,
                draft_claims=
                    len(
                        parsed.evidence_claims
                    ),
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
                ),
            )


        # ====================================================
        # Success
        # ====================================================

        return GenerationResult(
            answer=
                final_answer,
            raw_answer=
                raw_answer,
            direct_answer=
                parsed.direct_answer,
            abstained=
                False,
            cited_ids=(
                citation_validation
                .cited_ids
            ),
            invalid_citation_ids=(
                citation_validation
                .invalid_ids
            ),
            citation_valid=
                True,
            model_name=
                self.model_name,
            draft_claims=
                len(
                    parsed.evidence_claims
                ),
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
            ),
        )