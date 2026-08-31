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
# Structured generation
# ============================================================

@dataclass
class DraftFact:

    text: str

    # --------------------------------------------------------
    # Generator never chooses trusted citations.
    #
    # Grounder assigns citation IDs after verification.
    #
    # This field remains only for diagnostic compatibility.
    # --------------------------------------------------------

    citation_id: int | None = None


@dataclass
class ParsedDraft:

    direct_answer: str | None

    evidence_facts: list[DraftFact]


    @property
    def evidence_claims(
        self,
    ) -> list[str]:

        return [
            fact.text

            for fact
            in self.evidence_facts
        ]


# ============================================================
# Final generation result
# ============================================================

@dataclass
class GenerationResult:

    answer: str

    raw_answer: str | None

    direct_answer: str | None

    # --------------------------------------------------------
    # Evaluation compatibility.
    #
    # In the production single-pass architecture,
    # draft_direct_answer == direct_answer.
    # --------------------------------------------------------

    draft_direct_answer: str | None

    abstained: bool

    cited_ids: list[int]

    invalid_citation_ids: list[int]

    citation_valid: bool

    model_name: str | None

    draft_claims: int

    supported_claims: int

    unsupported_claims: int

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
        # Lazy-loaded components
        # ====================================================

        self.tokenizer = None

        self.model = None

        self.claim_grounder = None


        # ====================================================
        # ClaimRelevanceFilter V2
        #
        # IMPORTANT:
        #
        # This is the production relevance filter containing:
        #
        # - supported-claim filtering
        # - calibrated catastrophic relevance safety floor
        # - top-k selection among surviving claims
        #
        # Do not replace this with the old top-k-only filter.
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
    # Generator loading
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
    # Claim grounder loading
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

                    do_sample=
                        False,

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
    # Generated-text normalization
    # ========================================================

    @staticmethod
    def _clean_generated_text(
        text: str,
    ) -> str:

        text = (
            text
            or ""
        )


        # ----------------------------------------------------
        # Qwen is NOT trusted to assign citations.
        #
        # Grounder assigns citations later.
        # ----------------------------------------------------

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
    # Explicit model abstention normalization
    # ========================================================

    @staticmethod
    def _is_insufficient_answer(
        answer: str | None,
    ) -> bool:

        if not answer:

            return False


        normalized = (
            answer
            .strip()
            .strip(
                " .,:;!?"
            )
            .upper()
        )


        normalized = re.sub(
            r"[\s_-]+",
            "_",
            normalized,
        )


        return normalized in {
            "INSUFFICIENT_EVIDENCE",
            "INSUFFICIENT",
        }


    # ========================================================
    # Parse structured model output
    #
    # Expected production format:
    #
    # DIRECT_ANSWER: ...
    #
    # FACTS:
    # - ...
    # - ...
    #
    #
    # Legacy bullet-only output remains supported because
    # generation contract tests depend on this compatibility.
    # ========================================================

    @classmethod
    def _parse_draft(
        cls,
        raw_answer: str,
    ) -> ParsedDraft:

        direct_answer = None

        evidence_facts = []

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
            # DIRECT_ANSWER
            #
            # DRAFT_ANSWER remains supported for compatibility
            # with older diagnostic artifacts.
            # =================================================

            direct_match = re.match(
                (
                    r"^(?:DIRECT_ANSWER|DRAFT_ANSWER)"
                    r"\s*:\s*(.*)$"
                ),
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

                    evidence_facts.append(
                        DraftFact(
                            text=
                                inline_fact,
                        )
                    )


                continue


            # =================================================
            # Bullet fact
            #
            # Legacy bullet-only model output is intentionally
            # still accepted.
            # =================================================

            bullet_match = re.match(
                r"^[-*â€¢]\s*(.+)$",
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

                    evidence_facts.append(
                        DraftFact(
                            text=
                                fact,
                        )
                    )


                continue


            # =================================================
            # Small-model fallback:
            #
            # After FACTS:, accept a standalone line even when
            # the model forgets the bullet marker.
            # =================================================

            if inside_facts:

                fact = (
                    cls._clean_generated_text(
                        line
                    )
                )


                if fact:

                    evidence_facts.append(
                        DraftFact(
                            text=
                                fact,
                        )
                    )


        # ====================================================
        # Deduplicate facts while preserving generation order
        # ====================================================

        unique = []

        seen = set()


        for fact in (
            evidence_facts
        ):

            key = (
                fact.text
            )


            if key in seen:

                continue


            seen.add(
                key
            )


            unique.append(
                fact
            )


        return ParsedDraft(
            direct_answer=
                direct_answer,

            evidence_facts=
                unique,
        )


    # ========================================================
    # Convert FACTS into ClaimGrounder input
    # ========================================================

    @staticmethod
    def _render_claims_for_grounding(
        facts: list[DraftFact],
    ) -> str:

        return "\n".join(
            (
                f"- {fact.text}"
            )

            for fact
            in facts
        )


    # ========================================================
    # Build final grounded answer
    #
    # DIRECT_ANSWER still comes from Qwen.
    #
    # Supporting facts come ONLY from:
    #
    # supported
    # +
    # relevant
    #
    # grounded claims.
    # ========================================================

    @staticmethod
    def _build_grounded_answer(
        direct_answer: str,
        relevant_claims,
    ) -> str:

        lines = []


        citation_ids = list(
            dict.fromkeys(
                (
                    claim.citation_id
                )

                for claim
                in relevant_claims

                if (
                    claim.citation_id
                    is not None
                )
            )
        )


        citation_suffix = "".join(
            (
                f"[{citation_id}]"
            )

            for citation_id
            in citation_ids
        )


        direct_line = (
            direct_answer.strip()
        )


        if citation_suffix:

            direct_line += (
                f" {citation_suffix}"
            )


        lines.append(
            direct_line
        )


        for claim in (
            relevant_claims
        ):

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
    # Main generation pipeline
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
        # Evidence Grader rejected current evidence.
        #
        # Do not load Qwen.
        # ====================================================

        if not evidence_sufficient:

            return GenerationResult(
                answer=
                    ABSTENTION_MESSAGE,

                raw_answer=
                    None,

                direct_answer=
                    None,

                draft_direct_answer=
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
        # Structured single-pass generation
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
        # Parse DIRECT_ANSWER separately from FACTS
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
            "Evidence facts:",
            len(
                parsed.evidence_facts
            ),
        )


        for fact in (
            parsed.evidence_facts
        ):

            print(
                "FACT:",
                fact.text,
            )


        # ====================================================
        # Gate 2:
        # Generator explicitly says evidence is insufficient.
        # ====================================================

        if (
            self._is_insufficient_answer(
                parsed.direct_answer
            )
        ):

            return GenerationResult(
                answer=
                    ABSTENTION_MESSAGE,

                raw_answer=
                    raw_answer,

                direct_answer=
                    None,

                draft_direct_answer=
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
                        parsed.evidence_facts
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
        # Gate 3:
        # Structured output contract failed.
        # ====================================================

        if not (
            parsed.direct_answer
        ):

            return GenerationResult(
                answer=
                    GROUNDING_FAILURE_MESSAGE,

                raw_answer=
                    raw_answer,

                direct_answer=
                    None,

                draft_direct_answer=
                    None,

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
                        parsed.evidence_facts
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


        if not (
            parsed.evidence_facts
        ):

            return GenerationResult(
                answer=
                    GROUNDING_FAILURE_MESSAGE,

                raw_answer=
                    raw_answer,

                direct_answer=
                    parsed.direct_answer,

                draft_direct_answer=
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
        # Ground FACTS only.
        #
        # DIRECT_ANSWER does NOT enter ClaimGrounder/NLI.
        # ====================================================

        self._load_claim_grounder()


        grounding_input = (
            self._render_claims_for_grounding(
                parsed.evidence_facts
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
        # Gate 4:
        # None of the generated evidence facts are supported.
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

                draft_direct_answer=
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
                        parsed.evidence_facts
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
        # Query relevance over SUPPORTED facts only.
        #
        # ClaimRelevanceFilter V2 performs:
        #
        # supported
        #   ↓
        # malformed guard
        #   ↓
        # BGE rerank
        #   ↓
        # calibrated safety floor
        #   ↓
        # top-k surviving claims
        # ====================================================

        relevance_result = (
            self.relevance_filter.filter(
                query=
                    query,

                grounded_claims=
                    grounded_claims,

                context=
                    context,
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
        print(
            "Selection mode:",
            relevance_result
            .selection_mode,
        )


        print(
            "Adaptive budget:",
            relevance_result
            .adaptive_budget,
        )


        print(
            "Required sources:",
            relevance_result
            .required_sources,
        )


        print(
            "Covered sources:",
            relevance_result
            .covered_sources,
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
        # Gate 5:
        # Supported claims exist but none are sufficiently
        # relevant to the original question.
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

                draft_direct_answer=
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
                        parsed.evidence_facts
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
        # Build final answer.
        #
        # IMPORTANT:
        #
        # No AnswerConsistencyGuard.
        #
        # No ConstrainedAnswerSynthesizer.
        #
        # No second Qwen pass.
        #
        # No Yes/No logit scoring.
        #
        # Production baseline:
        #
        # original DIRECT_ANSWER
        # +
        # verified relevant facts
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
        # Citation validation
        # ====================================================

        citation_validation = (
            validate_citations(
                answer=
                    final_answer,

                context=
                    context,
            )
        )


        # ====================================================
        # Gate 6:
        # Invalid citation structure
        # ====================================================

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

                draft_direct_answer=
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
                        parsed.evidence_facts
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

            draft_direct_answer=
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
                    parsed.evidence_facts
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