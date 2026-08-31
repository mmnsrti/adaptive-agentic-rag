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

from adaptive_agentic_rag.generation.constrained_answer_synthesis import (
    ConstrainedAnswerSynthesizer,
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
# Final result
# ============================================================

@dataclass
class GenerationResult:

    answer: str

    raw_answer: str | None

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # direct_answer is now the FINAL synthesized answer.
    #
    # For yes/no questions it may differ from the original
    # Qwen draft.
    # --------------------------------------------------------

    direct_answer: str | None

    # --------------------------------------------------------
    # Original free-form Qwen answer before constrained
    # synthesis.
    #
    # This allows evaluation such as:
    #
    # draft_direct_answer = "No"
    # direct_answer       = "Yes"
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
        # Supported claims are ranked against the original
        # question.
        #
        # RelevanceFilter V2 includes:
        #
        # calibrated catastrophic relevance safety floor
        # +
        # top-k among surviving claims.
        # ====================================================

        self.relevance_filter = (
            ClaimRelevanceFilter(
                reranker=
                    reranker,

                max_relevant_claims=
                    max_relevant_claims,
            )
        )


        # ====================================================
        # Final constrained synthesis.
        #
        # This component does NOT load another model.
        #
        # It reuses the already-loaded generator model and:
        #
        # yes/no:
        #   scores only "Yes" vs "No"
        #
        # entity:
        #   verifies draft entity against grounded claims
        #
        # other:
        #   preserves draft answer
        # ====================================================

        self.answer_synthesizer = (
            ConstrainedAnswerSynthesizer()
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
    # Grounder loading
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
        # Generator is not trusted to assign citations.
        #
        # Any accidental citation markers are removed.
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
    # Normalize explicit model abstention
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
    # Parse:
    #
    # DIRECT_ANSWER: ...
    #
    # FACTS:
    # - ...
    # - ...
    #
    #
    # Keep baseline behavior for compatibility.
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
            # Direct answer
            #
            # DRAFT_ANSWER remains accepted for compatibility
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
            # Baseline parser intentionally supports legacy
            # bullet-only generation artifacts too.
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
            # after FACTS: accept standalone lines when the
            # bullet marker was omitted.
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
        # Deduplicate while preserving order
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
    # Convert parsed facts into Grounder input
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
    # Reusable grounding/synthesis failure
    # ========================================================

    def _grounding_failure(
        self,
        *,
        raw_answer,
        draft_direct_answer,
        draft_claims,
        supported_claims,
        unsupported_claims,
        relevant_claims,
        filtered_irrelevant_claims,
        direct_answer=None,
    ) -> GenerationResult:

        return GenerationResult(
            answer=
                GROUNDING_FAILURE_MESSAGE,

            raw_answer=
                raw_answer,

            direct_answer=
                direct_answer,

            draft_direct_answer=
                draft_direct_answer,

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
                draft_claims,

            supported_claims=
                supported_claims,

            unsupported_claims=
                unsupported_claims,

            relevant_claims=
                relevant_claims,

            filtered_irrelevant_claims=
                filtered_irrelevant_claims,
        )


    # ========================================================
    # Final answer rendering
    #
    # direct_answer is now the POST-SYNTHESIS answer.
    # ========================================================

    @staticmethod
    def _build_grounded_answer(
        direct_answer: str,
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
        # Do not load/run generator.
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
        # Original single-pass structured generation.
        #
        # Qwen still produces:
        #
        # DIRECT_ANSWER
        # +
        # FACTS
        #
        # DIRECT_ANSWER is now considered a DRAFT.
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
        # Parse draft answer separately from factual claims.
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
            "Draft direct answer:",
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
        # Structured generation contract failed.
        # ====================================================

        if not (
            parsed.direct_answer
        ):

            return self._grounding_failure(
                raw_answer=
                    raw_answer,

                draft_direct_answer=
                    None,

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

            return self._grounding_failure(
                raw_answer=
                    raw_answer,

                draft_direct_answer=
                    parsed.direct_answer,

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
        # Draft DIRECT_ANSWER still does not enter NLI.
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
        # No generated evidence fact is supported.
        # ====================================================

        if (
            grounded_claims.supported_count
            ==
            0
        ):

            return self._grounding_failure(
                raw_answer=
                    raw_answer,

                draft_direct_answer=
                    parsed.direct_answer,

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
        # Query relevance over supported FACTS.
        #
        # RelevanceFilter V2:
        #
        # catastrophic floor
        # +
        # top-k
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
        # Gate 5:
        # Supported claims exist, but none are query-relevant.
        # ====================================================

        if not (
            relevance_result
            .relevant_claims
        ):

            return self._grounding_failure(
                raw_answer=
                    raw_answer,

                draft_direct_answer=
                    parsed.direct_answer,

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
        # CONSTRAINED ANSWER SYNTHESIS
        #
        # This is the key new architecture.
        #
        #
        # yes/no
        # -------
        #
        # Ignore Qwen's free-form Yes/No draft.
        #
        # Score:
        #
        # P(Yes | question + VERIFIED facts)
        #
        # versus:
        #
        # P(No | question + VERIFIED facts)
        #
        #
        # entity
        # ------
        #
        # Validate Qwen's draft entity against VERIFIED facts.
        #
        #
        # other
        # -----
        #
        # Preserve Qwen's original direct answer.
        # ====================================================

        synthesis = (
            self.answer_synthesizer.synthesize(
                query=
                    query,

                draft_direct_answer=
                    parsed.direct_answer,

                relevant_claims=(
                    relevance_result
                    .relevant_claims
                ),

                context=
                    context,

                model=
                    self.model,

                tokenizer=
                    self.tokenizer,

                device=
                    self.device,
            )
        )


        print(
            "\n===== CONSTRAINED ANSWER SYNTHESIS ====="
        )


        print(
            "Mode:",
            synthesis.mode,
        )


        print(
            "Accepted:",
            synthesis.accepted,
        )


        print(
            "Draft answer:",
            parsed.direct_answer,
        )


        print(
            "Final answer:",
            synthesis.final_answer,
        )


        if (
            synthesis.yes_score
            is not None
        ):

            print(
                "Yes score:",
                round(
                    synthesis.yes_score,
                    4,
                ),
            )


        if (
            synthesis.no_score
            is not None
        ):

            print(
                "No score:",
                round(
                    synthesis.no_score,
                    4,
                ),
            )


        print(
            "Unique citations:",
            synthesis.unique_citation_count,
        )


        for reason in (
            synthesis.reasons
        ):

            print(
                "INFO:",
                reason,
            )


        # ====================================================
        # Gate 6:
        # We could not safely produce final answer.
        # ====================================================

        if not (
            synthesis.accepted
        ):

            return self._grounding_failure(
                raw_answer=
                    raw_answer,

                draft_direct_answer=
                    parsed.direct_answer,

                direct_answer=
                    None,

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


        final_direct_answer = (
            synthesis.final_answer
        )


        if not final_direct_answer:

            return self._grounding_failure(
                raw_answer=
                    raw_answer,

                draft_direct_answer=
                    parsed.direct_answer,

                direct_answer=
                    None,

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
        # Step 6:
        # Final answer + verified supporting facts.
        #
        # IMPORTANT:
        #
        # We use final_direct_answer.
        #
        # Not parsed.direct_answer.
        # ====================================================

        final_answer = (
            self._build_grounded_answer(
                direct_answer=
                    final_direct_answer,

                relevant_claims=(
                    relevance_result
                    .relevant_claims
                ),
            )
        )


        # ====================================================
        # Step 7:
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

                # --------------------------------------------
                # Final synthesized answer.
                # --------------------------------------------

                direct_answer=
                    final_direct_answer,

                # --------------------------------------------
                # Original Qwen draft.
                # --------------------------------------------

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

            # ------------------------------------------------
            # FINAL answer after constrained synthesis.
            # ------------------------------------------------

            direct_answer=
                final_direct_answer,

            # ------------------------------------------------
            # Original Qwen draft preserved for evaluation.
            # ------------------------------------------------

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