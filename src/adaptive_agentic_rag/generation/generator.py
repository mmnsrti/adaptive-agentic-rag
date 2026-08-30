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
    build_synthesis_messages,
)

from adaptive_agentic_rag.generation.claim_grounder import (
    ClaimGrounder,
    ClaimSupport,
    GroundedClaims,
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
# Draft structures
# ============================================================

@dataclass
class DraftFact:

    text: str

    citation_id: int | None


@dataclass
class ParsedDraft:

    direct_answer: str | None

    evidence_facts: list[DraftFact]


    # --------------------------------------------------------
    # Backward-compatible diagnostic/test property
    # --------------------------------------------------------

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
# Generation result
# ============================================================

@dataclass
class GenerationResult:

    answer: str

    raw_answer: str | None

    # Final corrected direct answer
    direct_answer: str | None

    # Initial untrusted answer
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
        max_relevant_claims: int = 3,
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


        self.tokenizer = None

        self.model = None

        self.claim_grounder = None


        # ====================================================
        # Maximum is now 3 because Pass 1 itself generates
        # at most 3 query-focused facts.
        #
        # This prevents top-2 from deleting one side of a
        # comparison merely because its reranker score is lower.
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
    # Model loading
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
    # Shared local generation
    # ========================================================

    def _generate_from_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
    ) -> str:

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
    # Pass 1
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


        return (
            self._generate_from_messages(
                messages=
                    messages,
                max_new_tokens=
                    max_new_tokens,
            )
        )


    # ========================================================
    # Cleaning
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
    # Pass-1 parser
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
            # New contract
            # =================================================

            direct_match = re.match(
                (
                    r"^(?:DRAFT_ANSWER|DIRECT_ANSWER)"
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


            if re.match(
                r"^FACTS\s*:",
                line,
                flags=re.IGNORECASE,
            ):

                inside_facts = True

                continue


            # =================================================
            # Preferred:
            #
            # - [2] Some factual claim.
            # =================================================

            linked_fact_match = re.match(
                r"^[-*•]\s*\[(\d+)\]\s*(.+)$",
                line,
            )


            if linked_fact_match:

                citation_id = int(
                    linked_fact_match.group(
                        1
                    )
                )


                fact_text = (
                    cls._clean_generated_text(
                        linked_fact_match.group(
                            2
                        )
                    )
                )


                if fact_text:

                    evidence_facts.append(
                        DraftFact(
                            text=
                                fact_text,
                            citation_id=
                                citation_id,
                        )
                    )


                continue


            # =================================================
            # Legacy fallback:
            #
            # keep the fact but mark it as unlinked.
            # Production will NOT trust an unlinked fact.
            # =================================================

            bullet_match = re.match(
                r"^[-*•]\s*(.+)$",
                line,
            )


            if bullet_match:

                fact_text = (
                    cls._clean_generated_text(
                        bullet_match.group(
                            1
                        )
                    )
                )


                if fact_text:

                    evidence_facts.append(
                        DraftFact(
                            text=
                                fact_text,
                            citation_id=
                                None,
                        )
                    )


                continue


            if inside_facts:

                fact_text = (
                    cls._clean_generated_text(
                        line
                    )
                )


                if fact_text:

                    evidence_facts.append(
                        DraftFact(
                            text=
                                fact_text,
                            citation_id=
                                None,
                        )
                    )


        # ====================================================
        # Deduplicate using citation + text
        # ====================================================

        unique = []

        seen = set()


        for fact in evidence_facts:

            key = (
                fact.citation_id,
                fact.text,
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
    # Build citation-locked context
    # ========================================================

    @staticmethod
    def _context_for_citation(
        context: BuiltContext,
        citation_id: int,
    ) -> BuiltContext | None:

        matching_items = [

            item

            for item
            in context.items

            if (
                item.citation_id
                ==
                citation_id
            )
        ]


        if not matching_items:

            return None


        item = (
            matching_items[
                0
            ]
        )


        return BuiltContext(
            text=
                item.text,
            items=[
                item
            ],
            total_words=
                len(
                    item.text.split()
                ),
        )


    # ========================================================
    # Ground citation-linked facts
    # ========================================================

    def _ground_linked_facts(
        self,
        parsed: ParsedDraft,
        context: BuiltContext,
    ) -> GroundedClaims:

        results = []


        for fact in (
            parsed.evidence_facts
        ):

            # =================================================
            # Unlinked facts are not trusted.
            # =================================================

            if (
                fact.citation_id
                is None
            ):

                results.append(
                    ClaimSupport(
                        claim=
                            fact.text,
                        supported=
                            False,
                        citation_id=
                            None,
                        label=
                            "missing_citation",
                        entailment_score=
                            0.0,
                        supporting_text=
                            None,
                        evidence_relevance_score=
                            None,
                        premise_mode=
                            None,
                    )
                )

                continue


            locked_context = (
                self._context_for_citation(
                    context=
                        context,
                    citation_id=
                        fact.citation_id,
                )
            )


            # =================================================
            # Invalid citation ID
            # =================================================

            if (
                locked_context
                is None
            ):

                results.append(
                    ClaimSupport(
                        claim=
                            fact.text,
                        supported=
                            False,
                        citation_id=
                            None,
                        label=
                            "invalid_citation",
                        entailment_score=
                            0.0,
                        supporting_text=
                            None,
                        evidence_relevance_score=
                            None,
                        premise_mode=
                            None,
                    )
                )

                continue


            grounded = (
                self.claim_grounder
                .ground(
                    answer=(
                        f"- {fact.text}"
                    ),
                    context=
                        locked_context,
                )
            )


            if not grounded.claims:

                results.append(
                    ClaimSupport(
                        claim=
                            fact.text,
                        supported=
                            False,
                        citation_id=
                            None,
                        label=
                            "no_claim",
                        entailment_score=
                            0.0,
                        supporting_text=
                            None,
                        evidence_relevance_score=
                            None,
                        premise_mode=
                            None,
                    )
                )

                continue


            # =================================================
            # Prompt requires one atomic claim.
            #
            # If the extractor still creates multiple claims,
            # preserve all of them; every one is checked only
            # against the cited evidence item.
            # =================================================

            results.extend(
                grounded.claims
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


    # ========================================================
    # Pass 2
    # Grounded self-correction
    # ========================================================

    def _generate_final_direct_answer(
        self,
        query: str,
        draft_answer: str | None,
        relevant_claims,
        max_new_tokens: int = 64,
    ) -> str | None:

        verified_facts = [

            (
                claim.citation_id,
                claim.claim,
            )

            for claim
            in relevant_claims

            if (
                claim.citation_id
                is not None
            )
        ]


        if not verified_facts:

            return None


        messages = (
            build_synthesis_messages(
                query=
                    query,
                draft_answer=
                    draft_answer,
                verified_facts=
                    verified_facts,
            )
        )


        output = (
            self._generate_from_messages(
                messages=
                    messages,
                max_new_tokens=
                    max_new_tokens,
            )
        )


        match = re.search(
            r"FINAL_ANSWER\s*:\s*(.+)",
            output,
            flags=re.IGNORECASE,
        )


        if match:

            answer = (
                self._clean_generated_text(
                    match.group(
                        1
                    )
                )
            )

        else:

            answer = (
                self._clean_generated_text(
                    output
                )
            )


        if not answer:

            return None


        if (
            answer.upper()
            ==
            INSUFFICIENT_TOKEN
        ):

            return None


        return answer


    # ========================================================
    # Final answer
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


        if direct_answer:

            line = (
                direct_answer.strip()
            )


            if citation_suffix:

                line += (
                    f" {citation_suffix}"
                )


            lines.append(
                line
            )


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
        # Gate 1
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


        self._load_generator()


        # ====================================================
        # Pass 1
        # ====================================================

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


        parsed = (
            self._parse_draft(
                raw_answer
            )
        )


        print(
            "\n===== PARSED DRAFT ====="
        )

        print(
            "Draft answer:",
            parsed.direct_answer,
        )


        for fact in (
            parsed.evidence_facts
        ):

            print(
                "FACT:",
                fact.citation_id,
                fact.text,
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
        # Citation-locked grounding
        # ====================================================

        self._load_claim_grounder()


        grounded_claims = (
            self._ground_linked_facts(
                parsed=
                    parsed,
                context=
                    context,
            )
        )


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
        # Relevance
        # ====================================================

        relevance_result = (
            self.relevance_filter.filter(
                query=
                    query,
                grounded_claims=
                    grounded_claims,
            )
        )


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
        # Pass 2:
        # grounded self-correction
        # ====================================================

        final_direct_answer = (
            self._generate_final_direct_answer(
                query=
                    query,
                draft_answer=
                    parsed.direct_answer,
                relevant_claims=(
                    relevance_result
                    .relevant_claims
                ),
                max_new_tokens=
                    64,
            )
        )


        if not final_direct_answer:

            return GenerationResult(
                answer=
                    GROUNDING_FAILURE_MESSAGE,
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


        print(
            "\n===== SELF-CORRECTED ANSWER ====="
        )

        print(
            "Draft:",
            parsed.direct_answer,
        )

        print(
            "Final:",
            final_direct_answer,
        )


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
                    final_direct_answer,
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


        return GenerationResult(
            answer=
                final_answer,
            raw_answer=
                raw_answer,
            direct_answer=
                final_direct_answer,
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