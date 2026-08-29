from typing import Any

from adaptive_agentic_rag.agents.query_router import (
    QueryRouter
)

from adaptive_agentic_rag.agents.evidence_grader import (
    EvidenceGrader
)

from adaptive_agentic_rag.agents.query_rewriter import (
    QueryRewriter
)

from adaptive_agentic_rag.agents.answer_grader import (
    AnswerGrader
)

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder
)

from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator
)

from adaptive_agentic_rag.orchestration.state import (
    AgentState
)


class RAGNodes:
    """
    LangGraph node implementations.

    Shared models/services live here.

    Per-request information lives in AgentState.
    """

    def __init__(self):

        # ==================================================
        # Core services
        # ==================================================

        self.router = (
            QueryRouter()
        )

        self.retriever = (
            AdaptiveRetriever()
        )

        self.context_builder = (
            ContextBuilder()
        )

        self.evidence_grader = (
            EvidenceGrader()
        )

        self.query_rewriter = (
            QueryRewriter()
        )


        # ==================================================
        # IMPORTANT:
        # Reuse existing models.
        #
        # Do NOT load another reranker.
        # Do NOT load another embedding model.
        # ==================================================

        self.generator = (
            GroundedGenerator(

                reranker=(
                    self.retriever
                    .reranked
                    .reranker
                )
            )
        )


        self.answer_grader = (
            AnswerGrader(

                embedder=(
                    self.retriever
                    .dense
                    .embedder
                )
            )
        )


    # ======================================================
    # Helper
    # ======================================================

    @staticmethod
    def _safe_list(
        value: Any
    ) -> list[Any]:

        if value is None:
            return []

        if isinstance(
            value,
            list
        ):
            return value

        return list(value)


    # ======================================================
    # Helper
    # Extract citation validity from GenerationResult
    # ======================================================

    @staticmethod
    def _extract_citation_valid(
        generation_result
    ) -> bool | None:

        #
        # First check whether GenerationResult
        # exposes citation_valid directly.
        #

        direct_value = getattr(
            generation_result,
            "citation_valid",
            None
        )

        if direct_value is not None:
            return bool(
                direct_value
            )


        #
        # Otherwise inspect common nested
        # citation result fields.
        #

        nested_names = [

            "citation_validation",

            "citation_result",

            "citation_report"
        ]


        validity_names = [

            "valid",

            "is_valid",

            "citation_valid"
        ]


        for nested_name in nested_names:

            nested = getattr(
                generation_result,
                nested_name,
                None
            )

            if nested is None:
                continue


            for validity_name in (
                validity_names
            ):

                value = getattr(
                    nested,
                    validity_name,
                    None
                )

                if value is not None:

                    return bool(value)


        return None


    # ======================================================
    # Helper
    # Convert GenerationResult to State update
    # ======================================================

    def _generation_state_update(
        self,
        result
    ) -> dict:

        final_answer = getattr(
            result,
            "answer",
            None
        )


        if final_answer is None:

            final_answer = getattr(
                result,
                "final_answer",
                None
            )


        return {

            "generation_result":
                result,

            "generation_model_name":
                getattr(
                    result,
                    "model_name",
                    None
                ),

            "raw_answer":
                getattr(
                    result,
                    "raw_answer",
                    None
                ),

            "final_answer":
                final_answer,

            "abstained":
                bool(
                    getattr(
                        result,
                        "abstained",
                        False
                    )
                ),

            "supported_claims":
                int(
                    getattr(
                        result,
                        "supported_claims",
                        0
                    )
                ),

            "unsupported_claims":
                int(
                    getattr(
                        result,
                        "unsupported_claims",
                        0
                    )
                ),

            "relevant_claims":
                int(
                    getattr(
                        result,
                        "relevant_claims",
                        0
                    )
                ),

            "filtered_irrelevant_claims":
                int(
                    getattr(
                        result,
                        "filtered_irrelevant_claims",
                        0
                    )
                ),
            "citation_valid":
                self._extract_citation_valid(
                    result
                )
        }


    # ======================================================
    # Node 1
    # Query routing
    # ======================================================

    def route_query(
        self,
        state: AgentState
    ) -> dict:

        query = (
            state[
                "current_query"
            ]
        )


        decision = (
            self.router.route(
                query
            )
        )


        return {

            "query_type":
                decision[
                    "query_type"
                ],

            "retrieval_strategy":
                decision[
                    "retrieval_strategy"
                ],

            "use_reranker":
                decision[
                    "rerank"
                ],

            "use_mmr":
                decision[
                    "mmr"
                ]
        }


    # ======================================================
    # Node 2
    # Retrieval
    # ======================================================

    def retrieve(
        self,
        state: AgentState
    ) -> dict:

        query = (
            state[
                "current_query"
            ]
        )


        retrieval_output = (
            self.retriever.search(
                query,
                top_k=10
            )
        )


        results = (
            retrieval_output[
                "results"
            ]
        )


        return {

            "retrieved_results":
                results
        }


    # ======================================================
    # Node 3
    # Context construction
    # ======================================================

    def build_context(
        self,
        state: AgentState
    ) -> dict:

        results = (
            state[
                "retrieved_results"
            ]
        )


        context = (
            self.context_builder.build(

                results,

                query=(
                    state.get(
                        "original_query"
                    )
                    or
                    state.get(
                        "current_query",
                        ""
                    )
                )
            )
        )


        return {

            "context":
                context
        }


    # ======================================================
    # Node 4
    # Evidence grading
    # ======================================================

    def grade_evidence(
        self,
        state: AgentState
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )


        context = (
            state[
                "context"
            ]
        )


        query_type = (
            state[
                "query_type"
            ]
        )


        if context is None:

            raise ValueError(
                "Cannot grade evidence "
                "without a built context."
            )


        if query_type is None:

            raise ValueError(
                "Cannot grade evidence "
                "without query_type."
            )


        grade = (
            self.evidence_grader.grade(

                query=original_query,

                context=context,

                query_type=query_type
            )
        )


        return {

            "evidence_sufficient":
                grade.sufficient,

            "evidence_score":
                grade.evidence_score,

            "evidence_reasons":
                grade.reasons
        }


    # ======================================================
    # Node 5
    # Query rewriting
    # ======================================================

    def rewrite_query(
        self,
        state: AgentState
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )


        query_type = (
            state[
                "query_type"
            ]
        )


        if query_type is None:

            raise ValueError(
                "Cannot rewrite query "
                "without query_type."
            )


        attempt = (
            state[
                "retry_count"
            ]
            + 1
        )


        rewritten_query = (
            self.query_rewriter.rewrite(

                query=original_query,

                query_type=query_type,

                attempt=attempt
            )
        )


        return {

            "current_query":
                rewritten_query,

            "retry_count":
                attempt,

            "rewritten":
                True,


            #
            # Reset state belonging
            # to the previous retrieval round.
            #

            "retrieved_results":
                [],

            "context":
                None,

            "evidence_sufficient":
                None,

            "evidence_score":
                None,

            "evidence_reasons":
                []
        }


    # ======================================================
    # Node 6
    # Grounded generation
    # ======================================================

    def generate(
        self,
        state: AgentState
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )


        context = (
            state[
                "context"
            ]
        )


        evidence_sufficient = (
            state[
                "evidence_sufficient"
            ]
        )


        if context is None:

            raise ValueError(
                "Cannot generate answer "
                "without context."
            )


        if evidence_sufficient is not True:

            raise ValueError(
                "Generate node should only "
                "run when evidence is sufficient."
            )


        result = (
            self.generator.generate(

                query=original_query,

                context=context,

                evidence_sufficient=True
            )
        )


        return (
            self._generation_state_update(
                result
            )
        )


    # ======================================================
    # Node 7
    # Explicit abstention
    # ======================================================

    def abstain(
        self,
        state: AgentState
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )


        context = (
            state[
                "context"
            ]
        )


        if context is None:

            raise ValueError(
                "Cannot create abstention "
                "without context."
            )


        #
        # Reuse GroundedGenerator's existing
        # evidence safety gate.
        #
        # Because evidence_sufficient=False,
        # Qwen should NOT be executed here.
        #

        result = (
            self.generator.generate(

                query=original_query,

                context=context,

                evidence_sufficient=False
            )
        )


        return (
            self._generation_state_update(
                result
            )
        )


    # ======================================================
    # Node 8
    # Final answer grading
    # ======================================================

    def grade_answer(
        self,
        state: AgentState
    ) -> dict:

        generation_result = (
            state[
                "generation_result"
            ]
        )


        if generation_result is None:

            raise ValueError(
                "Cannot grade answer "
                "without GenerationResult."
            )


        original_query = (
            state[
                "original_query"
            ]
        )


        evidence_sufficient = (
            state[
                "evidence_sufficient"
            ]
        )


        if evidence_sufficient is None:

            raise ValueError(
                "Cannot grade answer "
                "before evidence grading."
            )


        grade = (
            self.answer_grader.grade(

                query=original_query,

                generation_result=(
                    generation_result
                ),

                evidence_sufficient=(
                    evidence_sufficient
                )
            )
        )


        return {

            "answer_grade":
                grade,

            "answer_passed":
                getattr(
                    grade,
                    "passed",
                    None
                ),

            "answer_relevance_score":
                getattr(
                    grade,
                    "relevance_score",
                    None
                ),

            "answer_grade_reasons":
                self._safe_list(
                    getattr(
                        grade,
                        "reasons",
                        []
                    )
                )
        }


    # ======================================================
    # Cleanup
    # ======================================================

    def close(self):

        close_method = getattr(
            self.retriever,
            "close",
            None
        )


        if callable(
            close_method
        ):

            close_method()