from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state for one Adaptive Agentic RAG execution.

    Important:
    This contains per-request data only.

    Models, retrievers, rerankers, graders,
    and other shared services do NOT belong here.
    """

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    original_query: str

    current_query: str


    # --------------------------------------------------
    # Query routing
    # --------------------------------------------------

    query_type: str | None

    retrieval_strategy: str | None

    use_reranker: bool

    use_mmr: bool


    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    retrieved_results: list[dict[str, Any]]


    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    context: Any | None


    # --------------------------------------------------
    # Evidence grading
    # --------------------------------------------------

    evidence_sufficient: bool | None

    evidence_score: float | None

    evidence_reasons: list[str]


    # --------------------------------------------------
    # Self-correction
    # --------------------------------------------------

    retry_count: int

    max_retries: int

    rewritten: bool


    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    raw_answer: str | None

    final_answer: str | None

    abstained: bool


    # --------------------------------------------------
    # Claim verification
    # --------------------------------------------------

    supported_claims: list[Any]

    unsupported_claims: list[Any]

    relevant_claims: list[Any]

    filtered_irrelevant_claims: list[Any]


    # --------------------------------------------------
    # Citation validation
    # --------------------------------------------------

    citation_valid: bool | None


    # --------------------------------------------------
    # Final answer grading
    # --------------------------------------------------

    answer_passed: bool | None

    answer_relevance_score: float | None


    # --------------------------------------------------
    # Error handling / observability
    # --------------------------------------------------

    error: str | None
def create_initial_state(
    query: str,
    max_retries: int = 1
) -> AgentState:

    return AgentState(

        original_query=query,

        current_query=query,

        query_type=None,

        retrieval_strategy=None,

        use_reranker=False,

        use_mmr=False,

        retrieved_results=[],

        context=None,

        evidence_sufficient=None,

        evidence_score=None,

        evidence_reasons=[],

        retry_count=0,

        max_retries=max_retries,

        rewritten=False,

        raw_answer=None,

        final_answer=None,

        abstained=False,

        supported_claims=[],

        unsupported_claims=[],

        relevant_claims=[],

        filtered_irrelevant_claims=[],

        citation_valid=None,

        answer_passed=None,

        answer_relevance_score=None,

        error=None
    )    