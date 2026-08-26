from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)


SYSTEM_PROMPT = """
You are a strict evidence-grounded question-answering system.

You must answer ONLY from the provided evidence.

Rules:

1. Use only facts explicitly supported by the evidence.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, products, prices, dates, or policies.
4. Every bullet point MUST end with at least one citation.
5. Citations must use only the allowed citation IDs.
6. Never invent citation IDs.
7. Do not write uncited introductions or conclusions.
8. If evidence does not support a useful answer, say that the evidence is insufficient.
9. Prefer a short supported answer over a long speculative answer.

Required format:

- Supported factual statement [1]
- Supported factual statement [2][3]

Every bullet MUST contain a citation.
""".strip()


REPAIR_SYSTEM_PROMPT = """
You repair answers produced by an evidence-grounded QA system.

The previous answer failed citation validation.

Rewrite the answer so that:

1. Every factual bullet ends with at least one valid citation.
2. Only the supplied evidence is used.
3. Unsupported claims are removed.
4. Only allowed citation IDs are used.
5. No uncited introduction or conclusion is included.
6. The answer is concise.

Required format:

- Supported claim [1]
- Supported claim [2][3]

If the evidence cannot support the answer, respond only with:

I don't have enough evidence in the provided sources to answer reliably.
""".strip()


def _allowed_citations(
    context: BuiltContext
) -> str:

    return ", ".join(
        f"[{item.citation_id}]"
        for item in context.items
    )


def build_grounded_messages(
    query: str,
    context: BuiltContext
) -> list[dict]:

    allowed = _allowed_citations(
        context
    )


    user_prompt = f"""
QUESTION:

{query}


ALLOWED CITATIONS:

{allowed}


EVIDENCE:

{context.text}


TASK:

Answer the question using ONLY the evidence above.

Write between 2 and 4 concise bullet points.

Every bullet MUST end with one or more citations from:

{allowed}

Do not write any factual sentence without a citation.
Do not invent citation IDs.
""".strip()


    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]


def build_citation_repair_messages(
    query: str,
    context: BuiltContext,
    draft_answer: str
) -> list[dict]:

    allowed = _allowed_citations(
        context
    )


    user_prompt = f"""
ORIGINAL QUESTION:

{query}


ALLOWED CITATIONS:

{allowed}


EVIDENCE:

{context.text}


INVALID DRAFT:

{draft_answer}


TASK:

Rewrite the invalid draft.

Remove every unsupported claim.

Every remaining factual bullet MUST end with one or more allowed citations.

Use ONLY these citation IDs:

{allowed}
""".strip()


    return [
        {
            "role": "system",
            "content": REPAIR_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]