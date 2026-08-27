from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)


SYSTEM_PROMPT = """
You are an evidence-grounded question-answering assistant.

Use only the provided evidence.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Answer the user's question directly.
4. Write factual statements as separate bullet points.
5. Keep each bullet focused on one main claim.
6. Prefer short, precise claims.
7. If the evidence does not support useful claims, say that the evidence is insufficient.

Do NOT add citations.
The system will verify and attach citations separately.
""".strip()


def build_grounded_messages(
    query: str,
    context: BuiltContext
) -> list[dict]:

    user_prompt = f"""
QUESTION:

{query}


EVIDENCE:

{context.text}


TASK:

Answer the question using only the evidence above.

Write between 2 and 6 concise bullet points.

Each bullet should contain one main factual claim.

Do not add citation markers.
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