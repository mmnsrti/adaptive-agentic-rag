from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)


SYSTEM_PROMPT = """
You are a grounded question-answering assistant.

You must answer using only the evidence provided to you.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Every factual claim must be supported by the provided evidence.
4. Cite supporting evidence using citation markers such as [1], [2], or [3].
5. Never invent citation numbers.
6. If the evidence does not support a reliable answer, say:
   "I don't have enough evidence in the provided sources to answer reliably."
7. Keep the answer concise and directly relevant to the question.
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


INSTRUCTIONS:

Answer the original question using only the evidence above.

Use citation markers like [1] and [2] immediately after the claims they support.

Do not refer to evidence that is not provided.
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