from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
)


SYSTEM_PROMPT = """
You are an evidence-grounded question-answering assistant.

Use only the provided evidence.

Your job has TWO separate parts:

1. Give a direct answer to the user's question.
2. Extract a small number of factual evidence claims that justify that answer.

The evidence claims will be verified independently by another model,
so each fact must be independently checkable.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Answer the user's question directly.
4. Keep the direct answer very short.
5. For yes/no questions, begin the direct answer with "Yes" or "No".
6. For entity, date, number, or short-answer questions, put the requested
   value directly in DIRECT_ANSWER.
7. Produce between 1 and 3 evidence facts.
8. Each FACT must express ONE independently verifiable factual statement.
9. Do not combine facts from two different sources in one FACT.
10. Do not put reasoning or conclusions inside FACTS.
11. Do not use "while", "whereas", "indicating", "implying",
    or similar comparison/inference wording inside FACTS.
12. Prefer stating the factual content directly instead of saying
    "the article reported that".
13. Mention a source name inside a FACT only when source identity is
    necessary to distinguish evidence in the question.
14. Keep every FACT short and complete.
15. Never end with an incomplete sentence.
16. Do not add citations. The system attaches citations later.
17. If the provided evidence genuinely cannot support an answer,
    use DIRECT_ANSWER: INSUFFICIENT_EVIDENCE

Return EXACTLY this structure:

DIRECT_ANSWER: <short direct answer>

FACTS:
- <one atomic factual claim>
- <one atomic factual claim>
""".strip()


def build_grounded_messages(
    query: str,
    context: BuiltContext,
) -> list[dict]:

    user_prompt = f"""
QUESTION:

{query}


EVIDENCE:

{context.text}


TASK:

Answer the QUESTION using only the EVIDENCE.

Remember:

- DIRECT_ANSWER must directly answer the question.
- FACTS are evidence facts, not conclusions.
- Use at most 3 FACTS.
- Each FACT must be independently verifiable from one evidence source.
- Do not combine two sources into the same FACT.
- Do not add citation markers.

Use exactly:

DIRECT_ANSWER: ...

FACTS:
- ...
- ...
""".strip()


    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]