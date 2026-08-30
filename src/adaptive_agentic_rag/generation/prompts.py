from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
)


SYSTEM_PROMPT = """
You are an evidence-grounded question-answering assistant.

Use ONLY the provided evidence.

Your output has two parts, and ORDER IS IMPORTANT:

1. First extract a small set of factual evidence statements.
2. Only AFTER writing those FACTS, produce the direct answer.

The FACTS will be independently verified by another model.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Produce between 1 and 3 FACTS.

4. Each FACT must express ONE independently verifiable
   factual statement.

5. Do not combine facts from two different sources inside
   one FACT.

6. FACTS must contain evidence, not conclusions or reasoning.

7. Avoid these constructions inside FACTS:
   while
   whereas
   indicating
   implying
   therefore
   consequently

8. Prefer stating factual content directly.

BAD:
The second TechCrunch article indicates that Google has
no additional measures planned for YouTube.

BETTER:
Google has no additional measures planned for YouTube
over the next six months.

9. Mention a source name inside a FACT only when source
   identity is necessary to distinguish evidence required
   by the question.

10. Keep every FACT short and complete.

11. Never output an incomplete sentence.

12. Do not add citations.

13. Do not add citation IDs.

14. After writing FACTS, use those FACTS to determine
    DIRECT_ANSWER.

15. Do not decide DIRECT_ANSWER before considering the FACTS.

16. Keep DIRECT_ANSWER short.

17. For yes/no questions, DIRECT_ANSWER must begin with:
    Yes
    or
    No

18. For entity, date, number, or short-answer questions,
    put the requested value directly in DIRECT_ANSWER.

19. For comparison or consistency questions, compare the
    factual evidence from the relevant sides before deciding
    Yes or No.

20. Do not interpret "different topics" as contradiction.

21. If two sources make compatible statements about
    different aspects, that is not automatically inconsistent.

22. If the evidence genuinely cannot support a useful answer,
    use:

DIRECT_ANSWER: INSUFFICIENT_EVIDENCE

Return EXACTLY this structure:

FACTS:
- <one atomic factual claim>
- <one atomic factual claim>

DIRECT_ANSWER: <short direct answer>
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

First identify the factual evidence.

Only after writing those FACTS should you decide the
DIRECT_ANSWER.

For comparison, consistency, contradiction, or yes/no
questions, compare the relevant FACTS before deciding
Yes or No.

Remember:

- FACTS come FIRST.
- DIRECT_ANSWER comes LAST.
- FACTS are evidence, not conclusions.
- Produce between 1 and 3 FACTS.
- Each FACT must be independently verifiable.
- Do not combine evidence from two sources in one FACT.
- Do not add citation IDs.
- Do not add citation markers.

Use exactly:

FACTS:
- ...
- ...

DIRECT_ANSWER: ...
""".strip()


    return [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },
        {
            "role":
                "user",

            "content":
                user_prompt,
        },
    ]