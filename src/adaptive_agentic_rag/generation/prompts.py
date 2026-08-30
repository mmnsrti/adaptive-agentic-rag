from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
)


# ============================================================
# Pass 1
#
# This is intentionally kept simple.
#
# Previous diagnostics showed that Qwen 1.5B follows this
# contract substantially better than citation-selection or
# source-selection contracts.
# ============================================================

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

5. For yes/no questions, begin the direct answer with:
   Yes
   or
   No

6. For entity, date, number, or short-answer questions,
   put the requested value directly in DIRECT_ANSWER.

7. Produce between 1 and 3 evidence facts.

8. Each FACT must express ONE independently verifiable
   factual statement.

9. Do not combine facts from two different sources
   in one FACT.

10. Do not put reasoning or conclusions inside FACTS.

11. Do not use:
    while
    whereas
    indicating
    implying
    therefore
    consequently

    inside FACTS.

12. Prefer stating factual content directly.

BAD:
The second TechCrunch article indicates that Google
has no additional measures planned for YouTube.

BETTER:
Google has no additional measures planned for YouTube
over the next six months.

13. Mention a source name inside a FACT only when the
    source identity is necessary to distinguish evidence
    required by the question.

14. Keep every FACT short and complete.

15. Never end with an incomplete sentence.

16. Do not add citations.

17. Do not add citation IDs.

18. If the provided evidence genuinely cannot support
    an answer, use:

DIRECT_ANSWER: INSUFFICIENT_EVIDENCE

Return EXACTLY:

DIRECT_ANSWER: <short direct answer>

FACTS:
- <one atomic factual claim>
- <one atomic factual claim>
""".strip()


# ============================================================
# Pass 2
#
# The initial DIRECT_ANSWER is intentionally NOT provided.
#
# This prevents anchoring the second pass to the first answer.
#
# Pass 2 reasons only over facts that have already survived:
#
# Grounding
# +
# relevance filtering
# ============================================================

SYNTHESIS_SYSTEM_PROMPT = """
You are the final reasoning stage of an evidence-grounded
question-answering system.

You receive:

- the original question
- factual claims that have already been independently verified
- source metadata for those verified facts

Answer the original question FROM SCRATCH.

Rules:

1. Use ONLY the VERIFIED FACTS.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not assume any previous answer was correct.

5. Determine whether the VERIFIED FACTS are sufficient
   to answer the complete question.

6. For a yes/no question, return exactly one of:

FINAL_ANSWER: Yes

FINAL_ANSWER: No

7. For a non-boolean question, return one short direct answer.

8. If the VERIFIED FACTS are insufficient to answer the
   complete question, return exactly:

FINAL_ANSWER: INSUFFICIENT_EVIDENCE

9. Do not add citations.
10. Do not explain your reasoning.
11. Do not introduce new facts.
""".strip()


# ============================================================
# Pass 1 messages
# ============================================================

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
- Use between 1 and 3 FACTS.
- Each FACT must be independently verifiable.
- Do not combine two sources inside one FACT.
- Do not add citation IDs or citation markers.

For yes/no questions, DIRECT_ANSWER must begin with
Yes or No.

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


# ============================================================
# Pass 2 messages
# ============================================================

def build_synthesis_messages(
    query: str,
    verified_facts: list[dict],
) -> list[dict]:

    rendered_facts = []


    for fact in verified_facts:

        citation_id = (
            fact[
                "citation_id"
            ]
        )

        source = (
            fact.get(
                "source",
                ""
            )
            or ""
        )

        title = (
            fact.get(
                "title",
                ""
            )
            or ""
        )

        text = (
            fact[
                "text"
            ]
        )


        metadata_parts = []


        if source:

            metadata_parts.append(
                f"Source: {source}"
            )


        if title:

            metadata_parts.append(
                f"Title: {title}"
            )


        metadata = (
            " | ".join(
                metadata_parts
            )
        )


        if metadata:

            rendered_facts.append(
                (
                    f"[{citation_id}] "
                    f"{metadata}\n"
                    f"Fact: {text}"
                )
            )

        else:

            rendered_facts.append(
                (
                    f"[{citation_id}] "
                    f"Fact: {text}"
                )
            )


    facts_text = (
        "\n\n".join(
            rendered_facts
        )
    )


    user_prompt = f"""
QUESTION:

{query}


VERIFIED FACTS:

{facts_text}


TASK:

Re-answer the QUESTION from scratch.

Use ONLY VERIFIED FACTS.

First determine whether the verified facts cover all parts
required by the question.

For a yes/no question, return exactly:

FINAL_ANSWER: Yes

or:

FINAL_ANSWER: No

If the verified facts do not cover enough of the question:

FINAL_ANSWER: INSUFFICIENT_EVIDENCE
""".strip()


    return [
        {
            "role": "system",
            "content": SYNTHESIS_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]