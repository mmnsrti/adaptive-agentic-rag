from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
)


# ============================================================
# Pass 1:
# Evidence fact proposal
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-grounded question-answering assistant.

Use ONLY the provided evidence.

Your task has two parts:

1. Propose a short draft answer.
2. Select a small number of atomic factual claims from the evidence.

IMPORTANT:

The draft answer is NOT trusted yet.
Another stage will verify the facts and may revise the answer.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Keep DRAFT_ANSWER short.
4. For yes/no questions, DRAFT_ANSWER must begin with Yes or No.
5. Produce between 1 and 3 FACTS.
6. Every FACT must contain exactly one independently verifiable fact.
7. Every FACT must begin with exactly one evidence citation ID.
8. The citation ID must refer to the evidence item that directly supports the fact.
9. The factual text should contain the fact itself, NOT attribution language.

BAD:
- [2] The TechCrunch article says that Google has no plans...

GOOD:
- [2] Google has no additional measures planned for YouTube over the next six months.

10. Do not use phrases such as:
    "the article says"
    "the report states"
    "the first article"
    "the second article"
    "while"
    "whereas"
    "indicating"
    "implying"

11. For comparison, consistency, contradiction, or multi-source questions:
    include evidence from EACH side needed for the comparison.

12. When two different sources are being compared, prefer separate FACTS
    with separate citation IDs.

13. Never combine two evidence sources inside one FACT.

14. Do not create a FACT containing a conclusion such as:
    "therefore they differ"
    "this indicates a contradiction"
    "the reports are consistent"

Conclusions belong only in DRAFT_ANSWER.

15. Keep every FACT short and complete.
16. Never produce an incomplete sentence.

Return EXACTLY:

DRAFT_ANSWER: <short answer>

FACTS:
- [<citation id>] <atomic factual claim>
- [<citation id>] <atomic factual claim>
""".strip()


# ============================================================
# Pass 2:
# Verified-fact synthesis
# ============================================================

SYNTHESIS_SYSTEM_PROMPT = """
You are the final reasoning stage of an evidence-grounded QA system.

You will receive:

- the original question
- an untrusted draft answer
- factual claims that have already been independently verified

The draft answer may be wrong.

Your job is to reconsider the question using ONLY the VERIFIED FACTS.

Rules:

1. Ignore any unsupported information from the draft answer.
2. Use only the verified facts.
3. Answer the original question directly.
4. For yes/no questions, begin with Yes or No.
5. Keep the final answer to one concise sentence.
6. Do not add citations.
7. Do not introduce new facts.
8. If the verified facts are insufficient to answer reliably, return:

FINAL_ANSWER: INSUFFICIENT_EVIDENCE

Otherwise return exactly:

FINAL_ANSWER: <one concise direct answer>
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

Create a short draft answer and 1 to 3 atomic evidence facts.

Each FACT must begin with the citation ID of the exact evidence item
that supports it.

For comparison or consistency questions, make sure the selected FACTS
cover the different sides needed to answer the question.

Use exactly:

DRAFT_ANSWER: ...

FACTS:
- [1] ...
- [2] ...
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


def build_synthesis_messages(
    query: str,
    draft_answer: str | None,
    verified_facts: list[tuple[int, str]],
) -> list[dict]:

    facts_text = "\n".join(
        (
            f"- [{citation_id}] {fact}"
        )
        for citation_id, fact
        in verified_facts
    )


    user_prompt = f"""
QUESTION:

{query}


UNTRUSTED DRAFT ANSWER:

{draft_answer or "None"}


VERIFIED FACTS:

{facts_text}


TASK:

Reconsider the original question.

The draft answer may be wrong.

Use ONLY the VERIFIED FACTS to produce the final direct answer.

Return exactly:

FINAL_ANSWER: ...
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