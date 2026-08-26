from dataclasses import dataclass


@dataclass
class ContextItem:
    citation_id: int
    chunk_id: str
    document_id: str
    title: str
    source: str
    url: str | None
    text: str
    score: float


@dataclass
class BuiltContext:
    text: str
    items: list[ContextItem]
    total_words: int


class ContextBuilder:

    def __init__(
        self,
        max_words: int = 1800,
        max_chunks: int = 8,
        max_chunks_per_document: int = 2
    ):

        self.max_words = max_words
        self.max_chunks = max_chunks
        self.max_chunks_per_document = (
            max_chunks_per_document
        )


    def build(
        self,
        results: list[dict]
    ) -> BuiltContext:

        if not results:

            return BuiltContext(
                text="",
                items=[],
                total_words=0
            )


        selected_items = []

        document_counts = {}

        total_words = 0


        for result in results:

            if len(selected_items) >= self.max_chunks:
                break


            document_id = result[
                "document_id"
            ]


            current_document_count = (
                document_counts.get(
                    document_id,
                    0
                )
            )


            if (
                current_document_count
                >= self.max_chunks_per_document
            ):
                continue


            text = (
                result.get(
                    "text",
                    ""
                )
                .strip()
            )


            if not text:
                continue


            words = text.split()

            remaining_words = (
                self.max_words
                - total_words
            )


            if remaining_words <= 0:
                break


            if len(words) > remaining_words:

                text = " ".join(
                    words[:remaining_words]
                )

                words = text.split()


            metadata = result.get(
                "metadata",
                {}
            )


            item = ContextItem(

                citation_id=(
                    len(selected_items)
                    + 1
                ),

                chunk_id=result[
                    "id"
                ],

                document_id=document_id,

                title=metadata.get(
                    "title",
                    ""
                ),

                source=metadata.get(
                    "source",
                    ""
                ),

                url=metadata.get(
                    "url"
                ),

                text=text,

                score=float(
                    result.get(
                        "score",
                        0.0
                    )
                )
            )


            selected_items.append(
                item
            )


            document_counts[
                document_id
            ] = (
                current_document_count
                + 1
            )


            total_words += len(
                words
            )


        context_parts = []


        for item in selected_items:

            header = (
                f"[{item.citation_id}] "
                f"{item.title}"
            )


            if item.source:

                header += (
                    f" — {item.source}"
                )


            context_parts.append(

                f"{header}\n"
                f"{item.text}"

            )


        context_text = (
            "\n\n".join(
                context_parts
            )
        )


        return BuiltContext(
            text=context_text,
            items=selected_items,
            total_words=total_words
        )