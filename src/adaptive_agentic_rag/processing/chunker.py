from langchain_text_splitters import RecursiveCharacterTextSplitter

from adaptive_agentic_rag.schemas import Document, Chunk



def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Chunk]:


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


    texts = splitter.split_text(
        document.text
    )


    chunks = []


    for idx, text in enumerate(texts):

        chunk = Chunk(
            id=f"{document.id}_chunk_{idx}",

            document_id=document.id,

            text=text,

            metadata={
                **document.metadata,
                "chunk_index": idx
            }
        )


        chunks.append(chunk)


    return chunks