from adaptive_agentic_rag.schemas import Document
from adaptive_agentic_rag.processing.chunker import chunk_document



def test_chunk_creation():

    doc = Document(
        id="doc_001",

        text="This is a test document. " * 500,

        metadata={
            "source": "test"
        }
    )


    chunks = chunk_document(doc)


    assert len(chunks) > 1

    assert chunks[0].document_id == "doc_001"

    assert "source" in chunks[0].metadata