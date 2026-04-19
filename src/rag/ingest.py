from pathlib import Path

from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.config import (
    KNOWLEDGE_BASE,
    VECTOR_DB,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

load_dotenv(override=True)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def normalize_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def fetch_documents() -> list[dict]:
    """
    Load all markdown documents from the knowledge base.
    Each top-level folder name becomes the document type.
    """
    documents = []

    if not KNOWLEDGE_BASE.exists():
        raise FileNotFoundError(f"Knowledge base path not found: {KNOWLEDGE_BASE}")

    for folder in KNOWLEDGE_BASE.iterdir():
        if not folder.is_dir():
            continue

        doc_type = folder.name

        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read().strip()

            documents.append(
                {
                    "doc_type": doc_type,
                    "source": file.as_posix(),
                    "source_name": normalize_name(file),
                    "text": text,
                }
            )

    print(f"Loaded {len(documents)} documents")
    return documents


def create_chunks(documents: list[dict]) -> list[dict]:
    """
    Split documents into overlapping chunks using a local text splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
    )

    chunks = []

    for doc in documents:
        enriched_text = (
            f"Source Name: {doc['source_name']}\n"
            f"Document Type: {doc['doc_type']}\n"
            f"Source Path: {doc['source']}\n\n"
            f"{doc['text']}"
        )

        split_texts = splitter.split_text(enriched_text)

        for i, chunk_text in enumerate(split_texts):
            chunks.append(
                {
                    "page_content": chunk_text,
                    "metadata": {
                        "source": doc["source"],
                        "doc_type": doc["doc_type"],
                        "source_name": doc["source_name"],
                        "chunk_id": i,
                    },
                }
            )

    print(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings(chunks: list[dict]) -> None:
    """
    Create embeddings locally and store them in Chroma.
    """
    VECTOR_DB.mkdir(parents=True, exist_ok=True)

    chroma = PersistentClient(path=str(VECTOR_DB))

    existing = [c.name for c in chroma.list_collections()]
    if COLLECTION_NAME in existing:
        chroma.delete_collection(COLLECTION_NAME)

    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    texts = [chunk["page_content"] for chunk in chunks]
    metas = [chunk["metadata"] for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]

    vectors = embedding_model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metas,
    )

    print(f"Vectorstore created with {collection.count()} chunks")


def main():
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")


if __name__ == "__main__":
    main()