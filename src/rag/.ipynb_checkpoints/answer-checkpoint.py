import json
import re

from dotenv import load_dotenv
from chromadb import PersistentClient
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.rag.config import (
    VECTOR_DB,
    COLLECTION_NAME,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    FINAL_K,
)

load_dotenv(override=True)

# Local embeddings
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Local Chroma DB
chroma = PersistentClient(path=str(VECTOR_DB))
collection = chroma.get_or_create_collection(COLLECTION_NAME)

# Local Ollama via OpenAI-compatible API
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.

Rules:
- Answer only using the provided context.
- Do not use outside knowledge.
- Do not guess.
- Be accurate, relevant, and complete.
- If the answer is not in the context, say exactly:
I could not find that in the knowledge base.

Context:
{context}
""".strip()


def format_history(history: list[dict] | None = None) -> str:
    history = history or []
    lines = []

    for item in history:
        role = item.get("role", "unknown")
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "[]"


def llm_text(messages: list[dict]) -> str:
    """
    Call local Ollama through its OpenAI-compatible endpoint.
    """
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    history_text = format_history(history)

    prompt = f"""
You are helping search a company knowledge base for Insurellm.

Conversation history:
{history_text}

User question:
{question}

Rewrite the question into one short search query most likely to retrieve the right chunks.

Rules:
- Keep important names, job titles, and entities.
- Keep it short.
- Output only the rewritten search query.
""".strip()

    try:
        rewritten = llm_text([{"role": "system", "content": prompt}])
        return rewritten if rewritten else question
    except Exception as e:
        print(f"[rewrite_query fallback] {type(e).__name__}: {e}")
        return question


def fetch_context_unranked(query_text: str) -> list[dict]:
    query_embedding = embedding_model.encode([query_text]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVAL_K,
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(docs, metas):
        chunks.append(
            {
                "page_content": doc,
                "metadata": meta or {},
            }
        )

    return chunks


def merge_chunks(chunks_a: list[dict], chunks_b: list[dict]) -> list[dict]:
    merged = []
    seen = set()

    for chunk in chunks_a + chunks_b:
        meta = chunk.get("metadata", {})
        key = (
            chunk.get("page_content", ""),
            tuple(sorted(meta.items())),
        )
        if key not in seen:
            merged.append(chunk)
            seen.add(key)

    return merged


def heuristic_score(question: str, chunk: dict) -> int:
    q = question.lower()
    text = chunk.get("page_content", "").lower()
    meta = chunk.get("metadata", {})

    source_name = str(meta.get("source_name", "")).lower().replace("_", " ").replace("-", " ")
    doc_type = str(meta.get("doc_type", "")).lower()

    score = 0

    for word in re.findall(r"\w+", q):
        if word in text:
            score += 2
        if word and word in source_name:
            score += 4

    if any(term in q for term in ["ceo", "cto", "cfo", "founder", "manager", "scientist", "engineer"]):
        if doc_type in {"employees", "employee", "people", "leadership", "team"}:
            score += 8

    if "ceo" in q and ("chief executive officer" in text or " ceo " in f" {text} "):
        score += 15

    if source_name and source_name in q:
        score += 20

    return score


def parse_ranked_ids(text: str, n_chunks: int) -> list[int]:
    """
    Accept loose formats like:
    [3,1,2]
    3,1,2
    3 1 2
    """
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            ids = [int(x) for x in parsed if isinstance(x, int) or str(x).isdigit()]
            ids = [i for i in ids if 1 <= i <= n_chunks]
            if ids:
                return ids
    except Exception:
        pass

    nums = re.findall(r"\d+", text)
    ids = []
    seen = set()

    for num in nums:
        i = int(num)
        if 1 <= i <= n_chunks and i not in seen:
            ids.append(i)
            seen.add(i)

    return ids


def rerank(question: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    pre_ranked = sorted(
        chunks,
        key=lambda c: heuristic_score(question, c),
        reverse=True,
    )

    user_prompt = f"The user question is:\n{question}\n\n"
    user_prompt += "Rank these chunks by relevance from most relevant to least relevant.\n"
    user_prompt += "Return only a Python-style list of chunk ids like [3,1,2,...]\n\n"

    for idx, chunk in enumerate(pre_ranked, start=1):
        user_prompt += f"# CHUNK ID: {idx}\n{chunk['page_content']}\n\n"

    try:
        reply = llm_text(
            [
                {
                    "role": "system",
                    "content": "You are a document reranker. Return only a list of chunk ids.",
                },
                {"role": "user", "content": user_prompt},
            ]
        )

        ranked_ids = parse_ranked_ids(reply, len(pre_ranked))

        if not ranked_ids:
            return pre_ranked

        reranked = [pre_ranked[i - 1] for i in ranked_ids]

        used = set(ranked_ids)
        for i, chunk in enumerate(pre_ranked, start=1):
            if i not in used:
                reranked.append(chunk)

        return reranked

    except Exception as e:
        print(f"[rerank fallback] {type(e).__name__}: {e}")
        return pre_ranked


def make_rag_messages(question: str, history: list[dict] | None, chunks: list[dict]) -> list[dict]:
    history = history or []

    context = "\n\n".join(
        f"Extract from {chunk['metadata'].get('source_name', chunk['metadata'].get('source', 'unknown'))}:\n{chunk['page_content']}"
        for chunk in chunks
    )

    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def fetch_context(question: str, history: list[dict] | None = None) -> list[dict]:
    rewritten_question = rewrite_query(question, history)

    chunks_original = fetch_context_unranked(question)
    chunks_rewritten = fetch_context_unranked(rewritten_question)

    merged = merge_chunks(chunks_original, chunks_rewritten)
    reranked = rerank(question, merged)

    return reranked[:FINAL_K]


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    chunks = fetch_context(question, history)

    if not chunks:
        return "I could not find that in the knowledge base.", []

    messages = make_rag_messages(question, history, chunks)

    try:
        answer = llm_text(messages)
    except Exception as e:
        print(f"[answer fallback] {type(e).__name__}: {e}")
        return "I could not find that in the knowledge base.", chunks

    if not answer:
        answer = "I could not find that in the knowledge base."

    return answer, chunks