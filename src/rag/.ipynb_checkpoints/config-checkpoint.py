from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge-base"
VECTOR_DB = PROJECT_ROOT / "data" / "vectorstores" / "preprocessed_db"
TEST_FILE = PROJECT_ROOT / "tests" / "tests.jsonl"
COLLECTION_NAME = "docs"

# Local Ollama model
OLLAMA_MODEL = "llama3.2:latest"

# Local embeddings
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Retrieval
RETRIEVAL_K = 20
FINAL_K = 10

# Ollama OpenAI-compatible endpoint
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"