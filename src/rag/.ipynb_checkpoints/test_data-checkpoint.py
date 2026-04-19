import json
from pydantic import BaseModel, Field

from src.rag.config import TEST_FILE


class TestQuestion(BaseModel):
    question: str = Field(description="Question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that should appear in retrieved context")
    reference_answer: str = Field(description="Reference answer")
    category: str = Field(description="Question category")


def load_tests() -> list[TestQuestion]:
    tests = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tests.append(TestQuestion(**data))

    return tests