import math
import csv
from src.rag.answer import fetch_context_unranked
from src.rag.test_data import TestQuestion, load_tests


def get_page_content(doc: dict) -> str:
    return str(doc.get("page_content", ""))


def calculate_mrr(keyword: str, retrieved_docs: list[dict]) -> float:
    keyword_lower = keyword.lower()

    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in get_page_content(doc).lower():
            return 1.0 / rank

    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list[dict], k: int = 5) -> float:
    keyword_lower = keyword.lower()

    relevances = [
        1 if keyword_lower in get_page_content(doc).lower() else 0
        for doc in retrieved_docs[:k]
    ]

    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 5) -> dict:
    retrieved_docs = fetch_context_unranked(test.question)

    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]

    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    keywords_found = sum(1 for score in mrr_scores if score > 0)

    return {
        "question": test.question,
        "category": test.category,
        "mrr": round(avg_mrr, 4),
        "ndcg": round(avg_ndcg, 4),
        "keywords_found": keywords_found,
        "total_keywords": len(test.keywords),
    }


def run_all_retrieval_tests() -> list[dict]:
    tests = load_tests()
    results = []

    for test in tests:
        print(f"Running: {test.question}")
        results.append(evaluate_retrieval(test))

    return results


if __name__ == "__main__":
    results = run_all_retrieval_tests()
    for r in results:
        print(r)
        csv_file = "eval_results.csv"
    keys = results[0].keys()

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {csv_file}")