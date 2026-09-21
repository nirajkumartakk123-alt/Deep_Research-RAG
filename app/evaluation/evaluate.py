"""
Runs the full evaluation: for every question in EVAL_DATASET, runs
every pipeline variant, computes retrieval metrics for all 5, and
generation metrics only for the two LLM-based variants (4-5). Writes
results to data/evaluation/results/ as timestamped JSON - never
computed live and thrown away, since the whole point of a real
evaluation framework is having actual numbers to put in a README,
not numbers regenerated on demand.

DELAY_BETWEEN_LLM_CALLS_SECONDS exists specifically to stay under
Groq's free-tier TPM limit (8000 tokens/minute) - see Phase 8-10
retrospective notes on repeatedly hitting 429s during heavy test runs.
This is a deliberate, simple pacing mechanism (Option A from the
Phase 10 design discussion) rather than a retry-with-backoff wrapper -
appropriate for a bounded, infrequent evaluation run rather than
continuous dev-loop testing.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.dataset import EVAL_DATASET
from app.evaluation.generation_metrics import evaluate_faithfulness, evaluate_relevancy
from app.evaluation.pipeline_variants import VARIANTS
from app.evaluation.retrieval_metrics import mrr, ndcg_at_k, recall_at_k

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_FOR_METRICS = 5
LLM_VARIANTS = {"agentic_no_correction", "corrective_agentic"}
DELAY_BETWEEN_LLM_CALLS_SECONDS = 3


async def _evaluate_one(variant_name: str, variant_fn, eval_question) -> dict:
    result = await variant_fn(eval_question.question)

    retrieval_scores = {
        "recall_at_5": recall_at_k(result.ranked_document_names, eval_question.expected_document_name, k=TOP_K_FOR_METRICS),
        "mrr": mrr(result.ranked_document_names, eval_question.expected_document_name),
        "ndcg_at_5": ndcg_at_k(result.ranked_document_names, eval_question.expected_document_name, k=TOP_K_FOR_METRICS),
    }

    generation_scores = None
    if variant_name in LLM_VARIANTS and result.generated_answer:
        await asyncio.sleep(DELAY_BETWEEN_LLM_CALLS_SECONDS)
        faithfulness = await evaluate_faithfulness(result.generated_answer, result.evidence_text or "")

        await asyncio.sleep(DELAY_BETWEEN_LLM_CALLS_SECONDS)
        relevancy = await evaluate_relevancy(eval_question.question, result.generated_answer)

        generation_scores = {
            "faithful": faithfulness.faithful,
            "unsupported_claims": faithfulness.unsupported_claims,
            "relevant": relevancy.relevant,
            "relevancy_score": relevancy.relevancy_score,
        }

    return {
        "question": eval_question.question,
        "expected_document": eval_question.expected_document_name,
        "ranked_documents": result.ranked_document_names[:TOP_K_FOR_METRICS],
        "retrieval_metrics": retrieval_scores,
        "generation_metrics": generation_scores,
        "generated_answer": result.generated_answer,
    }


async def run_evaluation() -> dict:
    all_results: dict = {}

    for variant_name, variant_fn in VARIANTS.items():
        print(f"\n=== Variant: {variant_name} ===")
        variant_results = []

        for eval_question in EVAL_DATASET:
            print(f"  Question: {eval_question.question[:60]}...")
            try:
                result = await _evaluate_one(variant_name, variant_fn, eval_question)
                variant_results.append(result)
            except Exception as exc:
                logger.error(f"Evaluation failed for variant={variant_name}, question={eval_question.question}: {exc}")
                variant_results.append(
                    {
                        "question": eval_question.question,
                        "error": str(exc),
                    }
                )

            if variant_name in LLM_VARIANTS:
                await asyncio.sleep(DELAY_BETWEEN_LLM_CALLS_SECONDS)

        # Aggregate retrieval metrics across all questions for this variant
        valid_results = [r for r in variant_results if "retrieval_metrics" in r]
        aggregated_retrieval = {
            "mean_recall_at_5": sum(r["retrieval_metrics"]["recall_at_5"] for r in valid_results) / len(valid_results) if valid_results else 0.0,
            "mean_mrr": sum(r["retrieval_metrics"]["mrr"] for r in valid_results) / len(valid_results) if valid_results else 0.0,
            "mean_ndcg_at_5": sum(r["retrieval_metrics"]["ndcg_at_5"] for r in valid_results) / len(valid_results) if valid_results else 0.0,
        }

        gen_results = [r for r in valid_results if r.get("generation_metrics")]
        aggregated_generation = None
        if gen_results:
            aggregated_generation = {
                "faithfulness_rate": sum(1 for r in gen_results if r["generation_metrics"]["faithful"]) / len(gen_results),
                "relevancy_rate": sum(1 for r in gen_results if r["generation_metrics"]["relevant"]) / len(gen_results),
                "mean_relevancy_score": sum(r["generation_metrics"]["relevancy_score"] for r in gen_results) / len(gen_results),
            }

        all_results[variant_name] = {
            "aggregated_retrieval_metrics": aggregated_retrieval,
            "aggregated_generation_metrics": aggregated_generation,
            "per_question_results": variant_results,
        }

        print(f"  Aggregated retrieval: {aggregated_retrieval}")
        if aggregated_generation:
            print(f"  Aggregated generation: {aggregated_generation}")

    return all_results


def save_results(results: dict) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    return output_path


async def main():
    results = await run_evaluation()
    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())