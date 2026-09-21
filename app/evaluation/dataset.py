"""
Fixed evaluation dataset. Deliberately small (4 questions) - see
Phase 10 design notes: real, small-scale numbers are more defensible
than noisy numbers from a dataset sized to look impressive but never
actually validated against Groq's free-tier TPM budget.

Each question needs:
- question: the query text
- expected_document_name: which uploaded document should be the top
  retrieval hit - this is the ground truth Recall@K/MRR/NDCG are
  computed against
- ground_truth_answer: a short reference answer, used by the
  Faithfulness/Answer Relevancy LLM-judge metrics as a comparison point

IMPORTANT: before running evaluation, the exact documents referenced
here must be uploaded to a clean database - see evaluate.py's
setup instructions. Using a dirty/leftover corpus from earlier manual
testing will produce meaningless Recall@K numbers, since ground truth
here assumes ONLY these 4 documents exist.
"""
from dataclasses import dataclass


@dataclass
class EvalQuestion:
    question: str
    expected_document_name: str
    ground_truth_answer: str


EVAL_DATASET: list[EvalQuestion] = [
    EvalQuestion(
        question="According to my documents, what percentage of the world's oxygen does the Amazon rainforest produce?",
        expected_document_name="eval_amazon.txt",
        ground_truth_answer="The Amazon rainforest produces approximately 20% of the world's oxygen.",
    ),
    EvalQuestion(
        question="According to my documents, what year was the Eiffel Tower completed?",
        expected_document_name="eval_eiffel.txt",
        ground_truth_answer="The Eiffel Tower was completed in 1889.",
    ),
    EvalQuestion(
        question="According to my documents, how tall is Mount Kilimanjaro?",
        expected_document_name="eval_kilimanjaro.txt",
        ground_truth_answer="Mount Kilimanjaro stands at 5,895 meters, making it the highest mountain in Africa.",
    ),
    EvalQuestion(
        question="According to my documents, what is the boiling point of water at sea level?",
        expected_document_name="eval_water.txt",
        ground_truth_answer="Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level atmospheric pressure.",
    ),
]

EVAL_DOCUMENTS: dict[str, str] = {
    "eval_amazon.txt": "The Amazon rainforest produces approximately 20% of the world's oxygen and is home to millions of species of plants and animals.",
    "eval_eiffel.txt": "The Eiffel Tower was completed in 1889 for the World's Fair held in Paris, France.",
    "eval_kilimanjaro.txt": "Mount Kilimanjaro is the highest mountain in Africa, standing at 5,895 meters above sea level.",
    "eval_water.txt": "Water boils at 100 degrees Celsius, or 212 degrees Fahrenheit, at standard sea level atmospheric pressure.",
}