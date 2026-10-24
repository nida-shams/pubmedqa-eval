"""Run a small PubMedQA evaluation against a local Ollama model.

Usage:
    python run_eval.py                 # 20 questions with llama3.2
    python run_eval.py --limit 3       # quick test on the first 3 questions
    python run_eval.py --model mistral # try a different model
"""

import argparse
import json
import random
from pathlib import Path

import ollama

DATA_PATH = Path("~/pubmedqa/data/ori_pqal.json").expanduser()
OUTPUT_PATH = Path("predictions.json")
NUM_QUESTIONS = 20
SEED = 42
VALID_ANSWERS = {"yes", "no", "maybe"}

INSTRUCTIONS = (
    "You will be given a research abstract and a question. "
    "Based only on the abstract, answer the question with exactly one word: "
    "yes, no, or maybe. "
    'Respond in JSON format: {"answer": "yes"}'
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Ollama model on PubMedQA.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N of the selected questions (e.g. 3 for a quick test).")
    parser.add_argument("--model", default="llama3.2",
                        help="Name of the Ollama model to use (default: llama3.2).")
    return parser.parse_args()


def load_dataset(path):
    with open(path) as f:
        return json.load(f)


def select_questions(data, n, seed):
    # Sort the IDs first so the selection doesn't depend on file order.
    rng = random.Random(seed)
    return rng.sample(sorted(data.keys()), n)


def build_prompt(abstract, question):
    return f"{INSTRUCTIONS}\n\nAbstract:\n{abstract}\n\nQuestion: {question}"


def ask_model(model, prompt):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
    )
    return response["message"]["content"]


def parse_answer(raw):
    """Return yes/no/maybe, or "format_error" if the reply isn't usable."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return "format_error"
    if not isinstance(parsed, dict):
        return "format_error"
    answer = str(parsed.get("answer", "")).strip().lower()
    return answer if answer in VALID_ANSWERS else "format_error"


def main():
    args = parse_args()

    data = load_dataset(DATA_PATH)
    question_ids = select_questions(data, NUM_QUESTIONS, SEED)
    if args.limit is not None:
        question_ids = question_ids[:args.limit]

    total = len(question_ids)
    print(f"Running {total} questions with model '{args.model}'...")

    results = []
    for i, qid in enumerate(question_ids, start=1):
        item = data[qid]
        abstract = " ".join(item["CONTEXTS"])
        prompt = build_prompt(abstract, item["QUESTION"])

        raw = ask_model(args.model, prompt)

        results.append({
            "id": qid,
            "gold_answer": item["final_decision"],
            "model_answer": parse_answer(raw),
            "raw_response": raw,
        })
        print(f"Question {i} of {total} done.")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
