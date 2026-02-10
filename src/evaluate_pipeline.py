import json
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from src.generate_response import generate_answer

load_dotenv()
client = OpenAI()

# Test cases: question + expected grounded truth
EVAL_SUITE = [
    {
        "question": "How much does the Pro plan cost?",
        "expected": "The Pro plan costs $129 per month.",
        "category": "factual_retrieval"
    },
    {
        "question": "Can Lite plan users access the API?",
        "expected": "No, Lite plans do not include API access.",
        "category": "factual_retrieval"
    },
    {
        "question": "How long is data retained after cancellation?",
        "expected": "Data is retained for 90 days after cancellation.",
        "category": "factual_retrieval"
    },
    {
        "question": "Does ACME support Okta for SSO?",
        "expected": "Yes, Okta is a supported identity provider for SSO on Enterprise plans.",
        "category": "factual_retrieval"
    },
    {
        "question": "What is the CEO's favorite color?",
        "expected": "This information is not available in the knowledge base.",
        "category": "out_of_scope"
    }
]

def llm_judge(question, expected, actual):
    """Use GPT-4o-mini as an automated judge."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. Compare the actual answer to the expected answer. "
                    "Score as follows:\n"
                    "- CORRECT: The actual answer conveys the same essential information as expected.\n"
                    "- PARTIALLY_CORRECT: The answer is relevant but missing key details or adds inaccuracies.\n"
                    "- INCORRECT: The answer is wrong, hallucinated, or unrelated.\n\n"
                    "Respond with ONLY a JSON object: "
                    '{\"score\": \"CORRECT|PARTIALLY_CORRECT|INCORRECT\", \"reasoning\": \"brief explanation\"}'
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Expected answer: {expected}\n"
                    f"Actual answer: {actual}"
                )
            }
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def run_evals():
    results = []
    print("Running the evaluation suite...\n")

    for i, test in enumerate(EVAL_SUITE):
        print(f"[{i+1}/{len(EVAL_SUITE)}] {test['question']}")

        # Generate answer from the RAG pipeline
        answer_result = generate_answer(test["question"])

        # Judge the answer
        judgment = llm_judge(
            test["question"],
            test["expected"],
            answer_result["answer"]
        )

        result = {
            "question": test["question"],
            "category": test["category"],
            "expected": test["expected"],
            "actual": answer_result["answer"],
            "confidence": answer_result["confidence"],
            "sources_used": answer_result["sources_used"],
            "judgment": judgment["score"],
            "reasoning": judgment["reasoning"]
        }
        results.append(result)
        print(f"  → {judgment['score']} ({answer_result['confidence']} confidence)\n")

    # Summarize the results 
    scores = [r["judgment"] for r in results]
    summary = {
        "total": len(results),
        "correct": scores.count("CORRECT"),
        "partially_correct": scores.count("PARTIALLY_CORRECT"),
        "incorrect": scores.count("INCORRECT"),
        "accuracy": scores.count("CORRECT") / len(results)
    }

    output = {
        "run_timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": results
    }

    # Persist results as a JSON file
    os.makedirs("evaluation_results", exist_ok=True)
    filepath = f"evaluation_results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RESULTS: {summary['correct']}/{summary['total']} correct ({summary['accuracy']:.0%} accuracy)")
    print(f"Saved to {filepath}")

    return output

if __name__ == "__main__":
    run_evals()