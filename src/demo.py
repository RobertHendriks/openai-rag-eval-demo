from src.ingest_data import ingest
from src.generate_response import generate_answer
from src.evaluate_pipeline import run_evals
import json
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "eval_test":
        print("=" * 50)
        print("Running Evaluation Suite")
        print("=" * 50 + "\n")
        run_evals()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        print("Ingesting documents...\n")
        ingest()
        return

    # Interactive Q&A mode
    print("=" * 50)
    print("*" * 50)
    print("🤖-- Welcome to ACME Analytics Q&A (RAG Demo)--🤖" +"\n")
    print("Type 'quit' at any time to exit")
    print("*" * 50)
    print("=" * 50 + "\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = generate_answer(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {', '.join(result['sources_used'])}")
        if result.get("follow_up_suggestion"):
            print(f"Follow-up suggestion: {result['follow_up_suggestion']}")
        print()

if __name__ == "__main__":
    main()