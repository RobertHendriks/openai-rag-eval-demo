🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
 # Enterprise Q&A with RAG & Automated Evals
🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖

A lightweight RAG (Retrieval-Augmented Generation) pipeline with built-in 
evaluation framework, demonstrating patterns for LLM applications.

## Covers:

- **RAG Pipeline**: Document ingestion → embedding → semantic retrieval → grounded generation
- **Function Calling**: Structured output via OpenAI's tool use for reliable response formatting
- **Automated Evals**: 'LLM-as-judge evaluation framework with test evaluation suite and scoring
- **Trust & Safety Patterns**: Source attribution, confidence scoring, out-of-scope detection

## Architecture:
```
Documents → Chunking → Embeddings (text-embedding-3-small) → ChromaDB
                                                                  ↓
User Query → Query Embedding → Semantic Search → Context Assembly
                                                                  ↓
                                              GPT-4o-mini + Function Calling
                                                                  ↓
                                              Structured Answer + Sources
                                                                  ↓
                                              LLM-as-Judge Evaluation
```

## Quick Start:
```bash
git clone https://github.com/RobertHendriks/openai-rag-eval-demo.git
cd openai-rag-eval-demo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # ADD your OpenAI API key

python demo.py ingest    # Ingest sample documents
python demo.py           # Interactive Q&A
python demo.py eval_test      # Run test evaluation suite
```

## Eval Results:

The eval framework tests the pipeline against known question-answer pairs 
and uses GPT-4o-mini as an automated judge to score responses as 
CORRECT / PARTIALLY_CORRECT / INCORRECT.

Sample output:
```
RESULTS: 5/5 correct (100% accuracy)
```

## Tech Stack

- **OpenAI API**: GPT-4o-mini (generation + evaluation), text-embedding-3-small (embeddings)
- **ChromaDB**: Local vector store (lightweight, no infrastructure required)
- **Python**: Minimal dependencies, no frameworks - tested with Python 3.12*

## Possible Extensions

- Add chunking strategies for longer documents
- Implement hybrid search (semantic + keyword)
- Add prompt versioning and A/B comparison
- Integrate with OpenAI's fine-tuning API for style customization
- Add latency and cost tracking per query