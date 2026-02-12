import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.retrieve import retrieve
from src.prompt_registry import get_prompt
from src.query_logger import log_query, build_log_entry

load_dotenv()
client = OpenAI()

# --- Configuration ---
# Change the model here to swap between models globally.
DEFAULT_MODEL = "gpt-4o-mini"

# Define a function the model can call to structure its answer
tools = [
    {
        "type": "function",
        "function": {
            "name": "format_answer",
            "description": "Format a structured answer with source attribution and confidence level",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The direct answer to the user's question"
                    },
                    "sources_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of source document titles used to form the answer"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence level based on how well the sources cover the question"
                    },
                    "follow_up_suggestion": {
                        "type": "string",
                        "description": "A suggested follow-up question the user might want to ask"
                    }
                },
                "required": ["answer", "sources_used", "confidence"]
            }
        }
    }
]


def generate_answer(question, prompt_version=None, model=None):
    """
    Generate a RAG-grounded answer using the OpenAI API.

    Args:
        question: The user's question
        prompt_version: Optional prompt version string. If None, uses latest.
        model: Optional model override. If None, uses DEFAULT_MODEL.

    Returns:
        dict with answer, sources, confidence, and metadata
    """
    active_model = model or DEFAULT_MODEL
    start_time = time.time()

    # Load system prompt from registry
    prompt_config = get_prompt("customer_support", version=prompt_version)
    system_prompt = prompt_config["system_prompt"]
    active_version = str(prompt_config["version"])

    # Retrieve relevant context
    context_docs = retrieve(question, n_results=2)
    context = "\n\n".join(
        f"[Source: {doc['title']}]\n{doc['content']}" for doc in context_docs
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=active_model,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "format_answer"}}
    )

    tool_call = response.choices[0].message.tool_calls[0]
    result = json.loads(tool_call.function.arguments)
    result["retrieved_docs"] = [doc["title"] for doc in context_docs]
    result["model"] = active_model
    result["prompt_version"] = active_version

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Log the interaction
    log_entry = build_log_entry(
        question=question,
        retrieved_docs=context_docs,
        answer_result=result,
        prompt_version=active_version,
        latency_ms=round(latency_ms, 1),
    )
    log_query(log_entry)

    result["latency_ms"] = round(latency_ms, 1)

    return result


if __name__ == "__main__":
    question = "Can I get a refund on my annual plan?"
    result = generate_answer(question)
    print(json.dumps(result, indent=2))