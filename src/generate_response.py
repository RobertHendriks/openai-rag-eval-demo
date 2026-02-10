import json
from openai import OpenAI
from dotenv import load_dotenv
from src.retrieve_data import retrieve

load_dotenv()
client = OpenAI()

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

def generate_answer(question):
    # Retrieve relevant context
    context_docs = retrieve(question, n_results=2)
    context = "\n\n".join(
        f"[Source: {doc['title']}]\n{doc['content']}" for doc in context_docs
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant for Acme Analytics. "
                "Answer questions using ONLY the provided context. "
                "If the context doesn't contain enough information, say so honestly. "
                "Always use the format_answer function to structure your response."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "format_answer"}}
    )

    tool_call = response.choices[0].message.tool_calls[0]
    result = json.loads(tool_call.function.arguments)
    result["retrieved_docs"] = [doc["title"] for doc in context_docs]
    result["model"] = "gpt-4o-mini"

    return result

if __name__ == "__main__":
    question = "Can I get a refund on my annual plan?"
    result = generate_answer(question)
    print(json.dumps(result, indent=2))